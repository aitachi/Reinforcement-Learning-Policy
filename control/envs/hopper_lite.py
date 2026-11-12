"""E2 - planar monopod hopper ("HopperLite"), custom rigid-body physics.

Floating-base planar articulated body: torso (free x, y, pitch) + thigh
(revolute hip) + shank (revolute knee) -> 5 DOF, underactuated (torques on
hip and knee only, like real hoppers).  Dynamics assembled from analytic
per-body Jacobians:  M(q) = sum_i m_i J_vi^T J_vi + I_i J_wi^T J_wi,
generalized gravity from the potential gradient, penalty-based foot contact
(spring-damper normal force + smooth Coulomb friction), semi-implicit Euler
at 200 Hz with 4 substeps per 50 Hz control step.

Geometry convention (phi = absolute torso pitch, 0 upright):
  hip     H = C0 + 0.25*(sin phi, -cos phi)
  knee    K = H  + 0.20*(sin t1, -cos t1),          t1 = phi + q_hip
  foot    F = K  + 0.20*(sin t2, -cos t2),          t2 = t1 + q_knee

Obs (14): [phi, sin/cos hip, sin/cos knee, qd(5), torso_y, foot_rel(2), contact].
Signals: vx (forward speed), alive, fallen, effort, height, contact, timeout.
"""
from __future__ import annotations

import torch

G = 9.81
L_T, L_H, L_S = 0.25, 0.20, 0.20          # half lengths: torso hip offset, thigh, shank
M_T, M_H, M_S = 3.0, 1.0, 0.5
I_T = M_T * 0.5 ** 2 / 12
I_H = M_H * 0.4 ** 2 / 12
I_S = M_S * 0.4 ** 2 / 12
K_CONTACT, D_CONTACT, MU = 3500.0, 120.0, 0.9
DAMP = torch.tensor([0.05, 0.05, 0.15, 1.8, 2.2])
# passive torsion spring at the knee (series-elastic hopper design, cf. Raibert):
# stores landing energy and holds the standing pose against knee buckling.
KNEE_SPRING, KNEE_REST = 18.0, -0.70
HIP_LIM, KNEE_LIM = 1.4, 2.35             # knee in [-KNEE_LIM, 0.05]
TAU_MAX = 90.0
DT = 0.02
SUBSTEPS = 4


class HopperLiteEnv:
    obs_dim = 14
    act_dim = 2
    signal_keys = ("vx", "alive", "fallen", "effort", "height", "contact", "timeout")

    def __init__(self, num_envs, seed, device, max_steps=250, target_speed=1.2):
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.max_steps = max_steps
        self.target_speed = target_speed
        self._progress = 0.0
        self.q = torch.zeros(num_envs, 5, device=device)
        self.qd = torch.zeros(num_envs, 5, device=device)
        self.x0 = torch.zeros(num_envs, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.fallen_flag = torch.zeros(num_envs, device=device)
        self.contact_flag = torch.zeros(num_envs, device=device)
        # precomputed constants (kernel-launch economy)
        B = num_envs
        self._eye5 = torch.eye(5, device=device)
        J0 = torch.zeros(2, 5, device=device)
        J0[0, 0] = 1.0
        J0[1, 1] = 1.0
        self._J0 = J0.expand(B, 2, 5).contiguous()
        self._Jw0 = self._eye5[2].expand(B, 5)
        self._Jw1 = (self._eye5[2] + self._eye5[3]).expand(B, 5)
        self._Jw2 = (self._eye5[2] + self._eye5[3] + self._eye5[4]).expand(B, 5)
        self._Jw0o = self._Jw0.unsqueeze(-1)
        self._Jw1o = self._Jw1.unsqueeze(-1)
        self._Jw2o = self._Jw2.unsqueeze(-1)
        self._Jc0 = J0.expand(B, 2, 5).contiguous()
        self._damp = DAMP.to(device)
        self._lo = torch.tensor([-9.0, 0.15, -0.9, -HIP_LIM, -KNEE_LIM], device=device)
        self._hi = torch.tensor([9.0, 9.0, 0.9, HIP_LIM, 0.05], device=device)

    def set_progress(self, p):
        self._progress = float(min(max(p, 0.0), 1.0))

    # ---------------------------------------------------------------- setup
    def _spawn(self, mask):
        n = int(mask.sum())
        if n == 0:
            return
        # quasi-static standing pose: foot under the CoM, knee at spring rest
        hip = 0.35 + 0.06 * torch.randn(n, generator=self.gen, device=self.device)
        knee = -0.70 + 0.06 * torch.randn(n, generator=self.gen, device=self.device)
        t1, t2 = hip, hip + knee
        y = L_T + L_H * t1.cos() + L_S * t2.cos()          # foot exactly on ground
        q = torch.zeros(n, 5, device=self.device)
        q[:, 2] = 0.08 * torch.randn(n, generator=self.gen, device=self.device)  # pitch
        q[:, 3] = hip
        q[:, 4] = knee
        q[:, 1] = y + 0.005                                  # tiny drop to seat contact
        q[:, 0] = 0.2 * torch.randn(n, generator=self.gen, device=self.device)
        self.q[mask] = q
        self.qd[mask] = 0.05 * torch.randn(n, 5, generator=self.gen, device=self.device)
        self.qd[mask, 0] = 0.3 * torch.randn(n, generator=self.gen, device=self.device)
        self.x0[mask] = q[:, 0]
        self.steps[mask] = 0
        self.fallen_flag[mask] = 0.0

    def observe(self):
        return self._observe()

    def reset(self):
        self._spawn(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._observe()

    # ------------------------------------------------------------ kinematics
    def _kin(self, q):
        phi, hip, knee = q[:, 2], q[:, 3], q[:, 4]
        t1 = phi + hip
        t2 = t1 + knee
        sp, cp = phi.sin(), phi.cos()
        s1, c1 = t1.sin(), t1.cos()
        s2, c2 = t2.sin(), t2.cos()
        hip_p = torch.stack([q[:, 0] + L_T * sp, q[:, 1] - L_T * cp], -1)
        knee_p = torch.stack([hip_p[:, 0] + L_H * s1, hip_p[:, 1] - L_H * c1], -1)
        foot_p = torch.stack([knee_p[:, 0] + L_S * s2, knee_p[:, 1] - L_S * c2], -1)
        mid_thigh = torch.stack([hip_p[:, 0] + 0.5 * L_H * s1, hip_p[:, 1] - 0.5 * L_H * c1], -1)
        mid_shank = torch.stack([knee_p[:, 0] + 0.5 * L_S * s2, knee_p[:, 1] - 0.5 * L_S * c2], -1)
        return sp, cp, s1, c1, s2, c2, mid_thigh, mid_shank, foot_p

    def _dynamics(self, kin, q, qd, tau, contact_f):
        """kin = precomputed _kin tuple; tau (B,2); contact_f (B,2) -> qdd (B,5)."""
        sp, cp, s1, c1, s2, c2, mid_thigh, mid_shank, foot_p = kin
        B = q.shape[0]

        vT = torch.stack([L_T * cp, L_T * sp], -1)
        v1 = torch.stack([0.5 * L_H * c1, 0.5 * L_H * s1], -1)
        v2f = torch.stack([L_H * c1, L_H * s1], -1)
        v2 = torch.stack([0.5 * L_S * c2, 0.5 * L_S * s2], -1)
        v3 = torch.stack([L_S * c2, L_S * s2], -1)

        Jh = self._Jc0.clone(); Jh[:, :, 2] += vT
        Jt = Jh.clone(); Jt[:, :, 2] += v1; Jt[:, :, 3] += v1
        Jk = Jh.clone(); Jk[:, :, 2] += v2f; Jk[:, :, 3] += v2f
        Js = Jk.clone(); Js[:, :, 2] += v2; Js[:, :, 3] += v2; Js[:, :, 4] += v2
        Jf = Jk.clone(); Jf[:, :, 2] += v3; Jf[:, :, 3] += v3; Jf[:, :, 4] += v3

        M = (M_T * (self._Jc0.transpose(1, 2) @ self._Jc0) + I_T * (self._Jw0o @ self._Jw0o.transpose(1, 2))
             + M_H * (Jt.transpose(1, 2) @ Jt) + I_H * (self._Jw1o @ self._Jw1o.transpose(1, 2))
             + M_S * (Js.transpose(1, 2) @ Js) + I_S * (self._Jw2o @ self._Jw2o.transpose(1, 2)))
        M = 0.5 * (M + M.transpose(1, 2))
        M[:, range(5), range(5)] += 1e-6

        g5 = torch.zeros(B, 5, device=self.device)
        g5[:, 1] = G * (M_T + M_H + M_S)
        g5[:, 2] = G * (M_H * (L_T * sp + 0.5 * L_H * s1) + M_S * (L_T * sp + L_H * s1 + 0.5 * L_S * s2))
        g5[:, 3] = G * (M_H * 0.5 * L_H * s1 + M_S * (L_H * s1 + 0.5 * L_S * s2))
        g5[:, 4] = G * (M_S * 0.5 * L_S * s2)

        tau5 = torch.zeros(B, 5, device=self.device)
        tau5[:, 3] = tau[:, 0]
        tau5[:, 4] = tau[:, 1] - KNEE_SPRING * (q[:, 4] - KNEE_REST)
        rhs = tau5 - g5 - self._damp * qd + torch.einsum("bfi,bf->bi", Jf, contact_f)
        return torch.linalg.solve(M, rhs)

    def _foot_jac(self, kin):
        sp, cp, s1, c1, s2, c2, *_ = kin
        v1 = torch.stack([L_H * c1, L_H * s1], -1)
        v2 = torch.stack([L_S * c2, L_S * s2], -1)
        J = self._Jc0.clone()
        J[:, :, 2] += torch.stack([L_T * cp, L_T * sp], -1) + v1 + v2
        J[:, :, 3] += v1 + v2
        J[:, :, 4] += v2
        return J

    # ----------------------------------------------------------------- step
    def step(self, action):
        a = action.clamp(-1, 1)
        tau = a * TAU_MAX
        dt = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            kin = self._kin(self.q)
            foot_y = kin[8][:, 1]
            Jf = self._foot_jac(kin)
            foot_v = torch.einsum("bfi,bi->bf", Jf, self.qd)
            pen = (-foot_y).clamp_min(0)
            fn = (K_CONTACT * pen - D_CONTACT * foot_v[:, 1] * (foot_y < 0).float()).clamp_min(0)
            ft = -MU * fn * torch.tanh(foot_v[:, 0] / 0.05)
            contact_f = torch.stack([ft, fn], -1)
            self.contact_flag = (fn > 1.0).float()
            qdd = self._dynamics(kin, self.q, self.qd, tau, contact_f)
            self.qd = (self.qd + qdd * dt).clamp(-12, 12)
            self.qd = torch.nan_to_num(self.qd, nan=0.0, posinf=0.0, neginf=0.0)
            self.q = self.q + self.qd * dt
            # vectorized hard joint limits with damped velocity reflection
            hit = (self.q < self._lo) | (self.q > self._hi)
            self.q = self.q.clamp_min(self._lo).clamp_max(self._hi)
            self.qd = torch.where(hit, -self.qd * 0.3, self.qd)
        self.steps += 1

        foot_y = self._kin(self.q)[8][:, 1]
        height = self.q[:, 1]
        fallen = ((height < 0.30) | (self.q[:, 2].abs() > 0.9)).float()
        self.fallen_flag = torch.maximum(self.fallen_flag, fallen)
        timeout = (self.steps >= self.max_steps).float()
        done_b = (fallen > 0.5) | (timeout > 0.5)

        signals = {
            "vx": self.qd[:, 0], "alive": 1.0 - fallen, "fallen": fallen,
            "effort": (a * a).sum(-1), "height": height,
            "contact": self.contact_flag, "timeout": timeout,
        }
        info = {"done": done_b.clone(),
                "episode_outcome": (self.q[:, 0] - self.x0) / 10.0,
                "episode_x": self.q[:, 0] - self.x0,
                "episode_success": ((self.q[:, 0] - self.x0 > 3.0).float() * done_b.float())}
        return self._observe(), signals, done_b, info

    # ----------------------------------------------------- episode control
    def reset_done(self, done):
        self._spawn(done)
        return self._observe()

    def initial_signals(self):
        z = torch.zeros(self.num_envs, device=self.device)
        return {"vx": self.qd[:, 0].clone(), "alive": torch.ones_like(z),
                "fallen": z.clone(), "effort": z.clone(),
                "height": self.q[:, 1].clone(), "contact": z.clone(), "timeout": z.clone()}

    _STATE_FIELDS = ("q", "qd", "x0", "steps", "fallen_flag")

    def get_state(self):
        return {f: getattr(self, f).clone() for f in self._STATE_FIELDS}

    def set_state(self, state):
        for f, v in state.items():
            getattr(self, f)[...] = v

    def _observe(self):
        sp, cp, s1, c1, s2, c2, _, _, foot_p = self._kin(self.q)
        rel = torch.stack([foot_p[:, 0] - self.q[:, 0], foot_p[:, 1] - self.q[:, 1]], -1)
        return torch.cat([self.q[:, 2:3], self.q[:, 3:4].sin(), self.q[:, 3:4].cos(),
                          self.q[:, 4:5].sin(), self.q[:, 4:5].cos(),
                          self.qd / 6.0,
                          self.q[:, 1:2] - 0.55, rel, self.contact_flag.unsqueeze(-1)], -1)
