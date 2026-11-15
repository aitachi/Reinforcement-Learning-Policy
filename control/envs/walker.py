"""E5 - planar biped walker ("WalkerLite"), whole-body gait coordination.

Generalises E2's monopod to two legs sharing one torso: floating base
(x, y, pitch) + left/right thigh (hip) + shank (knee) -> 7 DOF,
underactuated (torques on the four hip/knee joints only).  Walking requires
inter-leg coordination: stance-leg support, swing-leg placement and torso
pitch balance must be learned jointly - the whole-body coordination axis
the monopod cannot test.

Dynamics follow E2: analytic per-body Jacobians
M(q) = sum_i m_i J_vi^T J_vi + I_i J_wi^T J_wi, generalized gravity,
penalty foot contact per foot, weak series-elastic knee springs
(k=10, rest -0.55) so double-support standing is passively stable,
semi-implicit Euler at 200 Hz (4 substeps per 50 Hz control step).

q = [x, y, phi, hipL, kneeL, hipR, kneeR]

Obs (23): phi(1), sin/cos of 4 joints (8), qd(7), torso_y(1),
          footL rel(2), footR rel(2), contactL(1), contactR(1).
Signals: vx, alive, fallen, effort, height, contact (any), timeout.
Episode: 250 steps; success = displace 3 m; outcome = displacement / 10.
"""
from __future__ import annotations

import torch

G = 9.81
L_T, L_H, L_S = 0.25, 0.20, 0.20
M_T, M_H, M_S = 3.0, 0.7, 0.35
I_T = M_T * 0.5 ** 2 / 12
I_H = M_H * 0.4 ** 2 / 12
I_S = M_S * 0.4 ** 2 / 12
K_CONTACT, D_CONTACT, MU = 3500.0, 120.0, 0.9
DAMP = torch.tensor([0.05, 0.05, 0.15, 0.08, 1.5, 0.08, 1.5])
KNEE_SPRING, KNEE_REST = 10.0, -0.55
HIP_LIM, KNEE_LIM = 1.4, 2.35
TAU_MAX = 60.0
DT = 0.02
SUBSTEPS = 4


class WalkerEnv:
    obs_dim = 23
    act_dim = 4
    signal_keys = ("vx", "alive", "fallen", "effort", "height", "contact", "timeout")

    def __init__(self, num_envs, seed, device, max_steps=250, target_speed=0.9):
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.max_steps = max_steps
        self.target_speed = target_speed
        self._progress = 0.0
        self.q = torch.zeros(num_envs, 7, device=device)
        self.qd = torch.zeros(num_envs, 7, device=device)
        self.x0 = torch.zeros(num_envs, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.fallen_flag = torch.zeros(num_envs, device=device)
        self.contactL = torch.zeros(num_envs, device=device)
        self.contactR = torch.zeros(num_envs, device=device)
        B = num_envs
        self._eye7 = torch.eye(7, device=device)
        J0 = torch.zeros(2, 7, device=device)
        J0[0, 0] = 1.0
        J0[1, 1] = 1.0
        self._Jbase = J0.expand(B, 2, 7).contiguous()
        self._JwT = self._eye7[2]                                  # torso omega row
        self._JwH = [self._eye7[2] + self._eye7[3],                # thigh L
                     self._eye7[2] + self._eye7[5]]                # thigh R
        self._JwS = [self._eye7[2] + self._eye7[3] + self._eye7[4],
                     self._eye7[2] + self._eye7[5] + self._eye7[6]]
        self._JwTo = self._JwT.expand(B, 7).unsqueeze(-1)
        self._JwHo = [j.expand(B, 7).unsqueeze(-1) for j in self._JwH]
        self._JwSo = [j.expand(B, 7).unsqueeze(-1) for j in self._JwS]
        self._damp = DAMP.to(device)
        self._lo = torch.tensor([-9.0, 0.15, -0.9, -HIP_LIM, -KNEE_LIM, -HIP_LIM, -KNEE_LIM], device=device)
        self._hi = torch.tensor([9.0, 9.0, 0.9, HIP_LIM, 0.05, HIP_LIM, 0.05], device=device)

    def set_progress(self, p):
        self._progress = float(min(max(p, 0.0), 1.0))

    # ---------------------------------------------------------------- setup
    def _spawn(self, mask):
        n = int(mask.sum())
        if n == 0:
            return
        dev = self.device
        hip = 0.32 + 0.05 * torch.randn(n, generator=self.gen, device=dev)
        knee = -0.55 + 0.05 * torch.randn(n, generator=self.gen, device=dev)
        phi = 0.08 * torch.randn(n, generator=self.gen, device=dev)
        t1, t2 = phi + hip, phi + hip + knee
        y = L_T * phi.cos() + L_H * t1.cos() + L_S * t2.cos()   # feet on ground
        q = torch.zeros(n, 7, device=dev)
        q[:, 0] = 0.2 * torch.randn(n, generator=self.gen, device=dev)
        q[:, 1] = y + 0.005
        q[:, 2] = phi
        q[:, 3] = hip
        q[:, 4] = knee
        q[:, 5] = hip + 0.03 * torch.randn(n, generator=self.gen, device=dev)
        q[:, 6] = knee + 0.03 * torch.randn(n, generator=self.gen, device=dev)
        self.q[mask] = q
        self.qd[mask] = 0.05 * torch.randn(n, 7, generator=self.gen, device=dev)
        self.qd[mask, 0] = 0.2 * torch.randn(n, generator=self.gen, device=dev)
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
        phi = q[:, 2]
        sp, cp = phi.sin(), phi.cos()
        hipx = torch.stack([q[:, 0] + L_T * sp, q[:, 1] - L_T * cp], -1)      # shared hip
        legs = []
        for hi, ki in ((3, 4), (5, 6)):
            t1 = phi + q[:, hi]
            t2 = t1 + q[:, ki]
            s1, c1 = t1.sin(), t1.cos()
            s2, c2 = t2.sin(), t2.cos()
            knee_p = torch.stack([hipx[:, 0] + L_H * s1, hipx[:, 1] - L_H * c1], -1)
            foot_p = torch.stack([knee_p[:, 0] + L_S * s2, knee_p[:, 1] - L_S * c2], -1)
            mid_thigh = torch.stack([hipx[:, 0] + 0.5 * L_H * s1, hipx[:, 1] - 0.5 * L_H * c1], -1)
            mid_shank = torch.stack([knee_p[:, 0] + 0.5 * L_S * s2, knee_p[:, 1] - 0.5 * L_S * c2], -1)
            legs.append((s1, c1, s2, c2, mid_thigh, mid_shank, foot_p))
        return sp, cp, hipx, legs

    def _dynamics(self, kin, q, qd, tau, contactL, contactR):
        sp, cp, hipx, legs = kin
        B = q.shape[0]
        dev = self.device
        M = (M_T * (self._Jbase.transpose(1, 2) @ self._Jbase)
             + I_T * (self._JwTo @ self._JwTo.transpose(1, 2)))
        g7 = torch.zeros(B, 7, device=dev)
        g7[:, 1] = G * (M_T + 2.0 * (M_H + M_S))
        g2phi = torch.zeros(B, device=dev)
        vT = torch.stack([L_T * cp, L_T * sp], -1)                 # d(hip)/d(phi)
        for i, (hi, ki) in enumerate(((3, 4), (5, 6))):
            s1, c1, s2, c2, mid_thigh, mid_shank, foot_p = legs[i]
            v1h = torch.stack([0.5 * L_H * c1, 0.5 * L_H * s1], -1)
            v1f = torch.stack([L_H * c1, L_H * s1], -1)
            v2h = torch.stack([0.5 * L_S * c2, 0.5 * L_S * s2], -1)
            v2f = torch.stack([L_S * c2, L_S * s2], -1)

            Jt = self._Jbase.clone(); Jt[:, :, 2] += vT + v1h; Jt[:, :, hi] += v1h
            Jk = self._Jbase.clone(); Jk[:, :, 2] += vT + v1f; Jk[:, :, hi] += v1f
            Js = Jk.clone();  Js[:, :, 2] += v2h; Js[:, :, hi] += v2h; Js[:, :, ki] += v2h
            Jf = Jk.clone();  Jf[:, :, 2] += v2f; Jf[:, :, hi] += v2f; Jf[:, :, ki] += v2f

            M = M + M_H * (Jt.transpose(1, 2) @ Jt) + I_H * (self._JwHo[i] @ self._JwHo[i].transpose(1, 2)) \
                    + M_S * (Js.transpose(1, 2) @ Js) + I_S * (self._JwSo[i] @ self._JwSo[i].transpose(1, 2))

            g2phi += G * (M_H * (L_T * sp + 0.5 * L_H * s1)
                          + M_S * (L_T * sp + L_H * s1 + 0.5 * L_S * s2))
            g7[:, hi] += G * (0.5 * M_H * L_H * s1 + M_S * (L_H * s1 + 0.5 * L_S * s2))
            g7[:, ki] += G * (0.5 * M_S * L_S * s2)

            # foot contact forces enter through the foot Jacobian
            cf = contactL if i == 0 else contactR
            fc = torch.einsum("bfi,bf->bi", Jf, cf)
            if i == 0:
                rhs_contact = fc
            else:
                rhs_contact = rhs_contact + fc

        M = 0.5 * (M + M.transpose(1, 2))
        M[:, range(7), range(7)] += 1e-6
        g7[:, 2] = g2phi
        tau7 = torch.zeros(B, 7, device=dev)
        tau7[:, 3] = tau[:, 0]
        tau7[:, 4] = tau[:, 1] - KNEE_SPRING * (q[:, 4] - KNEE_REST)
        tau7[:, 5] = tau[:, 2]
        tau7[:, 6] = tau[:, 3] - KNEE_SPRING * (q[:, 6] - KNEE_REST)
        rhs = tau7 - g7 - self._damp * qd + rhs_contact
        return torch.linalg.solve(M, rhs)

    def _foot_jac(self, kin):
        sp, cp, hipx, legs = kin
        vT = torch.stack([L_T * cp, L_T * sp], -1)
        Js = []
        for i, (hi, ki) in enumerate(((3, 4), (5, 6))):
            s1, c1, s2, c2, *_ = legs[i]
            v1 = torch.stack([L_H * c1, L_H * s1], -1)
            v2 = torch.stack([L_S * c2, L_S * s2], -1)
            J = self._Jbase.clone()
            J[:, :, 2] += vT + v1 + v2
            J[:, :, hi] += v1 + v2
            J[:, :, ki] += v2
            Js.append(J)
        return Js

    # ----------------------------------------------------------------- step
    def step(self, action):
        a = action.clamp(-1, 1)
        tau = a * TAU_MAX
        dt = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            kin = self._kin(self.q)
            JL, JR = self._foot_jac(kin)
            foots = [kin[3][0][6], kin[3][1][6]]
            fs = []
            for J, foot in ((JL, foots[0]), (JR, foots[1])):
                foot_v = torch.einsum("bfi,bi->bf", J, self.qd)
                pen = (-foot[:, 1]).clamp_min(0)
                fn = (K_CONTACT * pen - D_CONTACT * foot_v[:, 1] * (foot[:, 1] < 0).float()).clamp_min(0)
                ft = -MU * fn * torch.tanh(foot_v[:, 0] / 0.05)
                fs.append(torch.stack([ft, fn], -1))
            self.contactL = (fs[0][:, 1] > 1.0).float()
            self.contactR = (fs[1][:, 1] > 1.0).float()
            qdd = self._dynamics(kin, self.q, self.qd, tau, fs[0], fs[1])
            self.qd = (self.qd + qdd * dt).clamp(-12, 12)
            self.qd = torch.nan_to_num(self.qd, nan=0.0, posinf=0.0, neginf=0.0)
            self.q = self.q + self.qd * dt
            hit = (self.q < self._lo) | (self.q > self._hi)
            self.q = self.q.clamp_min(self._lo).clamp_max(self._hi)
            self.qd = torch.where(hit, -self.qd * 0.3, self.qd)
        self.steps += 1

        height = self.q[:, 1]
        fallen = ((height < 0.30) | (self.q[:, 2].abs() > 0.9)).float()
        self.fallen_flag = torch.maximum(self.fallen_flag, fallen)
        timeout = (self.steps >= self.max_steps).float()
        done_b = (fallen > 0.5) | (timeout > 0.5)
        any_contact = ((self.contactL + self.contactR) > 0).float()

        signals = {
            "vx": self.qd[:, 0], "alive": 1.0 - fallen, "fallen": fallen,
            "effort": (a * a).sum(-1), "height": height,
            "contact": any_contact, "timeout": timeout,
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
        kin = self._kin(self.q)
        sp, cp, hipx, legs = kin
        fL, fR = legs[0][6], legs[1][6]
        relL = torch.stack([fL[:, 0] - self.q[:, 0], fL[:, 1] - self.q[:, 1]], -1)
        relR = torch.stack([fR[:, 0] - self.q[:, 0], fR[:, 1] - self.q[:, 1]], -1)
        return torch.cat([self.q[:, 2:3],
                          self.q[:, 3:4].sin(), self.q[:, 3:4].cos(),
                          self.q[:, 4:5].sin(), self.q[:, 4:5].cos(),
                          self.q[:, 5:6].sin(), self.q[:, 5:6].cos(),
                          self.q[:, 6:7].sin(), self.q[:, 6:7].cos(),
                          self.qd / 6.0,
                          self.q[:, 1:2] - 0.60, relL, relR,
                          self.contactL.unsqueeze(-1), self.contactR.unsqueeze(-1)], -1)
