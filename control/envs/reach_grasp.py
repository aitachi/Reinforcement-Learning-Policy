"""E3 - planar 3-link arm reach-grasp-lift ("ReachGrasp").

A velocity-controlled 3-link arm (as common on real industrial manipulators)
hangs from a base at (0, 0.65) above a table plane y=0.  The agent commands
3 joint velocities plus a gripper open/close channel.  An object rests on the
table; grasping is modeled kinematically: the object attaches when the
gripper tip is close to it while closing, and detaches when reopening.
Free-flight (gravity + bounce) applies once released.  Success = holding the
object above `LIFT_H` for 3 consecutive steps.

Obs (16): joint sin/cos(6), qd(3), gripper(1), obj rel EE(2), obj vel(2),
grasped(1), obj dist(1).
Signals: dist_ee_obj, grasp, height, lift_success, drop, effort, timeout.
"""
from __future__ import annotations

import torch

G = 9.81
L1, L2, L3 = 0.34, 0.30, 0.14
BASE_Y = 0.65
OBJ_HALF = 0.03
LIFT_H = 0.15
DT = 0.05                     # 20 Hz control
QD_MAX = 2.4                  # rad/s
JLIM = torch.tensor([1.3, 2.2, 2.0])


class ReachGraspEnv:
    obs_dim = 16
    act_dim = 4
    signal_keys = ("dist_ee_obj", "grasp", "height", "lift_success", "drop", "effort", "timeout")

    def __init__(self, num_envs, seed, device, max_steps=160):
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.max_steps = max_steps
        self._progress = 0.0
        self.q = torch.zeros(num_envs, 3, device=device)
        self.qd = torch.zeros(num_envs, 3, device=device)
        self.grip = torch.ones(num_envs, device=device)          # 1 open .. 0 closed
        self.obj = torch.zeros(num_envs, 2, device=self.device)  # (x, y)
        self.obj_v = torch.zeros(num_envs, 2, device=self.device)
        self.grasped = torch.zeros(num_envs, device=self.device)
        self.lift_hold = torch.zeros(num_envs, device=self.device)
        self.was_grasped = torch.zeros(num_envs, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def set_progress(self, p):
        self._progress = float(min(max(p, 0.0), 1.0))

    # ------------------------------------------------------------ kinematics
    def _ee(self):
        t1 = self.q[:, 0]
        t2 = t1 + self.q[:, 1]
        t3 = t2 + self.q[:, 2]
        x = L1 * t1.sin() + L2 * t2.sin() + L3 * t3.sin()
        y = BASE_Y - (L1 * t1.cos() + L2 * t2.cos() + L3 * t3.cos())
        return torch.stack([x, y], -1)

    # ----------------------------------------------------------------- spawn
    def _spawn(self, mask):
        n = int(mask.sum())
        if n == 0:
            return
        dev = self.device
        # curriculum: object spawn corridor widens with progress
        x_hi = 0.18 + 0.24 * self._progress
        reach = (L1 + L2 + L3) * 0.97
        x_max = (reach ** 2 - (BASE_Y - OBJ_HALF) ** 2) ** 0.5
        x_hi = min(x_hi, 0.9 * x_max)
        x = -x_hi + 2 * x_hi * torch.rand(n, generator=self.gen, device=dev)
        q0 = torch.zeros(n, 3, device=dev)
        q0[:, 0] = x * 0.9 + 0.05 * torch.randn(n, generator=self.gen, device=dev)
        q0[:, 1] = -0.9 + 0.15 * torch.randn(n, generator=self.gen, device=dev)
        q0[:, 2] = 0.6 + 0.15 * torch.randn(n, generator=self.gen, device=dev)
        self.q[mask] = q0.clamp(-JLIM.to(dev), JLIM.to(dev))
        self.qd[mask] = 0.0
        self.grip[mask] = 1.0
        self.obj[mask] = torch.stack([x, torch.full_like(x, OBJ_HALF)], -1)
        self.obj_v[mask] = 0.0
        self.grasped[mask] = 0.0
        self.lift_hold[mask] = 0.0
        self.was_grasped[mask] = 0.0
        self.steps[mask] = 0

    def observe(self):
        return self._observe()

    def reset(self):
        self._spawn(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._observe()

    # ----------------------------------------------------------------- step
    def step(self, action):
        a = action.clamp(-1, 1)
        dev = self.device
        # joints: smoothed velocity commands
        tgt = a[:, :3] * QD_MAX
        self.qd = self.qd + (tgt - self.qd) * 0.35
        self.q = self.q + self.qd * DT
        jlim = JLIM.to(self.device)
        for j in range(3):
            below = self.q[:, j] < -jlim[j]
            above = self.q[:, j] > jlim[j]
            self.q[:, j] = self.q[:, j].clamp(-jlim[j], jlim[j])
            self.qd[:, j] = torch.where(below | above, torch.zeros_like(self.qd[:, j]), self.qd[:, j])
        # gripper channel
        g_tgt = (a[:, 3] + 1.0) * 0.5
        self.grip = self.grip + (g_tgt - self.grip).clamp(-4 * DT, 4 * DT)

        ee = self._ee()
        dist = (self.obj - ee).norm(dim=-1)
        near = dist < 0.040
        # hysteresis: attach below 0.04, stay attached while within 0.06
        # (prevents grasp-state flicker farming at the boundary)
        attach = near & (self.grip < 0.35) & (self.grasped < 0.5)
        slip = (self.grasped > 0.5) & (dist > 0.060)
        detach = (self.grasped > 0.5) & (g_tgt > 0.55)
        self.was_grasped = torch.maximum(self.was_grasped, self.grasped)
        self.grasped = ((self.grasped > 0.5) | attach) & ~slip & ~detach
        self.grasped = self.grasped.float()

        # object dynamics
        held = self.grasped > 0.5
        prev_y = self.obj[:, 1]
        self.obj = torch.where(held.unsqueeze(-1), ee.detach(), self.obj + self.obj_v * DT)
        free = ~held
        self.obj_v = torch.where(free.unsqueeze(-1),
                                 self.obj_v + torch.tensor([0.0, -G * DT], device=dev), self.obj_v)
        # table contact for free object
        on_table = free & (self.obj[:, 1] <= OBJ_HALF)
        self.obj = torch.where(on_table.unsqueeze(-1),
                               torch.stack([self.obj[:, 0], torch.full_like(self.obj[:, 1], OBJ_HALF)], -1), self.obj)
        self.obj_v = torch.where(on_table.unsqueeze(-1),
                                 torch.stack([self.obj_v[:, 0] * 0.8, torch.abs(self.obj_v[:, 1]) * 0.2], -1), self.obj_v)
        self.obj_v = torch.nan_to_num(self.obj_v)

        height = self.obj[:, 1] - OBJ_HALF
        lifted = height > LIFT_H
        self.lift_hold = torch.where(held & lifted, self.lift_hold + 1, torch.zeros_like(self.lift_hold))
        lift_success = (self.lift_hold >= 2).float()
        drop = (self.was_grasped > 0.5) & (height < -0.01) & held.bitwise_not() & (lift_success < 0.5)
        self.steps += 1
        timeout = (self.steps >= self.max_steps).float()
        done = (lift_success > 0.5) | (timeout > 0.5)

        signals = {
            "dist_ee_obj": dist, "grasp": self.grasped,
            "height": height.clamp_min(0), "lift_success": lift_success,
            "drop": drop.float(), "effort": (a * a).sum(-1) * 0.2, "timeout": timeout,
        }
        info = {"done": done.clone(), "episode_outcome": lift_success,
                "episode_success": lift_success * done.float(),
                "diag": {"dist": float(dist.mean()), "grasp": float(self.grasped.mean()),
                         "grip": float(self.grip.mean()), "height": float(height.mean())}}
        return self._observe(), signals, done, info

    # ----------------------------------------------------- episode control
    def reset_done(self, done):
        self._spawn(done)
        return self._observe()

    def initial_signals(self):
        ee = self._ee()
        dist = (self.obj - ee).norm(dim=-1)
        z = torch.zeros_like(dist)
        return {"dist_ee_obj": dist, "grasp": z.clone(), "height": self.obj[:, 1] - OBJ_HALF,
                "lift_success": z.clone(), "drop": z.clone(), "effort": z.clone(),
                "timeout": z.clone()}

    _STATE_FIELDS = ("q", "qd", "grip", "obj", "obj_v", "grasped", "lift_hold", "was_grasped", "steps")

    def get_state(self):
        return {f: getattr(self, f).clone() for f in self._STATE_FIELDS}

    def set_state(self, state):
        for f, v in state.items():
            getattr(self, f)[...] = v

    def _observe(self):
        ee = self._ee()
        rel = self.obj - ee
        return torch.cat([self.q.sin(), self.q.cos(), self.qd / QD_MAX,
                          self.grip.unsqueeze(-1), rel, self.obj_v / 2.0,
                          self.grasped.unsqueeze(-1), rel.norm(dim=-1, keepdim=True) * 3.0], -1)
