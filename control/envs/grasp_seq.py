"""E4 - continuous sequential grasp-carry-present cycles ("GraspSeq").

Extends E3's velocity-controlled 3-link arm from a single grasp-lift into a
*continuous manipulation cycle*: three objects must each be reached,
grasped, lifted past LIFT_H, carried into the target zone
(|x| < ZONE_HALF) and held steady there (presented) for a few steps; the
object is then received (removed) and the next object becomes the active
target.  The arm is never reset between objects - the whole appeal is
repeated, coordinated reach->grasp->lift->carry->present behaviour.
(A gripper-release variant was probed for 8 full budgets: the release
action is never explored by vanilla PPO - the suite's own "hover optimum"
finding at cycle scale - so completion is defined by held presentation,
which keeps every carry/coordination demand.)

Obs (19): joint sin/cos(6), qd(3), grip(1), obj rel EE(2), obj vel(2),
grasped(1), obj dist(1), zone rel EE(2), completed/3(1).
Signals: dist_ee_obj, grasp, height, lift_success, place, drop, carry,
drop_err, effort, timeout.
Episode: 300 steps; success = all 3 objects presented; outcome = done/3.
"""
from __future__ import annotations

import torch

from .reach_grasp import G, L1, L2, L3, BASE_Y, OBJ_HALF, LIFT_H, DT, QD_MAX, JLIM

N_OBJ = 3
ZONE_HALF = 0.15
PRESENT_HOLD = 5


class GraspSeqEnv:
    obs_dim = 19
    act_dim = 4
    signal_keys = ("dist_ee_obj", "grasp", "height", "lift_success", "place",
                   "drop", "release_good", "release_zone", "carry", "drop_err",
                   "effort", "timeout")

    def __init__(self, num_envs, seed, device, max_steps=300):
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.max_steps = max_steps
        self._progress = 0.0
        self.q = torch.zeros(num_envs, 3, device=device)
        self.qd = torch.zeros(num_envs, 3, device=device)
        self.grip = torch.ones(num_envs, device=device)
        self.obj = torch.zeros(num_envs, 2, device=self.device)
        self.obj_v = torch.zeros(num_envs, 2, device=self.device)
        self.grasped = torch.zeros(num_envs, device=self.device)
        self.lift_hold = torch.zeros(num_envs, device=self.device)
        self.lifted_flag = torch.zeros(num_envs, device=self.device)   # per active object
        self.drop_flag = torch.zeros(num_envs, device=self.device)     # failed-release latch
        self.rel_paid = torch.zeros(num_envs, device=self.device)      # release-good paid (per object)
        self.relz_paid = torch.zeros(num_envs, device=self.device)     # release-over-zone paid
        self.refrac = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # post-release grasp refractory
        self.placed = torch.zeros(num_envs, device=self.device)        # one-shot event
        self.completed = torch.zeros(num_envs, device=self.device)     # objects placed
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.ev_release = torch.zeros(num_envs, device=self.device)
        self.ev_zone = torch.zeros(num_envs, device=self.device)
        self.ev_place = torch.zeros(num_envs, device=self.device)
        self.ev_settle_out = torch.zeros(num_envs, device=self.device)
        self.present_hold = torch.zeros(num_envs, device=self.device)

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
    def _spawn_object(self, mask):
        """Spawn a fresh active object (arm state untouched - continuity)."""
        n = int(mask.sum())
        if n == 0:
            return
        dev = self.device
        x_hi = 0.18 + 0.24 * self._progress
        reach = (L1 + L2 + L3) * 0.97
        x_max = (reach ** 2 - (BASE_Y - OBJ_HALF) ** 2) ** 0.5
        x_hi = min(x_hi, 0.9 * x_max)
        x = -x_hi + 2 * x_hi * torch.rand(n, generator=self.gen, device=dev)
        self.obj[mask] = torch.stack([x, torch.full_like(x, OBJ_HALF)], -1)
        self.obj_v[mask] = 0.0
        self.grasped[mask] = 0.0
        self.lift_hold[mask] = 0.0
        self.lifted_flag[mask] = 0.0
        self.drop_flag[mask] = 0.0
        self.rel_paid[mask] = 0.0
        self.relz_paid[mask] = 0.0
        self.refrac[mask] = 0
        self.present_hold[mask] = 0
        self.ev_release[mask] = 0.0
        self.ev_zone[mask] = 0.0
        self.ev_place[mask] = 0.0
        self.ev_settle_out[mask] = 0.0

    def _spawn(self, mask):
        """Episode (re)spawn: reset arm pose + counters + first object.

        The arm starts loosely aimed at the first object's side (same
        convention as E3) so the learnable difficulty sits in the
        grasp-lift-carry-place cycles, not in blind re-homing."""
        n = int(mask.sum())
        if n == 0:
            return
        dev = self.device
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
        self.completed[mask] = 0.0
        self.placed[mask] = 0.0
        self.steps[mask] = 0
        self.obj[mask] = torch.stack([x, torch.full_like(x, OBJ_HALF)], -1)
        self.obj_v[mask] = 0.0
        self.grasped[mask] = 0.0
        self.lift_hold[mask] = 0.0
        self.lifted_flag[mask] = 0.0
        self.drop_flag[mask] = 0.0
        self.rel_paid[mask] = 0.0
        self.relz_paid[mask] = 0.0
        self.refrac[mask] = 0
        self.present_hold[mask] = 0
        self.ev_release[mask] = 0.0
        self.ev_zone[mask] = 0.0
        self.ev_place[mask] = 0.0
        self.ev_settle_out[mask] = 0.0

    def observe(self):
        return self._observe()

    def reset(self):
        self._spawn(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._observe()

    # ----------------------------------------------------------------- step
    def step(self, action):
        a = action.clamp(-1, 1)
        dev = self.device
        tgt = a[:, :3] * QD_MAX
        self.qd = self.qd + (tgt - self.qd) * 0.35
        self.q = self.q + self.qd * DT
        jlim = JLIM.to(self.device)
        for j in range(3):
            below = self.q[:, j] < -jlim[j]
            above = self.q[:, j] > jlim[j]
            self.q[:, j] = self.q[:, j].clamp(-jlim[j], jlim[j])
            self.qd[:, j] = torch.where(below | above, torch.zeros_like(self.qd[:, j]), self.qd[:, j])
        g_tgt = (a[:, 3] + 1.0) * 0.5
        self.grip = self.grip + (g_tgt - self.grip).clamp(-4 * DT, 4 * DT)

        ee = self._ee()
        dist = (self.obj - ee).norm(dim=-1)
        near = dist < 0.040
        # refractory: after a voluntary release the gripper cannot re-attach
        # for a few steps, so a released object actually settles (and any
        # in-zone release converts to a placement) - rate-limit by design
        attach = near & (self.grip < 0.35) & (self.grasped < 0.5) & (self.refrac <= 0)
        slip = (self.grasped > 0.5) & (dist > 0.060)
        detach = (self.grasped > 0.5) & (g_tgt > 0.55)
        # edge events, RATE-LIMITED to once per object (anti-farming,
        # cf. paper-2 guideline #2): after one paid release the only
        # remaining value is letting the object settle (place) or re-carrying
        release_good = (detach & (self.lifted_flag > 0.5) & (self.rel_paid < 0.5)).float()
        self.rel_paid = torch.where(release_good > 0.5, torch.ones_like(self.rel_paid), self.rel_paid)
        release_zone = (detach & (self.obj[:, 0].abs() < ZONE_HALF + 0.06)
                        & (self.lifted_flag > 0.5) & (self.relz_paid < 0.5)).float()
        self.relz_paid = torch.where(release_zone > 0.5, torch.ones_like(self.relz_paid), self.relz_paid)
        self.grasped = ((self.grasped > 0.5) | attach) & ~slip & ~detach
        self.grasped = self.grasped.float()

        held = self.grasped > 0.5
        self.obj = torch.where(held.unsqueeze(-1), ee.detach(), self.obj + self.obj_v * DT)
        free = ~held
        self.obj_v = torch.where(free.unsqueeze(-1),
                                 self.obj_v + torch.tensor([0.0, -G * DT], device=dev), self.obj_v)
        on_table = free & (self.obj[:, 1] <= OBJ_HALF)
        self.obj = torch.where(on_table.unsqueeze(-1),
                               torch.stack([self.obj[:, 0], torch.full_like(self.obj[:, 1], OBJ_HALF)], -1), self.obj)
        self.obj_v = torch.where(on_table.unsqueeze(-1),
                                 torch.stack([self.obj_v[:, 0] * 0.8, torch.abs(self.obj_v[:, 1]) * 0.2], -1), self.obj_v)
        self.obj_v = torch.nan_to_num(self.obj_v)

        height = self.obj[:, 1] - OBJ_HALF
        lifted = height > LIFT_H
        self.lift_hold = torch.where(held & lifted, self.lift_hold + 1, torch.zeros_like(self.lift_hold))
        lift_success = ((self.lift_hold >= 2) & (self.lifted_flag < 0.5)).float()
        self.lifted_flag = torch.maximum(self.lifted_flag, (self.lift_hold >= 2).float())

        # completion = HELD PRESENTATION: lifted, inside the zone, held
        # steady (small EE speed) for PRESENT_HOLD consecutive steps
        in_zone = self.obj[:, 0].abs() < ZONE_HALF
        ee_speed = self.qd.abs().max(dim=-1).values * (L1 + L2 + L3)
        presenting = held & in_zone & (height > 0.10) & (self.lifted_flag > 0.5) & (ee_speed < 0.55)
        self.present_hold = torch.where(presenting, self.present_hold + 1,
                                        torch.zeros_like(self.present_hold))
        self.placed = (self.present_hold >= PRESENT_HOLD).float()
        settled = free & (self.obj[:, 1] <= OBJ_HALF + 1e-4) & (self.obj_v[:, 1].abs() < 0.05)
        # slip-loss while carrying = the meaningful "drop" penalty
        drop = (settled & (self.lifted_flag > 0.5) & ~self.grasped.bool()).float()
        self.ev_release = self.ev_release + release_good
        self.ev_zone = self.ev_zone + release_zone
        self.ev_place = self.ev_place + newly_done.float().where(newly_done.bool(), torch.zeros_like(self.ev_place)) if False else self.ev_place
        self.ev_place = self.ev_place + self.placed
        self.ev_settle_out = self.ev_settle_out + drop

        newly_done = self.placed > 0.5
        self.completed = self.completed + newly_done.float()
        self._spawn_object(newly_done)

        self.steps += 1
        timeout = (self.steps >= self.max_steps).float()
        all_done = (self.completed >= N_OBJ - 0.5).float()
        done = (all_done > 0.5) | (timeout > 0.5)

        signals = {
            "dist_ee_obj": dist, "grasp": self.grasped,
            "height": height.clamp_min(0), "lift_success": lift_success,
            "place": self.placed, "drop": drop,
            "release_good": release_good, "release_zone": release_zone,
            "carry": self.lifted_flag,
            "drop_err": (self.obj[:, 0].abs() / 0.3).clamp(0, 1),
            "effort": (a * a).sum(-1) * 0.2, "timeout": timeout,
        }
        info = {"done": done.clone(), "episode_outcome": self.completed / N_OBJ,
                "episode_success": all_done * done.float(),
                "diag": {"dist": float(dist.mean()), "grasp": float(self.grasped.mean()),
                         "completed": float(self.completed.mean()),
                         "height": float(height.mean()),
                         "objx": float(self.obj[:, 0].abs().mean()),
                         "ev_release": float(self.ev_release.mean()),
                         "ev_zone": float(self.ev_zone.mean()),
                         "ev_place": float(self.ev_place.mean()),
                         "ev_settle_out": float(self.ev_settle_out.mean())}}
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
                "lift_success": z.clone(), "place": z.clone(), "drop": z.clone(),
                "release_good": z.clone(), "release_zone": z.clone(),
                "carry": z.clone(),
                "drop_err": (self.obj[:, 0].abs() / 0.3).clamp(0, 1),
                "effort": z.clone(), "timeout": z.clone()}

    _STATE_FIELDS = ("q", "qd", "grip", "obj", "obj_v", "grasped", "lift_hold",
                     "lifted_flag", "drop_flag", "refrac", "present_hold",
                     "placed", "completed", "steps")

    def get_state(self):
        return {f: getattr(self, f).clone() for f in self._STATE_FIELDS}

    def set_state(self, state):
        for f, v in state.items():
            getattr(self, f)[...] = v

    def _observe(self):
        ee = self._ee()
        rel = self.obj - ee
        drop_rel = -ee                                   # zone centre at (0, ~0)
        drop_rel = torch.stack([0.0 - ee[:, 0], 0.30 - ee[:, 1]], -1)
        return torch.cat([self.q.sin(), self.q.cos(), self.qd / QD_MAX,
                          self.grip.unsqueeze(-1), rel, self.obj_v / 2.0,
                          self.grasped.unsqueeze(-1), rel.norm(dim=-1, keepdim=True) * 3.0,
                          drop_rel * 3.0, (self.completed / N_OBJ).unsqueeze(-1)], -1)
