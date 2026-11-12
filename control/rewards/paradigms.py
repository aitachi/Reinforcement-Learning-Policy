"""Reward paradigm library: one task, four paradigms.

A paradigm maps per-step *signals* (exposed by each env) to a scalar reward.
It is stateful: progress/potential terms need the previous step's signals
within the same episode.  The trainer calls:

    reward, terms = paradigm(signals, done)      # after env.step
    paradigm.on_reset(done, initial_signals)     # after env.reset_done

Paradigms per environment
-------------------------
sparse      : terminal outcome only (+3 success / -1 failure / tiny time cost).
              The hardest credit-assignment condition; favorable to
              group-relative methods (GRPO/DAPO) that baseline whole episodes.
dense       : engineered multi-term shaping (progress, energy, events).
              The classic "well-engineered" reward most on-policy baselines
              are tuned against.
pbs         : potential-based shaping r' = r_sparse + gamma*phi(s') - phi(s)
              added on top of the sparse reward.  By Ng, Harada & Russell
              (1999) the shaped reward preserves the optimal policy *exactly*,
              which isolates "shaping that cannot hurt" from "shaping that
              can" (dense engineered terms carry no such guarantee).
curriculum  : dense early -> sparse late (weight w(progress)), combined with
              env-side difficulty schedules (obstacle count, target speed,
              spawn radius) driven by the same progress signal.

Every paradigm also returns a per-term breakdown for the reward-audit
figures in the paper (detecting reward hacking = one term dominating).
"""
from __future__ import annotations

import torch

GAMMA = 0.99


class RewardParadigm:
    name = "base"
    needs_prev = ()

    def __init__(self, num_envs, device):
        self.B = num_envs
        self.device = device

    def on_reset(self, done, init_signals):
        pass

    def __call__(self, sig, done):
        raise NotImplementedError


def _term_dict(**kw):
    return kw


# ----------------------------------------------------------------- NAV ----
class NavSparse(RewardParadigm):
    name = "sparse"

    def __call__(self, sig, done):
        r = 3.0 * sig["success"] - 1.0 * sig["collision"] - 0.01 * sig["timeout"]
        return r, _term_dict(success=3.0 * sig["success"], collision=-1.0 * sig["collision"],
                             time=-0.01 * torch.ones_like(r))


class NavDense(RewardParadigm):
    name = "dense"
    needs_prev = ("dist",)

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.prev_dist = torch.zeros(num_envs, device=device)

    def on_reset(self, done, init):
        self.prev_dist = torch.where(done, init["dist"], self.prev_dist)

    def __call__(self, sig, done):
        progress = self.prev_dist - sig["dist"]
        r = (2.5 * progress - 0.01 - 0.02 * sig["effort"]
             + 3.0 * sig["success"] - 1.0 * sig["collision"])
        self.prev_dist = sig["dist"]
        return r, _term_dict(progress=2.5 * progress, time=-0.01 * torch.ones_like(r),
                             effort=-0.02 * sig["effort"], success=3.0 * sig["success"],
                             collision=-1.0 * sig["collision"])


class NavPBS(RewardParadigm):
    """sparse + gamma*phi(s') - phi(s), phi = -4 * dist(goal)."""
    name = "pbs"
    needs_prev = ("dist",)

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.prev_dist = torch.zeros(num_envs, device=device)

    def on_reset(self, done, init):
        self.prev_dist = torch.where(done, init["dist"], self.prev_dist)

    def __call__(self, sig, done):
        base = 3.0 * sig["success"] - 1.0 * sig["collision"] - 0.01 * sig["timeout"]
        shaping = GAMMA * (-4.0 * sig["dist"]) - (-4.0 * self.prev_dist)
        r = base + shaping
        self.prev_dist = sig["dist"]
        return r, _term_dict(base=base, shaping=shaping)


class NavCurriculum(RewardParadigm):
    """dense -> sparse annealing; env already ramps obstacle count / goal span."""
    name = "curriculum"
    needs_prev = ("dist",)

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.prev_dist = torch.zeros(num_envs, device=device)
        self.dense = NavDense(num_envs, device)
        self.sparse = NavSparse(num_envs, device)
        self.progress = 0.0

    def set_progress(self, p):
        self.progress = min(max(p, 0.0), 1.0)

    def on_reset(self, done, init):
        self.dense.on_reset(done, init)
        self.prev_dist = self.dense.prev_dist

    def __call__(self, sig, done):
        w = min(1.0, self.progress / 0.8) ** 1.5          # sparse weight ramp
        rd, td = self.dense(sig, done)
        rs, ts = self.sparse(sig, done)
        r = (1 - w) * rd + w * rs
        return r, _term_dict(mix_weight=w * torch.ones_like(r), dense=rd, sparse=rs)


# -------------------------------------------------------------- HOPPER ----
class HopperSparse(RewardParadigm):
    name = "sparse"

    def __call__(self, sig, done):
        # displacement comes from info["episode_x"] injected by the trainer as sig["episode_x"]
        x = sig.get("episode_x", sig["vx"])
        terminal = done.float() * (x / 3.0 - 2.0 * sig["fallen"])
        r = 0.01 * sig["alive"] + terminal + 0.5 * sig["timeout"] * sig["alive"]
        return r, _term_dict(alive=0.01 * sig["alive"], terminal=terminal,
                             survive_bonus=0.5 * sig["timeout"] * sig["alive"])


class HopperDense(RewardParadigm):
    name = "dense"

    def __init__(self, num_envs, device, target_speed=1.2):
        super().__init__(num_envs, device)
        self.target = target_speed

    def __call__(self, sig, done):
        vx = sig["vx"].clamp(-2.0, 2.0)
        speed_term = 1.5 * vx.clamp_max(self.target) - 0.6 * (vx - self.target).abs()
        r = speed_term + 0.5 * sig["alive"] - 2.0 * sig["fallen"] - 0.004 * sig["effort"]
        return r, _term_dict(speed=speed_term, alive=0.5 * sig["alive"],
                             fall=-2.0 * sig["fallen"], effort=-0.004 * sig["effort"])


class HopperPBS(RewardParadigm):
    """sparse + gamma*phi - phi with the state potential
    phi(s) = 2*clip(vx) + 0.5*alive + 0.5*clip(height)  (gamma<1 caveat in paper)."""
    name = "pbs"

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.prev_phi = torch.zeros(num_envs, device=device)
        self.sparse = HopperSparse(num_envs, device)

    def phi(self, sig):
        return 2.0 * sig["vx"].clamp(-2.0, 2.0) + 0.5 * sig["alive"] + 0.5 * sig["height"].clamp(0.4, 0.8)

    def on_reset(self, done, init):
        self.prev_phi = torch.where(done, self.phi(init), self.prev_phi)

    def __call__(self, sig, done):
        base = self.sparse(sig, done)[0]
        cur = self.phi(sig)
        shaping = GAMMA * cur - self.prev_phi
        self.prev_phi = cur
        return base + shaping, _term_dict(base=base, shaping=shaping)


class HopperCurriculum(RewardParadigm):
    name = "curriculum"

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.dense = HopperDense(num_envs, device)
        self.progress = 0.0

    def set_progress(self, p):
        self.progress = min(max(p, 0.0), 1.0)

    def __call__(self, sig, done):
        self.dense.target = 0.4 + 0.8 * min(1.0, self.progress / 0.7)
        r, t = self.dense(sig, done)
        t["target_speed"] = self.dense.target * torch.ones_like(r)
        return r, t


# --------------------------------------------------------------- GRASP ----
class GraspSparse(RewardParadigm):
    name = "sparse"

    def __call__(self, sig, done):
        r = 3.0 * sig["lift_success"] - 0.5 * sig["drop"] - 0.01
        return r, _term_dict(lift=3.0 * sig["lift_success"], drop=-0.5 * sig["drop"],
                             time=-0.01 * torch.ones_like(r))


class GraspDense(RewardParadigm):
    """approach + lift + one-shot height milestones (breaks the post-grasp
    plateau where pure d(height) shaping gives no exploration gradient)."""
    name = "dense"
    needs_prev = ("dist_ee_obj", "height")
    MILESTONES = (0.05, 0.10, 0.14)

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.prev_dist = torch.zeros(num_envs, device=device)
        self.prev_h = torch.zeros(num_envs, device=device)
        self.grasp_paid = torch.zeros(num_envs, device=device)
        self.crossed = torch.zeros(num_envs, len(self.MILESTONES), device=device)

    def on_reset(self, done, init):
        self.prev_dist = torch.where(done, init["dist_ee_obj"], self.prev_dist)
        self.prev_h = torch.where(done, init["height"], self.prev_h)
        self.grasp_paid = torch.where(done, torch.zeros_like(self.grasp_paid), self.grasp_paid)
        self.crossed = torch.where(done.unsqueeze(-1), torch.zeros_like(self.crossed), self.crossed)

    def __call__(self, sig, done):
        approach = 3.0 * (self.prev_dist - sig["dist_ee_obj"])
        lift = 8.0 * (sig["height"] - self.prev_h).clamp_min(0) * sig["grasp"]
        first_grasp = 2.0 * ((sig["grasp"] > 0.5) & (self.grasp_paid < 0.5)).float()
        self.grasp_paid = torch.maximum(self.grasp_paid, sig["grasp"])
        newly = (sig["height"].unsqueeze(-1) > torch.tensor(self.MILESTONES, device=self.device)) \
                & (self.crossed < 0.5)
        milestone = 1.0 * newly.float().sum(-1)
        self.crossed = ((self.crossed > 0.5) | newly).float()
        r = (approach + lift + milestone + first_grasp
             + 5.0 * sig["lift_success"] - 0.5 * sig["drop"] - 0.02 - 0.005 * sig["effort"])
        self.prev_dist = sig["dist_ee_obj"]
        self.prev_h = sig["height"]
        return r, _term_dict(approach=approach, lift=lift, milestone=milestone,
                             grasp_event=first_grasp,
                             success=5.0 * sig["lift_success"], drop=-0.5 * sig["drop"])


class GraspPBS(RewardParadigm):
    """sparse + shaping with phi = -4*dist_ee_obj + 2*grasped*min(height, LIFT)."""
    name = "pbs"
    needs_prev = ()

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.prev_phi = torch.zeros(num_envs, device=device)
        self.sparse = GraspSparse(num_envs, device)

    def phi(self, sig):
        return -4.0 * sig["dist_ee_obj"] + 2.0 * sig["grasp"] * sig["height"].clamp_max(0.18)

    def on_reset(self, done, init):
        self.prev_phi = torch.where(done, self.phi(init), self.prev_phi)

    def __call__(self, sig, done):
        base = self.sparse(sig, done)[0]
        cur = self.phi(sig)
        shaping = GAMMA * cur - self.prev_phi
        self.prev_phi = cur
        return base + shaping, _term_dict(base=base, shaping=shaping)


class GraspCurriculum(RewardParadigm):
    name = "curriculum"
    needs_prev = ("dist_ee_obj", "height")

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.dense = GraspDense(num_envs, device)
        self.sparse = GraspSparse(num_envs, device)
        self.progress = 0.0

    def set_progress(self, p):
        self.progress = min(max(p, 0.0), 1.0)

    def on_reset(self, done, init):
        self.dense.on_reset(done, init)

    def __call__(self, sig, done):
        w = min(1.0, self.progress / 0.8) ** 1.5
        rd, _ = self.dense(sig, done)
        rs, _ = self.sparse(sig, done)
        r = (1 - w) * rd + w * rs
        return r, _term_dict(mix_weight=w * torch.ones_like(r), dense=rd, sparse=rs)


PARADIGMS = {
    ("nav", "sparse"): NavSparse, ("nav", "dense"): NavDense,
    ("nav", "pbs"): NavPBS, ("nav", "curriculum"): NavCurriculum,
    ("hopper", "sparse"): HopperSparse, ("hopper", "dense"): HopperDense,
    ("hopper", "pbs"): HopperPBS, ("hopper", "curriculum"): HopperCurriculum,
    ("grasp", "sparse"): GraspSparse, ("grasp", "dense"): GraspDense,
    ("grasp", "pbs"): GraspPBS, ("grasp", "curriculum"): GraspCurriculum,
}
