"""E1 - 2D holonomic robot obstacle-avoidance navigation.

A point robot (radius 0.03) must cross a random field of circular obstacles to
a random goal in the unit arena.  Observation = goal vector (2) + own position
(2) + velocity (2) + 8 short-range lidar rays (8) -> 14 dims.  Action = planar
acceleration command (2), tanh-bounded.  Episodes truncate at `max_steps`.

Signals (per step): dist (to goal), prev_dist, success, collision, speed,
effort (|a|^2), timeout.  Reward paradigms consume these.
"""
from __future__ import annotations

import torch

LIDAR_RAYS = 8
MAX_RANGE = 0.40


class NavObstacleEnv:
    obs_dim = 14
    act_dim = 2
    signal_keys = ("dist", "prev_dist", "success", "collision", "speed", "effort", "timeout")

    def __init__(self, num_envs, seed, device, max_steps=150,
                 n_obstacles=(4, 8), goal_dist=(0.55, 0.85)):
        self.num_envs = num_envs
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.max_steps = max_steps
        self.n_obs_lo, self.n_obs_hi = n_obstacles
        self.goal_dist = goal_dist
        self.max_obstacles = n_obstacles[1]
        self._progress = 0.0

        self.obs_xy = torch.zeros(num_envs, self.max_obstacles, 2, device=device)
        self.obs_r = torch.zeros(num_envs, self.max_obstacles, device=device)
        self.n_obs = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pos = torch.zeros(num_envs, 2, device=device)
        self.vel = torch.zeros(num_envs, 2, device=device)
        self.goal = torch.zeros(num_envs, 2, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        ang = torch.linspace(0, 2 * torch.pi, LIDAR_RAYS + 1)[:-1].to(device)
        self.ray_dir = torch.stack([ang.cos(), ang.sin()], dim=-1)   # (R,2)

    # ------------------------------------------------------------- spawning
    def _spawn(self, mask):
        """Re-sample start/obstacles/goal for the envs in `mask` (bool B)."""
        B = self.num_envs
        m = mask
        n_m = int(m.sum())
        if n_m == 0:
            return
        # curriculum: fewer obstacles + shorter goal early in training
        lo = max(1, round(self.n_obs_lo * (0.4 + 0.6 * self._progress)))
        n = torch.randint(lo, self.n_obs_hi + 1, (n_m,), generator=self.gen, device=self.device)

        pos = 0.05 + 0.90 * torch.rand(n_m, 2, generator=self.gen, device=self.device)
        cand_xy = torch.rand(n_m, self.max_obstacles, 2, generator=self.gen, device=self.device)
        cand_r = 0.045 + 0.055 * torch.rand(n_m, self.max_obstacles, generator=self.gen, device=self.device)
        idx = torch.arange(self.max_obstacles, device=self.device).expand(n_m, -1)
        d_start = (cand_xy - pos.unsqueeze(1)).norm(dim=-1)
        valid = (idx < n.unsqueeze(1)) & (d_start > cand_r + 0.10)
        # push invalid obstacles outside the arena so they can never collide
        keep = valid.float().unsqueeze(-1)
        xy = cand_xy * keep + (1.0 - keep) * 5.0
        r = cand_r * keep.squeeze(-1)

        # goal: between 0.8*span and span+0.15 from start, clear of obstacles
        span = self.goal_dist[0] + (self.goal_dist[1] - self.goal_dist[0]) * (0.55 + 0.45 * self._progress)
        goal = pos.clone()
        d_obs_goal = ((goal.unsqueeze(1) - xy).norm(dim=-1) - r).amin(dim=1)
        for _ in range(16):
            cand = 0.05 + 0.90 * torch.rand(n_m, 2, generator=self.gen, device=self.device)
            d_start = (cand - pos).norm(dim=-1)
            d_obs = ((cand.unsqueeze(1) - xy).norm(dim=-1) - r).amin(dim=1)
            ok = (d_start > 0.8 * span) & (d_start < span + 0.15) & (d_obs > 0.08)
            goal = torch.where(ok.unsqueeze(-1), cand, goal)
        self.pos[m] = pos
        self.vel[m] = 0.0
        self.obs_xy[m] = xy
        self.obs_r[m] = r
        self.n_obs[m] = n
        self.goal[m] = goal
        self.steps[m] = 0

    def observe(self):
        return self._observe()

    def reset(self):
        self._spawn(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._observe()

    def set_progress(self, p):
        self._progress = float(min(max(p, 0.0), 1.0))

    # -------------------------------------------------------------- observe
    def _observe(self):
        rel = self.goal - self.pos
        # lidar ray-circle intersection: smallest positive t in [0, MAX_RANGE]
        oc = self.pos[:, None, None, :] - self.obs_xy[:, :, None, :]     # (B,O,R,2)
        b = (oc * self.ray_dir).sum(-1)                                  # (B,O,R)
        pp = (self.pos ** 2).sum(-1)[:, None, None]
        cc = (self.obs_xy ** 2).sum(-1)[:, :, None] - 2 * (self.obs_xy * self.pos[:, None, :]).sum(-1)[:, :, None] + pp
        c = cc - self.obs_r[:, :, None] ** 2
        disc = (b * b - c).clamp_min(0)
        t = (-b - disc.sqrt()).clamp_min(0)
        hit = (t > 0) & (t < MAX_RANGE)
        dist = torch.where(hit, t, torch.full_like(t, MAX_RANGE)).amin(dim=1)
        return torch.cat([rel, self.pos, self.vel, dist / MAX_RANGE], dim=-1)

    # ----------------------------------------------------------------- step
    def step(self, action):
        a = action.clamp(-1, 1)
        dt = 1.0 / 15.0
        prev_pos = self.pos.clone()
        self.vel = self.vel + a * 4.0 * dt
        self.vel = self.vel * (1.0 - 0.6 * dt)
        vn = self.vel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.vel = torch.where(vn > 1.0, self.vel / vn, self.vel)
        self.pos = (self.pos + self.vel * dt).clamp(0.0, 1.0)
        at_wall = ((self.pos <= 0.0) | (self.pos >= 1.0)).any(-1, keepdim=True)
        self.vel = self.vel * (~at_wall).float()
        self.steps += 1

        prev_dist = (self.goal - prev_pos).norm(dim=-1)
        dist = (self.goal - self.pos).norm(dim=-1)
        d_obs = ((self.pos[:, None, :] - self.obs_xy).norm(dim=-1) - self.obs_r).amin(dim=1)
        collision = d_obs < 0.03
        success = dist < 0.06
        timeout = self.steps >= self.max_steps
        done = collision | success | timeout

        signals = {
            "dist": dist, "prev_dist": prev_dist,
            "success": success.float(), "collision": collision.float(),
            "speed": self.vel.norm(dim=-1),
            "effort": (a * a).sum(-1), "timeout": timeout.float(),
        }
        info = {"done": done.clone(),
                "episode_outcome": success.float() - collision.float(),
                "episode_success": torch.where(done, success.float(), torch.zeros_like(dist))}
        return self._observe(), signals, done, info

    # ----------------------------------------------------- episode control
    def reset_done(self, done):
        self._spawn(done)
        return self._observe()

    def initial_signals(self):
        dist = (self.goal - self.pos).norm(dim=-1)
        z = torch.zeros_like(dist)
        return {"dist": dist, "prev_dist": dist, "success": z, "collision": z,
                "speed": z, "effort": z, "timeout": z}

    _STATE_FIELDS = ("pos", "vel", "goal", "obs_xy", "obs_r", "n_obs", "steps")

    def get_state(self):
        return {f: getattr(self, f).clone() for f in self._STATE_FIELDS}

    def set_state(self, state):
        for f, v in state.items():
            getattr(self, f)[...] = v
