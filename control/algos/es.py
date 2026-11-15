"""OpenAI-style Evolution Strategies (antithetic / mirrored sampling) - the
gradient-free baseline of the suite.

Uses the SAME 2x128 Gaussian policy trunk as every on-policy method (fair
network budget).  Each generation, a population of P=num_envs perturbed
parameter vectors (+/- sigma * eps) is rolled out for one fitness window
(es_window control steps) in the vectorized env - one member per env slot,
batched via torch.func.vmap.  The generation gradient is the rank-shaped
antithetic estimator (Salimans et al., 2017), applied with Adam to the
central parameters.  (CMA-ES proper needs a full covariance matrix and is
infeasible at this parameter count under a shared-network protocol.)

Sample-budget note (documented in README/report): under the suite's strict
per-step fairness protocol a sample-matched ES sees only
total_steps / (P * window) ~ 100-450 generations - far below ES's natural
thousands.  This row measures the *sample* price of gradient-free search
under parity, not ES's wall-clock-parallel regime.
"""
from __future__ import annotations

import torch
from torch.func import functional_call, stack_module_state, vmap

from .nets import GaussianActor


class ES:
    needs = "episode"          # same driver branch as GRPO/DAPO (collect/update)
    kind = "es"

    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.policy = GaussianActor(obs_dim, act_dim).to(device)
        self.sigma = cfg.get("es_sigma", 0.05)
        self.lr = cfg.get("es_lr", 0.05)
        self.window = cfg.get("es_window", 64)
        self.rank_shape = cfg.get("es_rank", True)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self._stacked = None
        self._eps = None
        self._gen = 0

    # ----------------------------------------------------- population utils
    def _refresh(self, num_envs):
        with torch.no_grad():
            params, buffers = stack_module_state([self.policy])
            self._stacked = {**{k: v.detach().clone() for k, v in params.items()},
                             **{k: v.detach().clone() for k, v in buffers.items()}}
        if self._eps is None or next(iter(self._eps.values())).shape[0] != num_envs // 2:
            half = num_envs // 2
            self._eps = {k: torch.randn(half, *v.shape[1:], device=self.device)
                         for k, v in self._stacked.items()}

    def _member_params(self):
        out = {}
        for k, v in self._stacked.items():
            plus = v[0] + self.sigma * self._eps[k]
            minus = v[0] - self.sigma * self._eps[k]
            out[k] = torch.cat([plus, minus], 0)
        return out

    # ---------------------------------------------------------------- acting
    def act_batch(self, obs):
        """obs (B, obs_dim) -> actions (B, act_dim); member i acts for env i."""
        params = self._member_params()
        keys = list(params.keys())
        vals = [params[k] for k in keys]

        def fwd(p, o):
            kw = dict(zip(keys, p))
            a, _, _ = functional_call(self.policy, kw, (o.unsqueeze(0),))
            return a.squeeze(0)

        with torch.no_grad():
            return vmap(fwd, in_dims=(0, 0), randomness="different")(vals, obs)

    # ------------------------------------------------------------- interface
    def collect(self, env, reward):
        B = env.num_envs
        self._refresh(B)
        obs = env.reset()
        reward.on_reset(torch.zeros(B, dtype=torch.bool, device=self.device),
                        env.initial_signals())
        fitness = torch.zeros(B, device=self.device)
        success = torch.zeros(B, device=self.device)
        with torch.no_grad():
            for _ in range(self.window):
                a = self.act_batch(obs)
                obs2, sig, done, info = env.step(a)
                if "episode_x" in info:
                    sig = dict(sig, episode_x=info["episode_x"])
                r, _ = reward(sig, done)
                fitness += r
                success = torch.where((done > 0) & (info["episode_success"] > 0),
                                      torch.ones_like(success), success)
                obs = obs2
                if done.any():
                    obs = env.reset_done(done)
        return {"returns": fitness, "success": success,
                "consumed": B * self.window, "steps_used": B * self.window}

    def _stack_state(self):
        params, buffers = stack_module_state([self.policy])
        self._stacked = {**{k: v.detach().clone() for k, v in params.items()},
                         **{k: v.detach().clone() for k, v in buffers.items()}}

    def update(self, data):
        F = data["returns"]
        B = F.shape[0]
        half = B // 2
        with torch.no_grad():
            if self.rank_shape:
                order = F.argsort().argsort().float()
                Fw = order / max(B - 1, 1) - 0.5          # ranks -> [-0.5, 0.5]
            else:
                Fw = (F - F.mean()) / (F.std() + 1e-8)
            wp, wm = Fw[:half], Fw[half:]                  # plus, minus halves
            shape = lambda k: (-1, *([1] * (self._eps[k].dim() - 1)))
            grads = {k: (((wp.view(shape(k)) - wm.view(shape(k))) * self._eps[k]).sum(0))
                     / (B * self.sigma) for k in self._eps}
        self.opt.zero_grad(set_to_none=True)
        for name, p in self.policy.named_parameters():
            if name in grads:
                p.grad = grads[name].clone().to(p.dtype).to(self.device)
        self.opt.step()
        self._stack_state()
        self._gen += 1
        return {"es_fitness": float(F.mean()),
                "episode_success": float(data["success"].mean()),
                "generations": float(self._gen)}
