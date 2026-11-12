"""GRPO-C and DAPO-C: group-relative policy optimization for robot control.

GRPO (Shao et al., 2024) replaces the learned critic with a group baseline:
sample G rollouts per task instance, advantage = (R_i - mean_G) / std_G with
R the *episode* return (outcome-level supervision).  Here the "task instance"
is the environment's initial state: N start states are sampled, each
replicated G times so the group shares s_0 exactly.

DAPO (Yu et al., 2025) adds four tricks, adapted to control:
  clip-higher      eps_high > eps_low  (0.28 / 0.2): widens the upside band
                   so low-probability (exploratory) actions keep gradient.
  dynamic sampling groups whose returns are all identical (std=0) carry zero
                   gradient; they are resampled (same s_0, fresh policy noise)
                   up to 2x budget, then dropped.
  token-level loss per-step (token->timestep) clipping is already step-level
                   here - noted for faithfulness.
  overlong filter  -> timeout filter: episodes ending by truncation under a
                   sparse outcome reward are masked out of the policy loss.
Both variants consume the SAME env steps as the on-policy family for fair
sample-efficiency comparison (resampled steps count toward the budget).
"""
from __future__ import annotations

import numpy as np
import torch

from .nets import GaussianActor, gaussian_kl


class GRPO:
    needs = "episode"
    kind = "grpo"

    def __init__(self, obs_dim, act_dim, device, cfg):
        self.device = device
        self.policy = GaussianActor(obs_dim, act_dim).to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.get("grpo_lr", 1e-4))
        self.cfg = cfg
        self.N = cfg.get("group_starts", 16)     # distinct initial states per iter
        self.G = cfg.get("group_size", 8)        # rollouts per initial state
        self.eps_low = cfg.get("clip", 0.2)
        self.eps_high = cfg.get("clip_high", 0.2)   # == eps_low for GRPO
        self.kl_coef = cfg.get("grpo_kl_coef", 0.0)
        self.epochs = cfg.get("grpo_epochs", 1)
        self.mb = cfg.get("minibatch", 4096)
        self.filter_timeout = cfg.get("filter_timeout", False)
        self.dynamic = cfg.get("dynamic_sampling", False)
        self.gamma = cfg.get("gamma", 0.99)

    # ----------------------------------------------------------- collection
    def _roll(self, env, reward, B):
        """Roll the full batch from current env states; returns dict of lists."""
        obs = env.observe()
        steps = {"obs": [], "pre": [], "logp": [], "act": [], "rew": [], "active": []}
        R = torch.zeros(B, device=self.device)
        alive = torch.ones(B, device=self.device)
        ended_by = torch.zeros(B, device=self.device)  # 1 timeout, 2 natural
        success = torch.zeros(B, device=self.device)
        disc = torch.ones(B, device=self.device)
        for t in range(env.max_steps):
            a, logp, pre = self.policy.act(obs)
            obs2, sig, done, info = env.step(a)
            r, _ = reward(sig, done)
            steps["obs"].append(obs)
            steps["pre"].append(pre)
            steps["logp"].append(logp)
            steps["act"].append(a)
            steps["rew"].append(r)
            steps["active"].append(alive.clone())
            R += alive * disc * r
            disc *= torch.where(done > 0, torch.zeros_like(disc), self.gamma * torch.ones_like(disc))
            alive = alive * (1.0 - done.float())
            success = torch.where(done & (info["episode_success"] > 0), torch.ones_like(success), success)
            ended_by = torch.where(done, torch.where(info["done"] & (sig.get("timeout", torch.zeros_like(done)) > 0.5),
                                                     torch.ones_like(ended_by) * 1, torch.ones_like(ended_by) * 2), ended_by)
            if float(alive.sum()) < 0.5:
                break
            obs = obs2
        T = len(steps["obs"])
        data = {k: torch.stack(v) for k, v in steps.items()}     # each (T, B, ...)
        data["returns"] = R
        data["success"] = success
        data["ended_by"] = ended_by
        # fair budget accounting: only active env-steps count
        data["steps_used"] = int(data["active"].sum().item())
        return data

    def collect(self, env, reward):
        B = self.N * self.G
        env.reset()
        st = env.get_state()
        tiled = {f: v[:self.N].repeat_interleave(self.G, dim=0) for f, v in st.items()}
        env.set_state(tiled)
        reward.on_reset(torch.ones(B, dtype=torch.bool, device=self.device),
                        env.initial_signals())

        data = self._roll(env, reward, B)
        consumed = data["steps_used"]
        # dynamic sampling (DAPO): replace zero-variance groups, max 2x budget
        rounds = 1
        if self.dynamic:
            while rounds < 2:
                Rg = data["returns"].view(self.N, self.G)
                degenerate = (Rg.std(dim=1) < 1e-6)
                if not bool(degenerate.any()) or consumed >= 2 * data["steps_used"]:
                    break
                env.set_state(self._tile_group_states(st, degenerate))
                reward.on_reset(degenerate.repeat_interleave(self.G),
                                env.initial_signals())
                d2 = self._roll(env, reward, B)
                consumed += d2["steps_used"]
                data = self._merge_groups(data, d2, degenerate)
                rounds += 1
        data["consumed"] = consumed
        data["init_state"] = st
        return data

    def _tile_group_states(self, st, keep_groups):
        idx = torch.nonzero(keep_groups).flatten()
        return {f: v[idx].repeat_interleave(self.G, dim=0) for f, v in st.items()}

    @staticmethod
    def _merge_groups(data, new, keep_groups):
        out = {}
        N, G = len(keep_groups), new["returns"].shape[0] // 1
        T = max(data["obs"].shape[0], new["obs"].shape[0])
        B = data["returns"].shape[0]
        gsel = keep_groups.repeat_interleave(G)
        for k in ("obs", "pre", "logp", "act", "rew", "active"):
            old = data[k]
            newt = new[k]
            pad_o = T - old.shape[0]
            pad_n = T - newt.shape[0]
            o = torch.nn.functional.pad(old, (0, 0) * (old.dim() - 1) + (0, pad_o))
            n = torch.nn.functional.pad(newt, (0, 0) * (newt.dim() - 1) + (0, pad_n))
            m = gsel.unsqueeze(0).expand_as(old[:, :, 0] if old.dim() == 3 else old)
            out[k] = torch.where(m, n, o)
        for k in ("returns", "success", "ended_by"):
            out[k] = torch.where(gsel, new[k], data[k])
        out["steps_used"] = data["steps_used"] + new["steps_used"]
        return out

    # -------------------------------------------------------------- update
    def update(self, data):
        N, G = self.N, self.G
        B = data["returns"].shape[0]
        Rg = data["returns"].view(N, G)
        adv_g = (Rg - Rg.mean(dim=1, keepdim=True)) / (Rg.std(dim=1, keepdim=True) + 1e-4)
        adv = adv_g.view(B)

        obs = data["obs"]            # (T, B, D)
        pre = data["pre"]
        old_logp = data["logp"]
        active = data["active"]
        T = obs.shape[0]
        if self.filter_timeout:
            bad_ep = (data["ended_by"] == 1) & (data["success"] < 0.5)
            active = active * (~bad_ep).unsqueeze(0).float()

        flat = lambda x: x.reshape(T * B, -1) if x.dim() > 2 else x.reshape(T * B)
        f_obs, f_pre, f_lp, f_a = flat(obs), flat(pre), flat(old_logp), flat(active)
        f_adv = adv.unsqueeze(0).expand(T, B).reshape(T * B)
        keep = f_a > 0.5
        f_obs, f_pre, f_lp, f_adv = f_obs[keep], f_pre[keep], f_lp[keep], f_adv[keep]
        n = f_obs.shape[0]
        if n < 32:
            return {"pi_loss": 0.0, "kept": float(n), "ratio_dev": 0.0,
                    "group_std": float(Rg.std(dim=1).mean()), "success": float(data["success"].mean()),
                    "optimizer_steps": 0}

        with torch.no_grad():
            omu, ols = self.policy.dist_params(f_obs)

        recs = []
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for idx in perm.split(self.mb):
                logp, mu, ls, _ = self.policy.evaluate(f_obs[idx], f_pre[idx])
                ratio = (logp - f_lp[idx]).exp()
                clipped = ratio.clamp(1 - self.eps_low, 1 + self.eps_high)
                sur = torch.min(ratio * f_adv[idx], clipped * f_adv[idx]).mean()
                loss = -sur
                if self.kl_coef > 0:
                    kl = gaussian_kl(omu[idx], ols[idx], mu, ls).mean()
                    loss = loss - self.kl_coef * kl
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()
                recs.append({"pi_loss": loss.item(),
                             "ratio_dev": (ratio - 1).abs().mean().item(),
                             "clip_high_frac": (ratio > 1 + self.eps_high).float().mean().item()})
        return {"pi_loss": float(np.mean([r["pi_loss"] for r in recs])),
                "ratio_dev": float(np.mean([r["ratio_dev"] for r in recs])),
                "clip_high_frac": float(np.mean([r["clip_high_frac"] for r in recs])),
                "kept": float(n), "group_std": float(Rg.std(dim=1).mean()),
                "success": float(data["success"].mean()),
                "log_std": self.policy.log_std.mean().item(),
                "optimizer_steps": len(recs)}


class DAPO(GRPO):
    kind = "dapo"

    def __init__(self, obs_dim, act_dim, device, cfg):
        super().__init__(obs_dim, act_dim, device, cfg)
        self.eps_high = cfg.get("clip_high", 0.28)
        self.dynamic = True
        self.filter_timeout = True
