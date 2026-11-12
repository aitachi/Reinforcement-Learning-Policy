"""Unified trainer.

  python3 control/train.py --env nav --algo ppo --reward dense --seed 0

Outputs results/<env>/<algo>_<reward>_s<seed>.csv with one row per eval and a
final .json summary.  All paradigms are evaluated with the SAME deterministic
protocol (fixed eval seed, full task difficulty), so success/outcome metrics
are comparable across reward paradigms even though reward scales are not.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch

from configs import BASE, ENVS_CFG
from envs import ENVS
from rewards import PARADIGMS
from algos import make_algo


def evaluate(eval_env, reward_eval, algo, device):
    """64 deterministic episodes; returns success, outcome, length, return."""
    obs = eval_env.reset()
    B = eval_env.num_envs
    alive = torch.ones(B, device=device, dtype=torch.bool)
    ret = torch.zeros(B, device=device)
    outcome = torch.zeros(B, device=device)
    success = torch.zeros(B, device=device)
    length = torch.zeros(B, device=device)
    reward_eval.on_reset(torch.zeros(B, dtype=torch.bool, device=device),
                         eval_env.initial_signals())
    steps = 0
    while alive.any() and steps < eval_env.max_steps:
        policy = getattr(algo, "policy", None)
        if algo.kind in ("sac", "td3"):
            a = algo.act(obs, deterministic=True) if algo.kind == "td3" else algo.act(obs)
        elif policy is not None:
            a = policy.act(obs, deterministic=True)[0]
        else:
            raise RuntimeError("no policy")
        obs, sig, done, info = eval_env.step(a)
        r, _ = reward_eval(sig, done)
        ret += r.float() * alive.float()
        outcome = torch.where(done & alive, info["episode_outcome"].float(), outcome)
        success = torch.where(done & alive, info["episode_success"].float(), success)
        length += alive.float()
        newly = done & alive
        if newly.any():
            reward_eval.on_reset(newly, eval_env.initial_signals())
        alive = alive & ~done
        steps += 1
    diag = {}
    if "diag" in info:
        diag = {f"diag_{k}": v for k, v in info["diag"].items()}
    return {"eval_success": success.mean().item(),
            "eval_outcome": outcome.mean().item(),
            "eval_len": length.mean().item(),
            "eval_return": ret.mean().item(),
            "eval_return_std": ret.std().item(), **diag}


def run(env_name, algo_name, reward_name, seed, total_steps=None, out_root=None,
        device_name=None, quiet=False):
    cfg = dict(BASE)
    ecfg = ENVS_CFG[env_name]
    total = total_steps or ecfg["total_steps"]
    if device_name is None:
        # hopper/grasp kernels are launch-bound at these batch sizes: CPU wins
        device_name = "cuda" if (torch.cuda.is_available() and env_name not in ("hopper", "grasp")) else "cpu"
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(2)   # more threads hurt small-batch env kernels
    out_root = out_root or os.path.join(os.path.dirname(__file__), "..", "results")
    out_dir = os.path.join(out_root, env_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{algo_name}_{reward_name}_s{seed}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    num_envs = ecfg["num_envs"]
    algo = make_algo(algo_name, ENVS[env_name].obs_dim, ENVS[env_name].act_dim,
                     device, cfg)
    if getattr(algo, "needs", "transition") == "episode":
        num_envs = algo.N * algo.G
    env = ENVS[env_name](num_envs, seed, device)
    eval_env = ENVS[env_name](64, 7777 + seed, device)
    eval_env.set_progress(1.0)
    Reward = PARADIGMS[(env_name, reward_name)]
    reward = Reward(num_envs, device)
    reward_eval = Reward(64, device)

    rows = []
    steps = 0
    t0 = time.time()
    next_eval = ecfg["eval_every"]
    train_metrics = {}
    term_audit = {}

    def maybe_eval():
        nonlocal next_eval
        ev = evaluate(eval_env, reward_eval, algo, device)
        row = {"env": env_name, "algo": algo_name, "reward": reward_name,
               "seed": seed, "steps": steps, "wall": round(time.time() - t0, 1), **ev,
               **{k: round(v, 5) for k, v in train_metrics.items() if isinstance(v, float)}}
        rows.append(row)
        if not quiet:
            print(f"[{env_name}/{algo_name}/{reward_name}/s{seed}] {steps:>8d} "
                  f"succ={ev['eval_success']:.3f} out={ev['eval_outcome']:.3f} "
                  f"ret={ev['eval_return']:.2f} len={ev['eval_len']:.0f} "
                  f"({row['wall']}s)", flush=True)
        next_eval += ecfg["eval_every"]

    def set_progress():
        p = steps / total
        env.set_progress(p)
        if hasattr(reward, "set_progress"):
            reward.set_progress(p)

    # ---------------- transition-style (PPO family / TRPO / A2C) ----------
    if algo.needs == "transition":
        from algos.nets import Rollout
        T = ecfg["rollout"]
        rollout = Rollout(T, num_envs, env.obs_dim, env.act_dim, device)
        obs = env.reset()
        reward.on_reset(torch.zeros(num_envs, dtype=torch.bool, device=device),
                        env.initial_signals())
        while steps < total:
            set_progress()
            rollout.t = 0
            last_done_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
            while not rollout.full:
                with torch.no_grad():
                    a, logp, pre = algo.policy.act(obs)
                    v = algo.critic(obs)
                obs2, sig, done, info = env.step(a)
                if "episode_x" in info:
                    sig = dict(sig, episode_x=info["episode_x"])
                r, terms = reward(sig, done)
                rollout.add(obs, pre, a, logp, r, done.float(), v)
                newly = done & ~last_done_all
                if newly.any():
                    pass
                obs = obs2
                if done.any():
                    obs = env.reset_done(done)
                    reward.on_reset(done, env.initial_signals())
                    last_done_all = last_done_all | done
                steps += num_envs
            train_metrics = algo.update(rollout, obs)
            term_audit = {k: float(v.mean()) for k, v in terms.items()}
            while steps >= next_eval:
                maybe_eval()
    # ---------------- episode-group style (GRPO / DAPO) --------------------
    elif algo.needs == "episode":
        while steps < total:
            set_progress()
            data = algo.collect(env, reward)
            train_metrics = algo.update(data)
            train_metrics["episode_return_mean"] = float(data["returns"].mean())
            train_metrics["episode_success"] = float(data["success"].mean())
            train_metrics["resampled_steps"] = float(data["consumed"] - data["steps_used"])
            steps += data["consumed"]
            while steps >= next_eval:
                maybe_eval()
    # ---------------- step style (SAC) -------------------------------------
    elif algo.needs == "step":
        obs = env.reset()
        reward.on_reset(torch.zeros(num_envs, dtype=torch.bool, device=device),
                        env.initial_signals())
        collect_metrics = {}
        while steps < total:
            set_progress()
            a = algo.act(obs)
            obs2, sig, done, info = env.step(a)
            if "episode_x" in info:
                sig = dict(sig, episode_x=info["episode_x"])
            r, terms = reward(sig, done)
            algo.store(obs, a, r.float(), obs2, done.float())
            obs = obs2
            if done.any():
                obs = env.reset_done(done)
                reward.on_reset(done, env.initial_signals())
            m = algo.update_if_ready()
            if m:
                collect_metrics = m
            steps += num_envs
            term_audit = {k: float(v.mean()) for k, v in terms.items()}
            if steps >= next_eval:
                train_metrics = collect_metrics
                maybe_eval()

    # ---------------- finalize ---------------------------------------------
    final = evaluate(eval_env, reward_eval, algo, device)
    summary = {"env": env_name, "algo": algo_name, "reward": reward_name,
               "seed": seed, "total_steps": steps, "wall_s": round(time.time() - t0, 1),
               **final, "last_train_metrics": train_metrics,
               "reward_term_audit": term_audit}
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    with open(csv_path.replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    if not quiet:
        print(f"DONE {csv_path} final_succ={final['eval_success']:.3f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENVS))
    ap.add_argument("--algo", required=True)
    ap.add_argument("--reward", default="dense",
                    choices=["sparse", "dense", "pbs", "curriculum"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(args.env, args.algo, args.reward, args.seed, args.steps, args.out,
        args.device, args.quiet)


if __name__ == "__main__":
    main()
