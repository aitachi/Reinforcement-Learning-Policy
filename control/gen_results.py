"""Distill results into headline numbers + draft findings for the papers.

  python3 control/gen_results.py     -> results/headline.json + findings_draft.md
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

# numpy 2.x renamed trapz -> trapezoid; support both
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

ENV_TOTAL = {"nav": 400_000, "hopper": 600_000, "grasp": 500_000}
METRIC = {"nav": "eval_success", "hopper": "eval_outcome", "grasp": "eval_success"}
ENV_NAME = {"nav": "E1 Nav", "hopper": "E2 Hopper", "grasp": "E3 Grasp"}


def load_all():
    import csv
    runs = []
    for f in glob.glob(os.path.join(RESULTS, "*", "*.csv")):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        r0 = rows[0]
        m = METRIC[r0["env"]]
        y = np.array([float(x[m]) for x in rows])
        s = np.array([float(x["steps"]) for x in rows])
        grid = np.linspace(0, ENV_TOTAL[r0["env"]], 100)
        yi = np.interp(grid, s, y)
        runs.append({"env": r0["env"], "algo": r0["algo"], "reward": r0["reward"],
                     "seed": int(r0["seed"]), "final": float(y[-5:].mean()),
                     "auc": float(_trapezoid(yi) / 100.0),
                     "wall": float(rows[-1]["wall"]),
                     "steps_to_half": (float(s[np.argmax(y >= 0.5 * y[-5:].mean())])
                                       if (y >= 0.5 * y[-5:].mean()).any() else None),
                     "curve": (s.tolist(), y.tolist())})
    return runs


def agg(runs, keys):
    g = defaultdict(list)
    for r in runs:
        g[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, v in g.items():
        f = [x["final"] for x in v]
        a = [x["auc"] for x in v]
        out["|".join(str(x) for x in k)] = {
            "final_mean": float(np.mean(f)), "final_std": float(np.std(f)),
            "auc_mean": float(np.mean(a)), "auc_std": float(np.std(a)),
            "wall_mean": float(np.mean([x["wall"] for x in v])),
            "n": len(v),
            "steps_to_half_med": float(np.median([x["steps_to_half"] for x in v
                                                  if x["steps_to_half"]])) if any(x["steps_to_half"] for x in v) else None}
    return out


def main():
    runs = load_all()
    dense_runs = [r for r in runs if r["reward"] == "dense"]
    by_env_algo = agg(dense_runs, ["env", "algo"])     # dense-only rankings
    by_env_algo_rw = agg(runs, ["env", "algo", "reward"])
    by_env_rw = agg(runs, ["env", "reward"])

    # dense-only per env ranking
    rankings = {}
    for env in ENV_TOTAL:
        rows = {k.split("|")[1]: v for k, v in by_env_algo.items()
                if k.startswith(env + "|") and k.count("|") == 1}
        rankings[env] = sorted(rows.items(), key=lambda kv: -kv[1]["final_mean"])

    # reward axis: for ppo/trpo/grpo on each env, which paradigm wins
    reward_axis = {}
    for env in ENV_TOTAL:
        reward_axis[env] = {}
        for algo in ("ppo", "trpo", "grpo"):
            key = f"{env}|{algo}"
            paradigms = {k.split("|")[2]: v for k, v in by_env_algo_rw.items()
                         if k.startswith(key + "|")}
            if paradigms:
                reward_axis[env][algo] = sorted(paradigms.items(),
                                                key=lambda kv: -kv[1]["final_mean"])

    # GRPO vs DAPO ablation evidence from logged train metrics
    dapo_ev = {}
    for env in ENV_TOTAL:
        clip_hi, resamp = [], []
        for f in glob.glob(os.path.join(RESULTS, env, "dapo_dense_s*.csv")):
            import csv as csvmod
            rows = list(csvmod.DictReader(open(f)))
            ch = [float(r["clip_high_frac"]) for r in rows if r.get("clip_high_frac")]
            rs = [float(r["resampled_steps"]) for r in rows if r.get("resampled_steps")]
            if ch:
                clip_hi.append(np.mean(ch))
            if rs:
                resamp.append(np.mean(rs))
        if clip_hi:
            dapo_ev[env] = {"clip_high_frac": float(np.mean(clip_hi)),
                            "resampled_steps_mean": float(np.mean(resamp)) if resamp else 0.0}

    headline = {"n_runs": len(runs),
                "dense_rankings": {e: [(a, round(v["final_mean"], 4), round(v["final_std"], 4),
                                        round(v["auc_mean"], 4), round(v["wall_mean"]))
                                       for a, v in rankings[e]] for e in rankings},
                "reward_axis": {e: {a: [(rw, round(v["final_mean"], 4))
                                        for rw, v in reward_axis[env][a]]
                                    for a in reward_axis[env]}
                                for e, env in enumerate(ENV_TOTAL)},
                "dapo_evidence": dapo_ev,
                "by_env_algo": by_env_algo, "by_env_algo_reward": by_env_algo_rw}
    with open(os.path.join(RESULTS, "headline.json"), "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=1, ensure_ascii=False)

    # draft findings text (curated into the papers by hand)
    lines = ["# Findings draft (auto-generated, curate before use)", ""]
    for env, rk in rankings.items():
        lines.append(f"## {ENV_NAME[env]} (dense, final mean±std / AUC / wall)")
        for a, v in rk:
            lines.append(f"- {a}: {v['final_mean']:.3f}±{v['final_std']:.3f} "
                         f"auc={v['auc_mean']:.3f} wall={v['wall_mean']:.0f}s")
        lines.append("")
    for env in ENV_TOTAL:
        lines.append(f"## {ENV_NAME[env]} reward axis")
        for algo, lst in reward_axis[env].items():
            lines.append(f"- {algo}: " + ", ".join(f"{rw}={v['final_mean']:.3f}"
                                                  for rw, v in lst))
        lines.append("")
    with open(os.path.join(RESULTS, "findings_draft.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"runs={len(runs)} -> headline.json / findings_draft.md")


if __name__ == "__main__":
    main()
