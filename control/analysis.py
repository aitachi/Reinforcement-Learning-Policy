"""Aggregate results -> publication figures + LaTeX tables.

  python3 control/analysis.py            # reads results/, writes results/figs/
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

# numpy 2.x renamed trapz -> trapezoid; support both
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
FIGS = os.path.join(RESULTS, "figs")

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "lines.linewidth": 1.3,
})

ENV_TITLES = {"nav": "E1 Nav-Obstacle", "hopper": "E2 Hopper-Locomotion",
              "grasp": "E3 Reach-Grasp", "graspseq": "E4 GraspSeq-Cycles",
              "walker": "E5 Walker-Biped"}
ENV_METRIC = {"nav": ("eval_success", "success rate"),
              "hopper": ("eval_outcome", "displacement /10 m"),
              "grasp": ("eval_success", "lift success rate"),
              "graspseq": ("eval_outcome", "cycles completed /3"),
              "walker": ("eval_outcome", "displacement /10 m")}
ALGOS = ["ppo", "ppo_kl", "ppo_rb", "tr_ppo_rb", "espo", "ppg",
         "trpo", "a2c", "grpo", "dapo", "sac", "td3",
         "sac_tr", "ppo_ent", "ppo_novc", "es"]
LABELS = {"ppo": "PPO", "ppo_kl": "PPO-KL", "ppo_rb": "PPO-RB",
          "tr_ppo_rb": "TR-PPO-RB", "espo": "ESPO", "ppg": "PPG",
          "trpo": "TRPO", "a2c": "A2C", "grpo": "GRPO-C", "dapo": "DAPO-C",
          "sac": "SAC", "td3": "TD3", "tr": "TRPO",
          "sac_tr": "SAC-TR", "ppo_ent": "PPO-Ent", "ppo_novc": "PPO-NoVC",
          "es": "ES", "ppo_e01": "PPO(eps=0.1)", "ppo_e03": "PPO(eps=0.3)"}
FAMILY = {"ppo": "#1f77b4", "ppo_kl": "#4c9be8", "ppo_rb": "#7fc3ff",
          "tr_ppo_rb": "#0d5a8f", "espo": "#9ecae1", "ppg": "#3182bd",
          "trpo": "#d62728", "a2c": "#ff9896", "grpo": "#2ca02c",
          "dapo": "#98df8a", "sac": "#9467bd", "td3": "#c5b0d5",
          "sac_tr": "#b159bd", "ppo_ent": "#7fb2b8", "ppo_novc": "#497a80",
          "es": "#8c564b", "ppo_e01": "#aaaaaa", "ppo_e03": "#666666"}
ENV_TOTAL = {"nav": 400_000, "hopper": 600_000, "grasp": 500_000,
             "graspseq": 800_000, "walker": 600_000}


def load_runs(env, reward=None, algo=None):
    runs = {}
    for f in glob.glob(os.path.join(RESULTS, env, "*.csv")):
        import csv
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        r0 = rows[0]
        if reward and r0["reward"] != reward:
            continue
        if algo and r0["algo"] != algo:
            continue
        key = (r0["algo"], r0["reward"], int(r0["seed"]))
        steps = np.array([float(r["steps"]) for r in rows])
        runs[key] = {m: np.array([float(r[m]) for r in rows]) for m in
                     ("eval_success", "eval_outcome", "eval_return", "eval_len")}
        runs[key]["steps"] = steps
        runs[key]["wall"] = float(rows[-1]["wall"])
    return runs


def curve_band(runs, algo, reward, metric, env):
    """Interpolate seed curves onto a common grid -> mean, 95% CI."""
    seeds = [k for k in runs if k[0] == algo and k[1] == reward]
    if not seeds:
        return None
    grid = np.linspace(0, ENV_TOTAL[env], 120)
    ys = []
    for k in seeds:
        s, y = runs[k]["steps"], runs[k][metric]
        ys.append(np.interp(grid, s, y))
    ys = np.stack(ys)
    ci = 1.96 * ys.std(0) / np.sqrt(max(len(ys), 1))
    return grid, ys.mean(0), ci, len(ys)


def fig_learning_curves(reward="dense", algos=None, fname=None, ncols=3):
    algos = algos or ALGOS
    envs = [e for e in ("nav", "hopper", "grasp")
            if any(k[1] == reward for k in load_runs(e, reward))]
    fig, axes = plt.subplots(1, len(envs), figsize=(3.1 * len(envs), 2.5))
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        runs = load_runs(env, reward)
        metric, ylab = ENV_METRIC[env]
        for a in algos:
            band = curve_band(runs, a, reward, metric, env)
            if band is None:
                continue
            g, m, ci, n = band
            ax.plot(g / 1000, m, label=LABELS[a], color=FAMILY[a])
            ax.fill_between(g / 1000, m - ci, m + ci, color=FAMILY[a], alpha=0.15, lw=0)
        ax.set_title(f"{ENV_TITLES[env]} ({reward})")
        ax.set_xlabel("env steps (k)")
        ax.set_ylabel(ylab)
    axes[0].legend(ncol=1, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname or f"curves_{reward}.pdf"))
    plt.close(fig)


def final_metrics(env, reward="dense"):
    runs = load_runs(env, reward)
    metric, _ = ENV_METRIC[env]
    out = []
    for a in ALGOS:
        finals = [runs[k][metric][-5:].mean() for k in runs if k[0] == a]
        walls = [runs[k]["wall"] for k in runs if k[0] == a]
        if not finals:
            continue
        auc = []
        for k in runs:
            if k[0] != a:
                continue
            grid = np.linspace(0, 1, 100)
            y = np.interp(grid, runs[k]["steps"] / ENV_TOTAL[env], runs[k][metric])
            auc.append(_trapezoid(y, grid))
        out.append({"algo": a, "final": np.mean(finals), "final_std": np.std(finals),
                    "auc": np.mean(auc), "auc_std": np.std(auc),
                    "cv_seeds": np.std(finals) / (abs(np.mean(finals)) + 1e-9),
                    "wall": np.mean(walls), "n": len(finals)})
    return sorted(out, key=lambda r: -r["final"])


def fig_final_bars(reward="dense"):
    envs = [e for e in ("nav", "hopper", "grasp")
            if any(k[1] == reward for k in load_runs(e, reward))]
    fig, axes = plt.subplots(1, len(envs), figsize=(3.1 * len(envs), 2.4))
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        fm = final_metrics(env, reward)
        names = [LABELS[r["algo"]] for r in fm]
        vals = [r["final"] for r in fm]
        errs = [r["final_std"] for r in fm]
        colors = [FAMILY[r["algo"]] for r in fm]
        ax.bar(range(len(fm)), vals, yerr=errs, color=colors, capsize=2)
        ax.set_xticks(range(len(fm)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        metric, ylab = ENV_METRIC[env]
        ax.set_title(ENV_TITLES[env])
        ax.set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"final_bars_{reward}.pdf"))
    plt.close(fig)


def fig_reward_paradigms():
    algos = ["ppo", "trpo", "grpo"]
    rewards = ["sparse", "dense", "pbs", "curriculum"]
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 6.2))
    for i, env in enumerate(("nav", "hopper", "grasp")):
        metric, _ = ENV_METRIC[env]
        for j, rw in enumerate(rewards):
            ax = axes[i, j]
            runs = load_runs(env, rw)
            for a in algos:
                band = curve_band(runs, a, rw, metric, env)
                if band is None:
                    continue
                g, m, ci, n = band
                ax.plot(g / 1000, m, label=LABELS[a], color=FAMILY[a])
                ax.fill_between(g / 1000, m - ci, m + ci, color=FAMILY[a], alpha=0.15, lw=0)
            if i == 0:
                ax.set_title(rw)
            if j == 0:
                ax.set_ylabel(f"{ENV_TITLES[env]}\n{metric.replace('eval_', '')}")
    axes[0, 0].legend(loc="lower right", framealpha=0.9)
    for ax in axes[-1]:
        ax.set_xlabel("env steps (k)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "reward_paradigm_grid.pdf"))
    plt.close(fig)


def fig_radar():
    envs = [e for e in ("nav", "hopper", "grasp")]
    metrics = {}
    for env in envs:
        for r in final_metrics(env, "dense"):
            metrics.setdefault(r["algo"], {})[env] = r
    algos = [a for a in ALGOS if a in metrics and len(metrics[a]) == len(envs)]
    axes_names = [ENV_TITLES[e] + " perf" for e in envs] + ["sample eff (AUC)", "stability"]
    norm = {a: [] for a in algos}
    for env in envs:
        vals = np.array([metrics[a][env]["final"] for a in algos])
        v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        for i, a in enumerate(algos):
            norm[a].append(v[i])
    auc = np.array([np.mean([metrics[a][e]["auc"] for e in envs]) for a in algos])
    auc_n = (auc - auc.min()) / (auc.max() - auc.min() + 1e-9)
    stab = np.array([-np.mean([metrics[a][e]["cv_seeds"] for e in envs]) for a in algos])
    stab_n = (stab - stab.min()) / (stab.max() - stab.min() + 1e-9)
    for i, a in enumerate(algos):
        norm[a] += [auc_n[i], stab_n[i]]

    angles = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.2, 4.0), subplot_kw={"polar": True})
    for a in algos:
        vals = norm[a] + norm[a][:1]
        ax.plot(angles, vals, label=LABELS[a], color=FAMILY[a])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_names, fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "radar.pdf"))
    plt.close(fig)


def tex_tables():
    all_lines = []
    for env in ("nav", "hopper", "grasp"):
        fm = final_metrics(env, "dense")
        metric, _ = ENV_METRIC[env]
        lines = [f"% {ENV_TITLES[env]} ({metric})",
                 "\\begin{tabular}{lccccc}",
                 "\\toprule",
                 "Algorithm & Final (mean$\\pm$std) & AUC & Seed CV & Wall (s) & $n$ \\\\",
                 "\\midrule"]
        for r in fm:
            lines.append(f"{LABELS[r['algo']]} & {r['final']:.3f}$\\pm${r['final_std']:.3f} & "
                         f"{r['auc']:.3f}$\\pm${r['auc_std']:.3f} & {r['cv_seeds']:.2f} & "
                         f"{r['wall']:.0f} & {r['n']} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(FIGS, f"tables_algo_{env}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        all_lines += lines + [""]

    # reward paradigm table (final metric per env x algo x reward)
    rows = []
    header = ["env", "algo"] + ["sparse", "dense", "pbs", "curriculum"]
    for env in ("nav", "hopper", "grasp"):
        for a in ("ppo", "trpo", "grpo"):
            row = [env, LABELS[a]]
            for rw in ("sparse", "dense", "pbs", "curriculum"):
                fm = None
                runs = load_runs(env, rw)
                metric, _ = ENV_METRIC[env]
                finals = [runs[k][metric][-5:].mean() for k in runs
                          if k[0] == a and k[1] == rw]
                row.append(f"{np.mean(finals):.3f}$\\pm${np.std(finals):.3f}" if finals else "--")
            rows.append(row)
    with open(os.path.join(FIGS, "tables_reward.tex"), "w", encoding="utf-8") as f:
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        f.write("\n".join(" & ".join(r) + " \\\\" for r in rows))
    print("tables written")


def summary_csv():
    import csv as csvmod
    out = []
    for f in glob.glob(os.path.join(RESULTS, "*", "*.json")):
        d = json.load(open(f))
        out.append({k: d.get(k) for k in
                    ("env", "algo", "reward", "seed", "total_steps", "wall_s",
                     "eval_success", "eval_outcome", "eval_return", "eval_len")})
    if out:
        with open(os.path.join(RESULTS, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"summary.csv: {len(out)} runs")


def main():
    os.makedirs(FIGS, exist_ok=True)
    for rw in ("dense", "sparse", "pbs", "curriculum"):
        fig_learning_curves(rw)
        fig_final_bars(rw)
    fig_reward_paradigms()
    fig_radar()
    tex_tables()
    summary_csv()
    print("figures in", FIGS)


if __name__ == "__main__":
    main()
