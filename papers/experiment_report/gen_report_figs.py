"""Extra vector figures for the formal experiment report (v2 robot suite).

Outputs (PDF, vector, Chinese labels; numbering/captions live in main.tex):
  figs/fig3d_algo_task.pdf      3-D bars: 12 algorithms x 3 tasks, dense final
  figs/fig3d_paradigm.pdf       3-D bars: {PPO,TRPO,GRPO-C} x 3 tasks x 4 paradigms
Data source: results/headline.json (dense-only, post-v2.1 fix).
Run from repo root:  python papers/experiment_report/gen_report_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

ENVS = ["nav", "hopper", "grasp", "graspseq", "walker"]
ENV_CN = {"nav": "E1 导航", "hopper": "E2 单足", "grasp": "E3 抓取",
          "graspseq": "E4 连续抓取", "walker": "E5 双足行走"}
ALGO_ORDER = ["sac", "sac_tr", "ppg", "trpo", "ppo_kl", "ppo_rb", "ppo",
              "ppo_ent", "ppo_novc", "espo", "tr_ppo_rb", "grpo",
              "a2c", "td3", "dapo", "es"]
ALGO_CN = {"sac": "SAC", "sac_tr": "SAC-TR", "ppg": "PPG", "trpo": "TRPO",
           "ppo_kl": "PPO-KL", "ppo_rb": "PPO-RB", "ppo": "PPO-Clip",
           "ppo_ent": "PPO-Ent", "ppo_novc": "PPO-NoVC", "espo": "ESPO",
           "tr_ppo_rb": "TR-PPO-RB", "grpo": "GRPO-C", "a2c": "A2C",
           "td3": "TD3", "dapo": "DAPO-C", "es": "ES"}
RW_CN = {"sparse": "稀疏 sparse", "dense": "稠密 dense",
         "pbs": "势函数 PBS", "curriculum": "课程 curriculum"}


def load():
    with open(os.path.join(HERE, "..", "..", "results", "headline.json"),
              encoding="utf-8") as f:
        return json.load(f)


def style3d(ax):
    ax.set_zlim(0, 1.02)
    ax.set_zlabel("最终指标(3 种子均值)", fontsize=11, labelpad=14)
    ax.view_init(elev=38, azim=-55)
    ax.zaxis.pane.fill = False
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane):
        pane.set_alpha(0.06)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150, bbox_inches="tight", pad_inches=0.15)


def fig3d_algo_task(h):
    Z = np.array([[dict((a, v) for a, v, *_ in h["dense_rankings"][e])[algo]
                   for e in ENVS] for algo in ALGO_ORDER])
    fig = plt.figure(figsize=(11, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    nx, ny = len(ALGO_ORDER), len(ENVS)
    # wide row gaps (y pitch 3) reduce front-row occlusion
    xpos, ypos = np.meshgrid(np.arange(nx), np.arange(ny) * 3.0, indexing="ij")
    xpos, ypos = xpos.ravel(), ypos.ravel()
    dz = Z.ravel()
    colors = plt.cm.viridis(dz / 1.0)
    ax.bar3d(xpos, ypos, np.zeros_like(dz), 0.55, 0.9, dz,
             color=colors, shade=True, alpha=0.9)
    ax.set_xticks(np.arange(nx) + 0.27)
    ax.set_xticklabels([ALGO_CN[a] for a in ALGO_ORDER], rotation=38,
                       ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(ny) * 3.0 + 0.45)
    ax.set_yticklabels([ENV_CN[e] for e in ENVS], fontsize=10)
    style3d(ax)
    ax.set_box_aspect((3.2, 1.15, 1))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.16, aspect=26)
    cb.ax.tick_params(labelsize=8)
    ax.set_zticks([0.0, 0.5, 1.0])
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    save(fig, "fig3d_algo_task")
    plt.close(fig)


def fig3d_paradigm(h):
    rws = ["dense", "curriculum", "pbs", "sparse"]
    algos = ["ppo", "trpo", "grpo"]
    palette = {"dense": "#2b8a3e", "curriculum": "#1c7ed6",
               "pbs": "#f08c00", "sparse": "#c92a2a"}
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, dz, cs = [], [], [], []
    for ai, algo in enumerate(algos):
        for ei, env in enumerate(ENVS):
            axis = dict(h["reward_axis"][str(ei)][algo])
            for ri, rw in enumerate(rws):
                xs.append(ai * 6.0 + ri * 1.15)
                ys.append(ei * 3.0)
                dz.append(axis[rw])
                cs.append(palette[rw])
    xs, ys, dz = map(np.array, (xs, ys, dz))
    ax.bar3d(xs, ys, np.zeros_like(dz), 0.8, 0.9, dz, color=cs,
             shade=True, alpha=0.9)
    ax.set_xticks([a * 6.0 + 1.7 for a in range(len(algos))])
    ax.set_xticklabels(["PPO-Clip", "TRPO", "GRPO-C"], fontsize=11)
    ax.set_yticks(np.arange(len(ENVS)) * 3.0 + 0.45)
    ax.set_yticklabels([ENV_CN[e] for e in ENVS], fontsize=10)
    style3d(ax)
    ax.set_box_aspect((3.2, 1.15, 1))
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[rw]) for rw in rws]
    ax.legend(handles, [RW_CN[rw] for rw in rws], ncol=4, loc="upper center",
              fontsize=9, title="奖励范式", title_fontsize=9,
              bbox_to_anchor=(0.5, -0.04), frameon=False)
    save(fig, "fig3d_paradigm")
    plt.close(fig)


def main():
    h = load()
    fig3d_algo_task(h)
    fig3d_paradigm(h)
    print("figures ->", FIGS)


if __name__ == "__main__":
    main()
