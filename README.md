# Reinforcement-Learning-Policy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

**An open, two-track comparison of policy-optimization algorithms: the current
v2 suite targets robot control; the preserved v1 track covers LLM
post-training.**

强化学习策略优化对比项目:当前 **v2** 聚焦**机器人控制**,完整保留的 **v1** 为大模型后训练研究。

**Language / 语言:** [English](README.md) | [中文](README_CN.md)

---

## Track map — read this first / 双轨道导航

This repository contains **two clearly separated research tracks**. They share
one question — *how do the trust-region proximal family and the group-relative
family behave under controlled, budget-matched conditions, and how does the
answer depend on the reward signal?* — but they live in different domains, in
different directories, and are documented separately.

| Track | Status | Domain | Scope | Entry |
|---|---|---|---|---|
| **v2 — Robot Control Suite** | **current, actively developed** | robot control: obstacle navigation, monopod locomotion, reach–grasp–lift | **12 algorithms × 3 tasks × 4 reward paradigms · 189 runs · 2 papers** | [`control/`](control/) · [`papers/`](papers/) · [`results/`](results/) |
| **v1 — LLM Post-Training Study** | original release, complete & frozen | LLM alignment on Qwen2-1.5B | PPO / DPO / GRPO / DAPO implementations + IEEE survey (EN/CN/JA) | [`v1-llm-post-training/`](v1-llm-post-training/README.md) |

- **Go to [Part I (v2)](#part-i--robot-control-suite-v2-current)** for the robot-control comparison — this is where new work happens.
- **Go to [Part II (v1)](#part-ii--llm-post-training-study-v1-preserved)** for a summary of the original LLM study; its full 600-line document now lives in [`v1-llm-post-training/README.md`](v1-llm-post-training/README.md).

---

# Part I · Robot Control Suite (v2, current)

A pure-PyTorch, GPU-batched benchmark measuring **twelve** policy-optimization
algorithms on three self-contained robot tasks under four swappable reward
paradigms, with a fairness protocol (identical budgets, networks, optimizers,
seeds) and per-term reward auditing.

## Quick start

```bash
# single run
python control/train.py --env nav --algo ppo --reward dense --seed 0
# full grid, resumable
python control/sweep.py --axis all --workers 2
# publication figures + LaTeX tables
python control/analysis.py
# headline numbers + findings draft
python control/gen_results.py
# environment & algorithm self-tests
python control/selftest_envs.py && python control/selftest_algos.py
```

No MuJoCo / external physics dependency — everything runs on PyTorch tensors.

## Environments (E1–E3)

| Task | Mechanism | Obs/Act | Episode | Success criterion | Budget |
|---|---|---|---|---|---|
| **E1 `nav`** | holonomic robot crossing a random obstacle field (8-ray lidar obs) | 14 / 2 | 150 | reach goal (≤0.06), collision terminates | 400k steps |
| **E2 `hopper`** | 5-DOF planar monopod, series-elastic passive knee (k=18), assembled from analytic rigid-body Jacobians | 14 / 2 | 250 | displace 3 m without falling | 600k steps |
| **E3 `grasp`** | velocity-controlled 3-link arm, kinematic grasp with hysteresis, lift-to-success | 16 / 4 | 160 | hold object 15 cm up | 500k steps |

## Reward paradigms (`control/rewards/paradigms.py`)

Each task ships **sparse / dense / potential-based (PBS) / curriculum** variants
(12 audited reward functions in total). Task signals are decoupled from
rewards, and every reward term is logged at training time (reward-hacking
telemetry — see the documented grasp-flicker case study in paper 2).

## Algorithms (`control/algos/`, 12)

| Family | Algorithms |
|---|---|
| PPO proximal (method switch) | **PPO-Clip, PPO-KL** (adaptive), **PPO-RB** (rollback), **TR-PPO-RB** (trust-region rollback), **ESPO** (early-stop) |
| Phasic / classical on-policy | **PPG, TRPO** (CG + backtracking), **A2C** |
| Group-relative (from LLM RL) | **GRPO-C, DAPO-C** (episode-outcome baselines, clip-higher, dynamic sampling, timeout filtering) |
| Off-policy anchors | **SAC, TD3** |

## Headline results (dense reward, mean of 3 seeds, 189 runs)

| Algorithm | E1 Nav success | E2 Hopper displ. | E3 Grasp lift | Wall-clock note |
|---|---|---|---|---|
| SAC | **0.830** | **0.899** | 0.629 | 5–8× slower (4403 s on E2) |
| PPG | 0.673 | 0.020 | 0.998 | 39 s on E1 |
| TRPO | 0.669±0.02 | 0.681 | **1.000±0.00** | CG+line-search cost |
| PPO-KL | 0.658 | 0.785 | 0.992 | |
| PPO-RB | 0.655 | 0.669 | 0.995 | |
| PPO-Clip | 0.652 | 0.620 | 0.996 | ±0.17 seed scatter on E2 |
| ESPO | 0.648 | 0.755 | 0.568 | |
| TR-PPO-RB | 0.625 | 0.754 | 0.694 | |
| GRPO-C | 0.557±0.07 | 0.017 | 0.000 | mid-pack only on E1 |
| A2C | 0.547 | 0.027 | 0.059 | |
| TD3 | 0.434 | 0.167 | 0.000 | seed lottery |
| DAPO-C | 0.324 | 0.015 | 0.000 | |

**Key findings**

1. **Trust-region safeguards decide reliability.** On contact-rich E2, the
   variance-controlled variants (adaptive-KL / early-stop / trust-region
   rollback) hold seed spread ≤ ±0.12 while plain clipping scatters to ±0.17.
2. **TRPO is quietly excellent.** Under dense rewards it takes E3 with a
   seed-perfect 1.000 and places third on E1 — earlier "high variance"
   impressions were an artifact of mixing reward paradigms in aggregation
   (fixed in `control/gen_results.py`; dense-only numbers above).
3. **SAC buys peak performance on E1+E2 at 5–8× wall-clock cost; TD3 is a
   seed lottery** (0.43±0.20 on E1, 0 on E3).
4. **Group-relative methods (GRPO-C/DAPO-C) only fit outcome-shaped tasks.**
   Mid-pack on the most outcome-shaped task (E1 nav), near-zero on
   locomotion and multi-stage manipulation; notably GRPO-C under *sparse*
   rewards on E2 (0.109) beats its own dense result (0.017) — clean terminal
   credit suits group baselines.
5. **Reward paradigm × algorithm interaction is real:** PBS rescues
   sparse-reward training on E1/E3 but collapses on E2 (γ<1 potential
   distortion); dense multi-term returns actually *pollute* the group
   baseline.

Full analysis: the two papers below + `results/findings_draft.md`.

## Papers (built from these experiments)

| Paper | Title | Contents |
|---|---|---|
| [`papers/paper1_algo_comparison/`](papers/paper1_algo_comparison/main.pdf) | *From Trust Regions to Group Baselines: A Unified Empirical Study of Proximal Policy Optimization Families for Robot Control* | 12 algorithms × 3 tasks × 3 seeds + reward axis, IEEE 6 pp, PDF included |
| [`papers/paper2_reward_design/`](papers/paper2_reward_design/main.pdf) | *Reward Paradigm Design for Robot Policy Optimization* | 12 audited reward functions, paradigm × algorithm interaction, documented reward-hacking case study, IEEE 4 pp, PDF included |

Rebuild both with `papers\build_papers.bat` (MiKTeX, zero-warning recipe).

## Results layout

```
results/
  <env>/<algo>_<reward>_s<seed>.csv   # per-eval learning curves (189 runs)
  <env>/<algo>_<reward>_s<seed>.json  # final summary + reward-term audit
  figs/                               # publication figures (PDF) + .tex tables
  headline.json / findings_draft.md   # distilled headline numbers
```

---

# Part II · LLM Post-Training Study (v1, preserved)

The original study that founded this repository: complete implementations of
**PPO, DPO, GRPO, DAPO** for LLM alignment on Qwen2-1.5B, with an IEEE-style
survey (EN/CN), algorithm derivations, trained checkpoints and an AIME-2024
ablation (dynamic sampling +8 pts, overlong filtering +6 pts).

**Everything of v1 lives in [`v1-llm-post-training/`](v1-llm-post-training/README.md) and runs from that directory** —
its full README (with the complete algorithm treatise), Japanese edition,
`algorithms/`, `docs/` (IEEE LaTeX sources + PDFs), experiment outputs and
checkpoints:

```bash
cd v1-llm-post-training
pip install -r requirements.txt
python run_comparison.py        # four-algorithm comparison
```

---

# Repository layout

```
Reinforcement-Learning-Policy/
│
├── README.md / README_CN.md        # this file — v2-first, two-track navigation
├── PROJECT_STRUCTURE.md            # annotated directory map (both tracks)
├── LICENSE / requirements.txt / .gitignore
│
├── control/                        # [v2] robot-control suite
│   ├── envs/                       #   nav_obstacle / hopper_lite / reach_grasp
│   ├── rewards/paradigms.py        #   sparse / dense / pbs / curriculum × 3 tasks
│   ├── algos/                      #   ppo_family / trpo / grpo_dapo / sac / td3 / nets
│   ├── train.py                    #   unified entry
│   ├── sweep.py                    #   full-grid scheduler (resumable)
│   ├── analysis.py                 #   publication figures + LaTeX tables
│   ├── gen_results.py              #   headline.json + findings_draft.md
│   ├── configs.py                  #   fairness-protocol hyper-parameters
│   └── selftest_envs.py / selftest_algos.py
│
├── papers/                         # [v2] two papers (IEEEtran, PDFs included)
│   ├── paper1_algo_comparison/     #   algorithm comparison
│   ├── paper2_reward_design/       #   reward paradigm design
│   └── build_papers.bat
│
├── results/                        # [v2] 189 runs + figures + distilled numbers
│
└── v1-llm-post-training/           # [v1] LLM post-training study (frozen)
    ├── README.md / README_CN.md / README_JA.md   # full treatise (EN/CN/JA)
    ├── algorithms/                 #   ppo / dpo / grpo / dapo trainers
    ├── docs/                       #   IEEE survey (PDF + LaTeX), deep dives, figures
    ├── data / models / checkpoints / scripts / src
    ├── DAPO_GRPO_PPO_DPO/          #   four-algorithm experiment outputs + weights
    ├── GRPO_PPO_DPO/               #   earlier three-algorithm outputs
    ├── run_comparison.py / requirements.txt
    └── DAPO.pdf / DeepSeek_R1.pdf / analysis reports  # reference papers & notes
```

# Version history

| Version | Date | Contents |
|---|---|---|
| v1 (3.0.0) | 2025-04 | Initial public release: PPO/DPO/GRPO/DAPO LLM study (originally `deepseek_r1_qwen2-1.5b`, later `PPOvDPOvGRPOvDAPO`) |
| v2 | 2026-11 | Robot-control suite: 12 algorithms × 3 tasks × 4 reward paradigms, 189 runs, two papers |
| v2.1 | 2026-11-14 | This restructure: versioned directories (`v1-llm-post-training/`), **corrected aggregation bugs in `gen_results.py`** (dense-only rankings; per-algorithm reward axis), papers rebuilt with corrected narrative + restored result tables, README split into two tracks |

# Citation

```bibtex
@software{aitachi2026rlp,
  author = {Aitachi},
  title = {Reinforcement-Learning-Policy: A Two-Track Comparison of Policy-Optimization Algorithms for Robot Control and LLM Post-Training},
  year = {2026},
  url = {https://github.com/aitachi/Reinforcement-Learning-Policy}
}
```

**Author:** Aitachi · **Contact:** 44158892@qq.com · **License:** MIT
