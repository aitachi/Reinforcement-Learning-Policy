# Reinforcement-Learning-Policy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

**An open, two-track comparison of policy-optimization algorithms: the current
v2 suite targets robot control; the preserved v1 track covers LLM
post-training.**

强化学习策略优化对比项目:当前 **v2** 聚焦**机器人控制**,完整保留的 **v1** 为大模型后训练研究。

**Language / 语言:** [English](README.md) | [中文 学术全量版](README_CN.md)

> **[中文学术全量版 README_CN.md](README_CN.md)** contains the complete experiment report inline:
> all five per-environment data tables (final±std / AUC / steps-to-80% / wall-clock / noisy),
> the reward-axis table, sensitivity & ablations, robustness analysis, and every figure
> (learning curves, bars, radar, two 3-D plots, workflow, knowledge graph) embedded.

---

## Track map — read this first / 双轨道导航

This repository contains **two clearly separated research tracks**. They share
one question — *how do the trust-region proximal family and the group-relative
family behave under controlled, budget-matched conditions, and how does the
answer depend on the reward signal?* — but live in different domains,
directories and documents.

| Track | Status | Domain | Scope | Entry |
|---|---|---|---|---|
| **v2 — Robot Control Suite** | **current, actively developed** | robot control: obstacle navigation, monopod locomotion, reach–grasp–lift, **continuous grasp cycles, biped walking** | **16 algorithms × 5 tasks × 4 reward paradigms · 393 runs · 2 papers + formal report** | [`control/`](control/) · [`papers/`](papers/) · [`results/`](results/) |
| **v1 — LLM Post-Training Study** | original release, complete & frozen | LLM alignment on Qwen2-1.5B | PPO / DPO / GRPO / DAPO implementations + IEEE survey (EN/CN/JA) | [`v1-llm-post-training/`](v1-llm-post-training/README.md) |

- **[Part I (v2)](#part-i--robot-control-suite-v2-current)** — the robot-control comparison; all new work happens here.
- **[Part II (v1)](#part-ii--llm-post-training-study-v1-preserved)** — summary of the original LLM study; full treatise in [`v1-llm-post-training/README.md`](v1-llm-post-training/README.md).

---

# Part I · Robot Control Suite (v2, current)

A pure-PyTorch, GPU/CPU-routed benchmark measuring **sixteen**
policy-optimization algorithms on **five** self-contained robot tasks under
four swappable reward paradigms, with a fairness protocol (identical budgets,
networks, optimizers, seeds), per-term reward auditing, a **noisy-evaluation
robustness channel**, and **steps-to-80%** sample-efficiency reporting.

## Scenario coverage / 场景覆盖

| Robot capability axis | Task |
|---|---|
| Mobile navigation under obstacles | E1 `nav` |
| Legged locomotion (contact-rich) | E2 `hopper` (monopod) |
| Single-arm grasp & lift | E3 `grasp` |
| **Continuous sequential grasping cycles** (reach→grasp→lift→carry→present ×3) | E4 `graspseq` |
| **Whole-body biped gait coordination** | E5 `walker` |

## Quick start

```bash
# single run
python control/train.py --env walker --algo espo --reward dense --seed 0
# full grid (resumable; device auto-routes off-policy to GPU)
python control/sweep.py --axis all --workers 2
# device-split parallel tracks (GPU: off-policy trio / CPU: on-policy)
python control/sweep.py --env walker --axis algo --algos sac,td3,sac_tr --workers 3
python control/sweep.py --axis sense --workers 2              # PPO eps sensitivity
# publication figures + LaTeX tables; headline numbers + findings
python control/analysis.py && python control/gen_results.py
# self-tests (envs physics, all 16 algorithms)
python control/selftest_envs.py && python control/selftest_algos.py
```

No MuJoCo / external physics dependency — everything runs on PyTorch tensors.

## Environments (E1–E5)

| Task | Mechanism | Obs/Act | Episode | Success criterion | Budget |
|---|---|---|---|---|---|
| **E1 `nav`** | holonomic robot crossing a random obstacle field (8-ray lidar) | 14 / 2 | 150 | reach goal (≤0.06); collision terminates | 400k |
| **E2 `hopper`** | 5-DOF planar monopod, series-elastic knee (k=18), analytic rigid-body Jacobians | 14 / 2 | 250 | displace 3 m upright | 600k |
| **E3 `grasp`** | velocity-controlled 3-link arm, kinematic grasp with hysteresis | 16 / 4 | 160 | hold object 15 cm up | 500k |
| **E4 `graspseq`** | same arm; **3 sequential objects**, each reach→grasp→lift→carry→**present** in the target zone (held steady 5 steps); arm never resets between objects | 19 / 4 | 300 | all 3 cycles; outcome = cycles/3 | 800k |
| **E5 `walker`** | 7-DOF planar **biped** (torso + 2 legs, 4 actuated joints), per-foot contact, weak knee springs — inter-leg gait coordination must be learned | 23 / 4 | 250 | displace 3 m upright | 600k |

## Reward paradigms (`control/rewards/paradigms.py`)

Every task ships **sparse / dense / potential-based (PBS) / curriculum**
variants — **20 audited reward functions** in total. The **reward axis is a
first-class experiment** (not an afterthought): 3 representative algorithms
× 4 paradigms × 5 tasks × 3 seeds, so sparse-reward behaviour is measured on
the same grid as dense headlines. Per-term telemetry runs throughout
(reward-hacking audits; see the documented flicker-farming case study).

## Algorithms (`control/algos/`, 16 + 2 sensitivity variants)

| Family | Algorithms |
|---|---|
| PPO proximal (method switch) | **PPO-Clip, PPO-KL** (adaptive), **PPO-RB** (rollback), **TR-PPO-RB** (trust-region rollback), **ESPO** (early-stop) |
| Phasic / classical on-policy | **PPG, TRPO** (CG + backtracking), **A2C** |
| Group-relative (from LLM RL) | **GRPO-C, DAPO-C** (episode-outcome baselines, clip-higher, dynamic sampling, timeout filtering) |
| Off-policy anchors | **SAC, TD3**, **SAC-TR** (SAC + KL trust-region actor — the "same constraint as PPO-KL, different sampling" counterpart) |
| Gradient-free | **ES** (antithetic OpenAI-ES on the same 2×128 policy) |
| PPO ablations / sensitivity | **PPO-Ent** (entropy bonus), **PPO-NoVC** (no value clipping), PPO-Clip ε∈{0.1, 0.3} (`sense` axis) |

## Headline results (dense reward, 3 seeds, 393 runs)

### E1–E3 (original tasks, now 16 algorithms)

| Algorithm | E1 Nav | E2 Hopper | E3 Grasp | Note |
|---|---|---|---|---|
| SAC | **0.830** | **0.899** | 0.629 | 5–8× wall cost |
| SAC-TR | 0.829±0.03 | 0.898 | 0.552 | TR ≈ SAC off-policy |
| PPG | 0.673 | 0.020 | 0.998 | 39 s on E1 |
| PPO-Ent | 0.666±0.02 | 0.781±0.13 | 0.990 | entropy ≈ neutral |
| TRPO | 0.669±0.02 | 0.681 | **1.000±0.00** | CG+line-search cost |
| PPO-KL | 0.658 | 0.785 | 0.992 | |
| PPO-RB | 0.655 | 0.669 | 0.995 | |
| PPO-Clip | 0.652 | 0.620 | 0.996 | ±0.17 on E2 |
| PPO-NoVC | 0.646±0.03 | 0.543±0.25 | 0.993 | value-clip matters |
| ESPO | 0.648 | 0.755 | 0.568 | |
| TR-PPO-RB | 0.625 | 0.754 | 0.694 | |
| GRPO-C | 0.557±0.07 | 0.017 | 0.000 | outcome-shaped only |
| A2C | 0.547 | 0.027 | 0.059 | |
| TD3 | 0.434 | 0.167 | 0.000 | seed lottery |
| ES | 0.296±0.00 | −0.024 | 0.000 | sample-parity bound |
| DAPO-C | 0.324 | 0.015 | 0.000 | |

### E4 continuous grasp cycles / E5 biped walking (new)

| Algorithm | E4 cycles | E5 walk (m/10) | | Algorithm | E4 | E5 |
|---|---|---|---|---|---|---|
| PPO-RB | **0.999±0.00** | 0.657±0.17 | | SAC | 0.792±0.09 | 0.559±0.04 |
| TRPO | **0.999±0.00** | 0.764±0.02 | | SAC-TR | 0.596±0.07 | **0.859±0.03** |
| PPO-NoVC | 0.997 | 0.659±0.09 | | ESPO | 0.837±0.12 | **0.838±0.06** |
| PPO-Clip | 0.994 | 0.461±0.13 | | PPG | 0.959 | 0.182±0.17 |
| PPO-KL | 0.988 | **0.753±0.08** | | TD3 | 0.195±0.28 | 0.136 |
| PPO-Ent | 0.985 | 0.528 | | A2C | 0.000 | 0.032 |
| TR-PPO-RB | 0.694±0.39 | 0.722±0.19 | | GRPO-C / DAPO-C | 0.000 | ~0.03 |
| | | | | ES | 0.000 | −0.024 |

**Key findings (v2.2)**

1. **Trust-region safeguards decide reliability** — sharpened on the biped:
   on E5 every constrained variant (SAC-TR 0.859, ESPO 0.838, TRPO 0.764,
   PPO-KL 0.753, TR-PPO-RB 0.722) crushes plain clipping (0.461); on E2
   spread ≤±0.12 vs ±0.17.
2. **Continuous manipulation cycles are an on-policy fortress**: E4 solved
   near-perfectly by the whole PPO family (≥0.99) while **GRPO-C/DAPO-C
   score exactly 0.000** — multi-stage cycles defeat episode-outcome
   baselines even more decisively than single grasps.
3. **SAC-TR crowns the biped (0.859±0.03 on E5, vs SAC 0.559)** but is
   parity-or-worse vs SAC on E1–E4 (0.829/0.898/0.552/0.596) — the KL
   trust region pays off off-policy only on long-horizon balance tasks;
   elsewhere replay-induced distribution shift dominates update size.
4. **PPO ablations localize the important tricks**: entropy bonus ≈ neutral;
   **removing value clipping costs 8 points and doubles seed scatter on E2**
   (0.543±0.25); clip-ε sensitivity is mild (E2: 0.51/0.62/0.65 for
   ε=0.1/0.2/0.3; E3: 0.95/1.00/0.99).
5. **Group-relative methods remain outcome-shaped specialists**; on E5 GRPO-C
   again prefers sparse (0.114) over dense (0.027) — clean terminal credit.
6. **ES (gradient-free) learns navigation (0.296) and nothing else** under
   strict sample parity (~450 generations) — the measured sample price of
   black-box search; its regime is wall-clock parallelism, not samples.
7. **Reward axis (measured on the same grid)**: dense is *required* for E4
   cycles (sparse 0.01, PBS 0.03, curriculum 0.66 vs dense 0.99); PBS
   collapses on E2/E5 locomotion (γ<1 potential distortion) but helps E1/E3
   over sparse.
8. **Robustness channel** (v2.2 protocol: action σ0.1 + obs σ0.05 at eval):
   trained policies retain ~94–100% (E4 PPO 0.999→0.999; E5 ESPO
   0.838→0.833; nav SAC-TR 0.829→0.775) — no brittleness cliff.
9. **Steps-to-80%** now reported per cell (e.g. E5 ESPO 180k, E4 PPO 410k)
   alongside AUC — the robot-RL sample-efficiency metric.

## Formal report & papers

| Document | Contents |
|---|---|
| [`papers/experiment_report/`](papers/experiment_report/main.pdf) | **Formal experiment report (Chinese, 正式实验报告)** — synced with README: all tables, training-loop & pipeline flowcharts, algorithm knowledge graph, two 3-D comparison figures |
| [`papers/paper1_algo_comparison/`](papers/paper1_algo_comparison/main.pdf) | *From Trust Regions to Group Baselines* — the frozen v2.1 study (12 algos × 3 tasks × 189 runs) |
| [`papers/paper2_reward_design/`](papers/paper2_reward_design/main.pdf) | *Reward Paradigm Design for Robot Policy Optimization* — 12 audited reward functions + reward-hacking case study |

## Results layout

```
results/
  <env>/<algo>_<reward>_s<seed>.csv   # learning curves + noisy-eval columns (393 runs)
  <env>/<algo>_<reward>_s<seed>.json  # final summary + reward-term audit
  figs/                               # publication figures (PDF) + .tex tables
  headline.json / findings_draft.md   # dense rankings, reward axis, st80, noisy
```

## Deferred algorithm roadmap (out of v2.2 scope, by design)

| Family | Why deferred |
|---|---|
| ACER / RE3-JR (continuous-action) | trust-region-corrected off-policy with Retrace is a paper-grade implementation; a rushed version is an unfair baseline |
| DreamerV3 / TD-MPC2 (model-based) | world-model pipelines are a separate study; entry cost ≫ incremental insight here |
| IQL / CQL / TD3+BC (offline) | require offline datasets this online suite does not provide |
| DAgger / BC / RMA (imitation, adaptation) | need expert demonstrations / real-robot deployment loop |
| CMA-ES | full covariance is infeasible at the protocol's ~13k parameters; OpenAI-ES used instead (with its sample-parity caveat measured) |

---

# Part II · LLM Post-Training Study (v1, preserved)

The original study that founded this repository: complete implementations of
**PPO, DPO, GRPO, DAPO** for LLM alignment on Qwen2-1.5B, with an IEEE-style
survey (EN/CN), algorithm derivations, trained checkpoints and an AIME-2024
ablation (dynamic sampling +8 pts, overlong filtering +6 pts).

**Everything of v1 lives in [`v1-llm-post-training/`](v1-llm-post-training/README.md) and runs from that directory.**

---

# Repository layout

```
Reinforcement-Learning-Policy/
├── README.md / README_CN.md        # this file — v2-first, two-track navigation
├── PROJECT_STRUCTURE.md            # annotated directory map (both tracks)
├── LICENSE / requirements.txt / .gitignore
├── control/                        # [v2] robot-control suite
│   ├── envs/                       #   nav / hopper / grasp / graspseq(E4) / walker(E5)
│   ├── rewards/paradigms.py        #   4 paradigms × 5 tasks = 20 audited reward fns
│   ├── algos/                      #   ppo_family / trpo / grpo_dapo / sac(+SAC-TR) / td3 / es / nets
│   ├── train.py                    #   unified entry (device auto-routing + noisy eval)
│   ├── sweep.py                    #   resumable grid; --algos device-split tracks; sense axis
│   ├── analysis.py / gen_results.py / configs.py
│   └── selftest_envs.py / selftest_algos.py
├── papers/                         # [v2] 2 papers + formal experiment report
├── results/                        # [v2] 393 runs + figures + headline numbers
└── v1-llm-post-training/           # [v1] LLM post-training study (frozen)
```

# Version history

| Version | Date | Contents |
|---|---|---|
| v1 (3.0.0) | 2025-04 | Initial release: PPO/DPO/GRPO/DAPO LLM study |
| v2 | 2026-11 | Robot-control suite: 12 algos × 3 tasks × 4 paradigms, 189 runs, two papers |
| v2.1 | 2026-11-14 | Two-track restructure; aggregation bugs fixed; papers rebuilt; formal report added |
| **v2.2** | **2026-11-15** | **+E4 continuous grasp cycles, +E5 biped walker; +SAC-TR/PPO-Ent/PPO-NoVC/ES (16 algos); ε-sensitivity axis; noisy-eval robustness protocol; steps-to-80%; 204 new runs → 393 total; GPU off-policy routing + dual-track sweeps** |

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
