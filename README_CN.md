# Reinforcement-Learning-Policy(强化学习策略优化对比项目)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

**双轨道策略优化算法开源对比:当前 v2 聚焦机器人控制,完整保留的 v1 为大语言模型后训练研究。**

**Language / 语言:** [English](README.md) | [中文](README_CN.md)

---

## 双轨道导航 —— 请先读这里

本仓库包含**两条物理隔离的研究轨道**,共享同一研究问题——*在受控、预算对齐的条件下,信赖域近端家族与组相对家族各自表现如何?结论又如何随奖励信号改变?*

| 轨道 | 状态 | 领域 | 规模 | 入口 |
|---|---|---|---|---|
| **v2 — 机器人控制套件** | **当前版本,持续开发** | 避障导航 / 单足运控 / 抓取提升 / **连续抓取循环** / **双足行走协调** | **16 算法 × 5 任务 × 4 奖励范式 · 393 组 · 两篇论文+正式实验报告** | [`control/`](control/) · [`papers/`](papers/) · [`results/`](results/) |
| **v1 — LLM 后训练研究** | 原版冻结 | Qwen2-1.5B 对齐 | PPO/DPO/GRPO/DAPO + IEEE 综述 | [`v1-llm-post-training/`](v1-llm-post-training/README_CN.md) |

---

# 第一部分:机器人控制套件(v2,当前版本)

纯 PyTorch、设备智能路由(GPU/CPU)的基准:**16 种**策略优化算法 × **5 个**自包含机器人任务 × 4 种可切换奖励范式;统一公平性协议、奖励逐项审计、**加噪鲁棒评估通道**与 **steps-to-80%** 样本效率指标。

## 场景覆盖(三维度对照)

| 机器人能力维度 | 任务 |
|---|---|
| 避障导航 | E1 `nav` |
| 腿足运控(接触密集) | E2 `hopper`(单足) |
| 单臂抓取提升 | E3 `grasp` |
| **机械臂连续抓取动作循环**(接近→抓→举→携→呈递 ×3) | E4 `graspseq` |
| **全身双足步态协调** | E5 `walker` |

## 快速开始

```bash
# 单次运行
python control/train.py --env walker --algo espo --reward dense --seed 0
# 全网格(可断点续跑;off-policy 自动路由 GPU)
python control/sweep.py --axis all --workers 2
# 设备分轨并行(GPU:off-policy 三剑客 / CPU:同策略家族)
python control/sweep.py --env walker --axis algo --algos sac,td3,sac_tr --workers 3
python control/sweep.py --axis sense --workers 2              # PPO ε 敏感性
# 出版级图表 + 核心数字
python control/analysis.py && python control/gen_results.py
# 自检(5 环境 × 16 算法)
python control/selftest_envs.py && python control/selftest_algos.py
```

不依赖 MuJoCo / 外部物理引擎。

## 任务环境(E1–E5)

| 任务 | 机制 | 观测/动作 | 回合 | 成功判据 | 预算 |
|---|---|---|---|---|---|
| **E1 `nav`** | 全向机器人穿越随机障碍场(8 线激光雷达) | 14 / 2 | 150 | 到达目标(≤0.06) | 400k |
| **E2 `hopper`** | 5-DOF 平面单足,串联弹性膝(k=18),解析雅可比 | 14 / 2 | 250 | 位移 3 m 不摔 | 600k |
| **E3 `grasp`** | 三连杆速度臂,滞回抓取 | 16 / 4 | 160 | 持举 15 cm | 500k |
| **E4 `graspseq`** | 同臂;**3 件物体顺序循环**:接近→抓→举→携带入区→**稳定呈递**(持稳 5 步)→换件;臂全程不归位 | 19 / 4 | 300 | 3 轮全成(outcome=完成比) | 800k |
| **E5 `walker`** | 7-DOF 平面**双足**(躯干+双腿,4 驱动),逐足接触,弱膝弹簧——须自学双腿步态协调 | 23 / 4 | 250 | 位移 3 m 不摔 | 600k |

## 奖励范式(`control/rewards/paradigms.py`)

每任务配 **sparse / dense / PBS / curriculum**(共 **20 个带审计奖励函数**)。**奖励轴是一等公民实验**:3 代表算法 × 4 范式 × 5 任务 × 3 种子——稀疏奖励行为与 dense 主表在同一网格测得;逐项遥测全程开启(奖励黑客审计)。

## 算法(`control/algos/`,16 + 2 敏感变体)

| 家族 | 算法 |
|---|---|
| PPO 近端(方法开关) | **PPO-Clip、PPO-KL、PPO-RB、TR-PPO-RB、ESPO** |
| 阶段化/经典同策略 | **PPG、TRPO、A2C** |
| 组相对(源自 LLM RL) | **GRPO-C、DAPO-C** |
| 异策略锚点 | **SAC、TD3、SAC-TR**(KL 信赖域 actor——与 PPO-KL"同约束、异采样"对照) |
| 无梯度 | **ES**(镜像采样 OpenAI-ES,同 2×128 网络) |
| PPO 消融/敏感性 | **PPO-Ent**(熵)、**PPO-NoVC**(去价值裁剪)、PPO-Clip ε∈{0.1,0.3}(sense 轴) |

## 核心结果(dense,3 种子,393 组)

### E1–E3(16 算法)

| 算法 | E1 导航 | E2 单足 | E3 抓取 | 备注 |
|---|---|---|---|---|
| SAC | **0.830** | **0.899** | 0.629 | 慢 5–8× |
| SAC-TR | 0.829±0.03 | 0.898 | 0.552 | E1–E4≈SAC,E5 登顶 |
| PPG | 0.673 | 0.020 | 0.998 | E1 仅 39 s |
| PPO-Ent | 0.666±0.02 | 0.781±0.13 | 0.990 | 熵奖励≈中性 |
| TRPO | 0.669±0.02 | 0.681 | **1.000±0.00** | |
| PPO-KL | 0.658 | 0.785 | 0.992 | |
| PPO-RB | 0.655 | 0.669 | 0.995 | |
| PPO-Clip | 0.652 | 0.620 | 0.996 | |
| PPO-NoVC | 0.646±0.03 | 0.543±0.25 | 0.993 | 价值裁剪关键 |
| ESPO | 0.648 | 0.755 | 0.568 | |
| TR-PPO-RB | 0.625 | 0.754 | 0.694 | |
| GRPO-C | 0.557±0.07 | 0.017 | 0.000 | 仅 outcome 型 |
| A2C | 0.547 | 0.027 | 0.059 | |
| TD3 | 0.434 | 0.167 | 0.000 | 种子彩票 |
| ES | 0.296±0.00 | −0.024 | 0.000 | 采样平价下界 |
| DAPO-C | 0.324 | 0.015 | 0.000 | |

### E4 连续抓取循环 / E5 双足行走(新增)

| 算法 | E4 完成比 | E5 位移/10 | | 算法 | E4 | E5 |
|---|---|---|---|---|---|---|
| PPO-RB | **0.999±0.00** | 0.657±0.17 | | SAC | 0.792±0.09 | 0.559±0.04 |
| TRPO | **0.999±0.00** | 0.764±0.02 | | SAC-TR | 0.596±0.07 | **0.859±0.03** |
| PPO-NoVC | 0.997 | 0.659±0.09 | | ESPO | 0.837±0.12 | **0.838±0.06** |
| PPO-Clip | 0.994 | 0.461±0.13 | | PPG | 0.959 | 0.182±0.17 |
| PPO-KL | 0.988 | **0.753±0.08** | | TD3 | 0.195±0.28 | 0.136 |
| PPO-Ent | 0.985 | 0.528 | | A2C | 0.000 | 0.032 |
| TR-PPO-RB | 0.694±0.39 | 0.722±0.19 | | GRPO-C/DAPO-C | 0.000 | ~0.03 |
| | | | | ES | 0.000 | −0.024 |

**核心发现(v2.2)**

1. **信赖域保险决定可靠性,双足以铁证加冕**:E5 上保险变体(ESPO 0.838 / TRPO 0.764 / PPO-KL 0.753 / TR-PPO-RB 0.722)碾压朴素裁剪(0.461)。
2. **连续抓取循环是同策略堡垒**:E4 被 PPO 全家 ≥0.99 解出,而 **GRPO-C/DAPO-C 恰为 0.000**——多阶段循环比单次抓取更彻底地击溃回合结果基线。
3. **SAC-TR 在 E5 双足以 0.859±0.03 登顶**(SAC 本体 0.559),但在 E1–E4 与 SAC 持平或略逊——KL 信赖域对 off-policy 的收益依赖任务结构:双足长时程平衡回报约束,单目标任务由回放分布漂移主导。
4. **消融定位关键技巧**:熵≈中性;**去价值裁剪 E2 掉 8 分且散布翻倍**(0.543±0.25);ε 敏感性温和(E2:0.51/0.62/0.65)。
5. **组相对仍是 outcome 型专家**:E5 上 GRPO-C sparse(0.114)>dense(0.027)再现"干净终端信用"规律。
6. **ES 仅学会导航(0.296)**:严格采样平价下仅 ~450 代——无梯度搜索的采样代价被量出。
7. **奖励轴**:E4 循环**必需 dense**(sparse 0.01/PBS 0.03 vs dense 0.99);PBS 在 E2/E5 运控塌方、在 E1/E3 优于稀疏。
8. **鲁棒性通道**(动作σ0.1+观测σ0.05):策略保留 94–100% 能力(E4 PPO 0.999→0.999;E5 ESPO 0.838→0.833)。
9. **steps-to-80%** 全格报告(E5 ESPO 180k、E4 PPO 410k)——机器人 RL 核心样本效率指标。

## 文档三件套

| 文档 | 内容 |
|---|---|
| [`papers/experiment_report/`](papers/experiment_report/main.pdf) | **正式实验报告(中文)**——与本 README 同步:全部表格、双流程图、知识图谱、两幅 3D 对比图 |
| [`papers/paper1_algo_comparison/`](papers/paper1_algo_comparison/main.pdf) | 算法对比论文(v2.1 冻结研究:12 算法×3 任务×189 组) |
| [`papers/paper2_reward_design/`](papers/paper2_reward_design/main.pdf) | 奖励范式设计论文(12 审计奖励函数+黑客案例) |

## 结果目录

```
results/
  <env>/<algo>_<reward>_s<seed>.csv   # 学习曲线+加噪列(393 组)
  <env>/<algo>_<reward>_s<seed>.json  # 汇总+奖励审计
  figs/                               # 出版图(PDF)+ .tex 表
  headline.json / findings_draft.md   # 排名/奖励轴/st80/加噪
```

## 延后算法路线图(v2.2 范围外,理由明示)

| 家族 | 延后理由 |
|---|---|
| ACER / RE3-JR(连续动作) | Retrace+信赖域校正属论文级工程,仓促实现=不公平基线 |
| DreamerV3 / TD-MPC2 | 世界模型管线独立成研究,入场成本≫增量洞见 |
| IQL / CQL / TD3+BC | 需离线数据集,本套件为在线协议 |
| DAgger / BC / RMA | 需示教数据/真机部署回路 |
| CMA-ES | 13k 参数下全协方差不可行;以 OpenAI-ES 替代并实测其采样代价 |

---

# 第二部分:LLM 后训练研究(v1,原版保留)

原奠基研究:Qwen2-1.5B 上 **PPO/DPO/GRPO/DAPO** 完整实现+IEEE 综述+权重+AIME 消融。全部内容在 [`v1-llm-post-training/`](v1-llm-post-training/README_CN.md) 且目录内自洽运行。

## 仓库结构

```
Reinforcement-Learning-Policy/
├── README.md / README_CN.md        # 本文件(v2 优先)
├── PROJECT_STRUCTURE.md            # 双轨道目录地图
├── control/                        # 【v2】套件(envs×5/范式×20/algos×16/sweep 分轨)
├── papers/                         # 【v2】两论文+正式报告
├── results/                        # 【v2】393 组+图表+headline
└── v1-llm-post-training/           # 【v1】LLM 研究(冻结)
```

## 版本历史

| 版本 | 日期 | 内容 |
|---|---|---|
| v1(3.0.0) | 2025-04 | LLM 四算法研究首发 |
| v2 | 2026-11 | 机器人套件:12 算法×3 任务×4 范式,189 组,两论文 |
| v2.1 | 2026-11-14 | 双轨道重组;聚合 bug 修复;论文重建;正式报告 |
| **v2.2** | **2026-11-15** | **+E4 连续抓取、+E5 双足行走;+SAC-TR/消融/ES(16 算法);ε 敏感轴;加噪鲁棒协议;steps-to-80%;新增 204 组→393;GPU 路由+双轨 sweep** |

## 引用

```bibtex
@software{aitachi2026rlp,
  author = {Aitachi},
  title = {Reinforcement-Learning-Policy: A Two-Track Comparison of Policy-Optimization Algorithms for Robot Control and LLM Post-Training},
  year = {2026},
  url = {https://github.com/aitachi/Reinforcement-Learning-Policy}
}
```

**作者:** Aitachi · **联系方式:** 44158892@qq.com · **许可证:** MIT
