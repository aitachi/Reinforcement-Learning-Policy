# Reinforcement-Learning-Policy(强化学习策略优化对比项目)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

**双轨道策略优化算法开源对比:当前 v2 聚焦机器人控制,完整保留的 v1 为大语言模型后训练研究。**

**Language / 语言:** [English](README.md) | [中文](README_CN.md)

---

## 双轨道导航 —— 请先读这里

本仓库包含**两条清晰分离的研究轨道**,共享同一个研究问题——*在受控、预算对齐的条件下,信赖域近端家族与组相对家族各自表现如何?结论又如何随奖励信号改变?*——但分属不同领域、不同目录、各自独立成文。

| 轨道 | 状态 | 领域 | 规模 | 入口 |
|---|---|---|---|---|
| **v2 — 机器人控制套件** | **当前版本,持续开发** | 机器人控制:避障导航 / 单足运控 / 抓取-提起 | **12 算法 × 3 任务 × 4 奖励范式 · 189 组实验 · 2 篇论文** | [`control/`](control/) · [`papers/`](papers/) · [`results/`](results/) |
| **v1 — LLM 后训练研究** | 原始版本,完整冻结保留 | 基于 Qwen2-1.5B 的大模型对齐 | PPO / DPO / GRPO / DAPO 完整实现 + IEEE 综述(英/中/日) | [`v1-llm-post-training/`](v1-llm-post-training/README_CN.md) |

- 新工作都在 **[第一部分(v2)](#第一部分机器人控制套件v2当前版本)** —— 机器人对比实验。
- **[第二部分(v1)](#第二部分llm-后训练研究v1原版保留)** 是原 LLM 研究的摘要;其 600 行完整文档现位于 [`v1-llm-post-training/README_CN.md`](v1-llm-post-training/README_CN.md)。

---

# 第一部分:机器人控制套件(v2,当前版本)

纯 PyTorch、GPU 批量化的基准测试:**12 种**策略优化算法 × 3 个自包含机器人任务 × 4 种可切换奖励范式,统一公平性协议(相同采样预算、网络规模、优化器、种子),奖励逐项审计。

## 快速开始

```bash
# 单次运行
python control/train.py --env nav --algo ppo --reward dense --seed 0
# 全网格(可断点续跑)
python control/sweep.py --axis all --workers 2
# 出版级图表 + LaTeX 表格
python control/analysis.py
# 核心结论数字 + 发现草稿
python control/gen_results.py
# 环境与算法自检
python control/selftest_envs.py && python control/selftest_algos.py
```

不依赖 MuJoCo / 外部物理引擎,全部在 PyTorch 张量上运行。

## 任务环境(E1–E3)

| 任务 | 机制 | 观测/动作 | 步长 | 成功判据 | 预算 |
|---|---|---|---|---|---|
| **E1 `nav`** | 全向机器人穿越随机障碍场(8 线激光雷达观测) | 14 / 2 | 150 | 到达目标(≤0.06),碰撞即终止 | 400k 步 |
| **E2 `hopper`** | 5-DOF 平面单足,串联弹性被动膝关节(k=18),解析刚体雅可比装配 | 14 / 2 | 250 | 位移 3 m 不摔倒 | 600k 步 |
| **E3 `grasp`** | 速度控制三连杆机械臂,带滞回的动力学抓取,提起即成功 | 16 / 4 | 160 | 将物体持举 15 cm | 500k 步 |

## 奖励范式(`control/rewards/paradigms.py`)

每个任务配 **sparse / dense / PBS(势函数)/ curriculum** 四种范式(共 12 个带审计的奖励函数)。任务信号与奖励解耦;每个奖励项训练时全程记录(奖励黑客遥测——见论文2 中抓取边界抖动刷奖案例)。

## 算法(`control/algos/`,12 种)

| 家族 | 算法 |
|---|---|
| PPO 近端家族(方法开关) | **PPO-Clip、PPO-KL**(自适应)、**PPO-RB**(回滚)、**TR-PPO-RB**(信赖域回滚)、**ESPO**(早停) |
| 阶段化 / 经典同策略 | **PPG、TRPO**(CG+回溯线搜)、**A2C** |
| 组相对(源自 LLM RL) | **GRPO-C、DAPO-C**(回合结果基线、clip-higher、动态采样、超时过滤) |
| 异策略锚点 | **SAC、TD3** |

## 核心结果(dense 奖励,3 种子均值,共 189 组)

| 算法 | E1 导航成功率 | E2 单足位移 | E3 抓取提起 | 备注 |
|---|---|---|---|---|
| SAC | **0.830** | **0.899** | 0.629 | 慢 5–8×(E2 4403 s) |
| PPG | 0.673 | 0.020 | 0.998 | E1 仅 39 s |
| TRPO | 0.669±0.02 | 0.681 | **1.000±0.00** | CG+线搜开销 |
| PPO-KL | 0.658 | 0.785 | 0.992 | |
| PPO-RB | 0.655 | 0.669 | 0.995 | |
| PPO-Clip | 0.652 | 0.620 | 0.996 | E2 种子散布 ±0.17 |
| ESPO | 0.648 | 0.755 | 0.568 | |
| TR-PPO-RB | 0.625 | 0.754 | 0.694 | |
| GRPO-C | 0.557±0.07 | 0.017 | 0.000 | 仅 E1 中游 |
| A2C | 0.547 | 0.027 | 0.059 | |
| TD3 | 0.434 | 0.167 | 0.000 | 种子彩票 |
| DAPO-C | 0.324 | 0.015 | 0.000 | |

**核心发现**

1. **信赖域保险决定可靠性。** 接触密集的 E2 上,方差受控变体(自适应 KL / 早停 / 信赖域回滚)种子散布 ≤ ±0.12,朴素裁剪散到 ±0.17。
2. **TRPO 沉默地优秀。** dense 奖励下以种子满分 1.000 拿下 E3,E1 位列第三——早前"高方差"的印象是聚合时混入多奖励范式造成的假象(已在 `control/gen_results.py` 修复,上表为纯 dense 数值)。
3. **SAC 用 5–8× 墙钟换 E1+E2 双冠;TD3 是种子彩票**(E1 0.43±0.20,E3 为 0)。
4. **组相对方法(GRPO-C/DAPO-C)只适合 outcome 型任务。** 在最接近 outcome 型的 E1 中游,在运控与多阶段抓取上趋近于零;值得注意的是 E2 上 GRPO-C 用 *sparse*(0.109)反超自己的 dense(0.017)——干净的终端信用更适合组基线。
5. **奖励范式 × 算法交互真实存在:** PBS 在 E1/E3 拯救稀疏训练、在 E2 崩塌(γ<1 势函数失真);dense 多项回报反而会*污染*组基线。

完整分析:下面两篇论文 + `results/findings_draft.md`。

## 论文(由本套件实验产出)

| 论文 | 标题 | 内容 |
|---|---|---|
| [`papers/paper1_algo_comparison/`](papers/paper1_algo_comparison/main.pdf) | *From Trust Regions to Group Baselines: A Unified Empirical Study of Proximal Policy Optimization Families for Robot Control* | 12 算法 × 3 任务 × 3 种子 + 奖励轴,IEEE 6 页,含 PDF |
| [`papers/paper2_reward_design/`](papers/paper2_reward_design/main.pdf) | *Reward Paradigm Design for Robot Policy Optimization* | 12 个审计奖励函数、范式 × 算法交互、奖励黑客案例研究,IEEE 4 页,含 PDF |

`papers\build_papers.bat` 可重建两篇(MiKTeX 零警告配方)。

## 结果目录

```
results/
  <env>/<algo>_<reward>_s<seed>.csv   # 每次评估的学习曲线(189 组)
  <env>/<algo>_<reward>_s<seed>.json  # 最终汇总 + 奖励项审计
  figs/                               # 出版图(PDF)+ .tex 表格
  headline.json / findings_draft.md   # 提炼后的核心数字
```

---

# 第二部分:LLM 后训练研究(v1,原版保留)

本仓库的奠基研究:基于 Qwen2-1.5B 的 **PPO / DPO / GRPO / DAPO** 完整实现,IEEE 风格综述(英/中)、算法推导、训练权重,以及 AIME-2024 消融(动态采样 +8 分、过长过滤 +6 分)。

**v1 的全部内容都在 [`v1-llm-post-training/`](v1-llm-post-training/README_CN.md) 中,并在该目录内运行**——完整 README(含全部算法长文)、日文版、`algorithms/`、`docs/`(IEEE LaTeX 源 + PDF)、实验输出与权重:

```bash
cd v1-llm-post-training
pip install -r requirements.txt
python run_comparison.py        # 四算法对比
```

---

# 仓库整体结构

```
Reinforcement-Learning-Policy/
│
├── README.md / README_CN.md        # 本文件 —— v2 优先、双轨道导航
├── PROJECT_STRUCTURE.md            # 带注释的目录地图(双轨道)
├── LICENSE / requirements.txt / .gitignore
│
├── control/                        # 【v2】机器人控制套件
│   ├── envs/                       #   nav_obstacle / hopper_lite / reach_grasp
│   ├── rewards/paradigms.py        #   sparse / dense / pbs / curriculum × 3 任务
│   ├── algos/                      #   ppo_family / trpo / grpo_dapo / sac / td3 / nets
│   ├── train.py                    #   统一入口
│   ├── sweep.py                    #   全网格调度(可断点续跑)
│   ├── analysis.py                 #   出版级图表 + LaTeX 表格
│   ├── gen_results.py              #   headline.json + findings_draft.md
│   ├── configs.py                  #   公平性协议超参
│   └── selftest_envs.py / selftest_algos.py
│
├── papers/                         # 【v2】两篇论文(IEEEtran,含 PDF)
│   ├── paper1_algo_comparison/     #   算法对比
│   ├── paper2_reward_design/       #   奖励范式设计
│   └── build_papers.bat
│
├── results/                        # 【v2】189 组实验 + 图表 + 提炼数字
│
└── v1-llm-post-training/           # 【v1】LLM 后训练研究(冻结)
    ├── README.md / README_CN.md / README_JA.md   # 完整长文(英/中/日)
    ├── algorithms/                 #   ppo / dpo / grpo / dapo 训练器
    ├── docs/                       #   IEEE 综述(PDF + LaTeX)、算法深剖、图
    ├── data / models / checkpoints / scripts / src
    ├── DAPO_GRPO_PPO_DPO/          #   四算法实验输出 + 权重
    ├── GRPO_PPO_DPO/               #   更早的三算法输出
    ├── run_comparison.py / requirements.txt
    └── DAPO.pdf / DeepSeek_R1.pdf / 分析报告      # 参考论文与笔记
```

# 版本历史

| 版本 | 日期 | 内容 |
|---|---|---|
| v1(3.0.0) | 2025-04 | 首次公开:PPO/DPO/GRPO/DAPO LLM 研究(初名 `deepseek_r1_qwen2-1.5b`,后名 `PPOvDPOvGRPOvDAPO`) |
| v2 | 2026-11 | 机器人控制套件:12 算法 × 3 任务 × 4 奖励范式,189 组实验,两篇论文 |
| v2.1 | 2026-11-14 | 本次重组:目录版本化(`v1-llm-post-training/`);**修复 `gen_results.py` 聚合双 bug**(dense-only 排名、按算法奖励轴);论文以修正叙述重建、三张结果表恢复为正规浮动体;README 拆分为双轨道 |

# 引用

```bibtex
@software{aitachi2026rlp,
  author = {Aitachi},
  title = {Reinforcement-Learning-Policy: A Two-Track Comparison of Policy-Optimization Algorithms for Robot Control and LLM Post-Training},
  year = {2026},
  url = {https://github.com/aitachi/Reinforcement-Learning-Policy}
}
```

**作者:** Aitachi · **联系方式:** 44158892@qq.com · **许可证:** MIT
