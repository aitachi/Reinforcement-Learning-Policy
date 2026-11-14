# 项目结构总览(双轨道)

仓库 `Reinforcement-Learning-Policy` 包含两条**物理隔离**的研究轨道:
**v2 = 机器人控制套件(根目录,当前版本)**,**v1 = LLM 后训练研究(`v1-llm-post-training/`,原版冻结)**。

```
Reinforcement-Learning-Policy/
│
├── README.md                       # 英文总入口:v2 优先 + 双轨道导航
├── README_CN.md                    # 中文总入口(同上)
├── PROJECT_STRUCTURE.md            # 本文件:带注释的目录地图
├── LICENSE                         # MIT
├── requirements.txt                # v2 依赖(torch/numpy/pandas/matplotlib)
├── .gitignore
│
│   ┌────────────────────────────────────────────────────────────┐
│   │  v2 · 机器人控制套件(当前版本,12 算法 × 3 任务 × 4 范式)   │
│   └────────────────────────────────────────────────────────────┘
│
├── control/                        # 套件源码(纯 PyTorch,无 MuJoCo)
│   ├── envs/
│   │   ├── nav_obstacle.py         #   E1 避障导航(14 观测 / 2 动作,激光雷达)
│   │   ├── hopper_lite.py          #   E2 单足运控(5-DOF 刚体 + 膝关节被动弹簧 k=18)
│   │   └── reach_grasp.py          #   E3 三连杆抓取(滞回抓取,提起即成功)
│   ├── rewards/
│   │   └── paradigms.py            #   4 范式(sparse/dense/PBS/curriculum)× 3 任务 = 12 奖励函数
│   ├── algos/
│   │   ├── ppo_family.py           #   PPO-Clip/KL/RB/TR-RB/ESPO + PPG + A2C(方法开关)
│   │   ├── trpo.py                 #   TRPO(CG + 回溯线搜)
│   │   ├── grpo_dapo.py            #   GRPO-C / DAPO-C(组相对 + clip-higher + 动态采样)
│   │   ├── sac.py                  #   SAC 基线
│   │   ├── td3.py                  #   TD3 基线
│   │   └── nets.py                 #   tanh 高斯策略 + GAE rollout
│   ├── train.py                    #   统一训练入口(train.py --env --algo --reward --seed)
│   ├── sweep.py                    #   全网格调度,可断点续跑
│   ├── catchup.py                  #   断点补偿工具
│   ├── analysis.py                 #   出版级图(PDF)+ LaTeX 表格(figs/)
│   ├── gen_results.py              #   headline.json + findings_draft.md(dense-only 聚合)
│   ├── configs.py                  #   公平性协议统一超参
│   ├── selftest_envs.py            #   环境物理自测
│   └── selftest_algos.py           #   12 算法 × 3 环境冒烟
│
├── results/                        # 实验输出(189 组)
│   ├── nav/ hopper/ grasp/         #   <algo>_<reward>_s<seed>.{csv,json,log}
│   ├── figs/                       #   论文图表(PDF)+ tables_*.tex
│   ├── headline.json               #   核心数字(dense 排名 + 奖励轴 + DAPO 证据)
│   ├── findings_draft.md           #   发现草稿
│   └── summary.csv / sweep_*.log / catchup.log
│
├── papers/                         # 【v2】两篇论文(IEEEtran,PDF 已含)
│   ├── paper1_algo_comparison/     #   论文1:算法对比(12 算法,6 页)
│   ├── paper2_reward_design/       #   论文2:奖励范式设计(4 页)
│   └── build_papers.bat            #   MiKTeX 一键重建(零警告配方)
│
│   ┌────────────────────────────────────────────────────────────┐
│   │  v1 · LLM 后训练研究(原版,冻结;在本目录内运行)           │
│   └────────────────────────────────────────────────────────────┘
│
└── v1-llm-post-training/
    ├── README.md / README_CN.md / README_JA.md   # 完整算法长文(英/中/日,约 600 行/语言)
    ├── requirements.txt            # v1 依赖(transformers/peft/accelerate 等)
    ├── algorithms/                 # 四算法完整实现
    │   ├── ppo_trainer.py          #   PPO(价值网络 + GAE,~517 行)
    │   ├── dpo_trainer.py          #   DPO(偏好优化,~431 行)
    │   ├── grpo_trainer.py         #   GRPO(组优势归一化,~533 行)
    │   └── dapo_trainer.py         #   DAPO(动态采样 + Token 级损失,~738 行)
    ├── docs/                       # 文档与可视化
    │   ├── PPO/DPO/GRPO/DAPO_Algorithm.md        # 四算法深剖
    │   ├── Algorithm_Comparison.md #   全面对比
    │   ├── RL_LLM_Survey_IEEE_EN.pdf / _CN.pdf   # IEEE 综述(英 10 页 / 中 8 页)
    │   ├── ieee_en/ ieee_cn/       #   LaTeX 源
    │   ├── figures/                #   全部可视化图
    │   └── gen_*.py / generate_all_figures.py    # 图表生成脚本
    ├── data/sample_reasoning_data.json           # 10 条示例推理数据
    ├── models/                     # 基础模型占位
    ├── checkpoints/                # 训练检查点(ppo/dpo/grpo/dapo_model)
    ├── scripts/                    # download_model.sh / train.sh
    ├── src/                        # 原有训练代码
    ├── DAPO_GRPO_PPO_DPO/          # 四算法对比实验输出(*.pth 权重 + CSV + PNG)
    ├── GRPO_PPO_DPO/               # 更早的三算法输出
    ├── run_comparison.py           # 四算法对比主脚本(在本目录运行)
    ├── DAPO.pdf / DeepSeek_R1.pdf  # 参考论文原文
    ├── RL_Four_Algorithms_Comparison.md          # 四算法对比分析
    ├── QWEN2-R1脚本分析报告.md
    ├── SFT训练中NaN问题的分析报告.md
    ├── 强化学习算法对比分析.md      # GRPO vs PPO 分析(LLM 域)
    └── 强化学习算法流程研究.html    # 同上 HTML 版
```

## 运行方式速查

| 目标 | 命令(仓库根) |
|---|---|
| v2 单次训练 | `python control/train.py --env nav --algo ppo --reward dense --seed 0` |
| v2 全网格 | `python control/sweep.py --axis all --workers 2` |
| v2 出图/表格 | `python control/analysis.py` |
| v2 核心数字 | `python control/gen_results.py` |
| v2 自检 | `python control/selftest_envs.py && python control/selftest_algos.py` |
| 重建论文 | `papers\build_papers.bat` |
| v1 四算法对比 | `cd v1-llm-post-training && python run_comparison.py` |

## 性能备注

- nav 走 GPU;hopper/grasp 环境步进在 CPU 上更快(内核启动开销主导,GPU 反慢约 6×);
- SAC/TD3 必须用 cuda 设备(CPU 上每任务约 6 小时);`torch.set_num_threads(2)`。

---

**作者:** Aitachi · **邮箱:** 44158892@qq.com · **GitHub:** https://github.com/aitachi/Reinforcement-Learning-Policy

**最后更新:** 2026-11-14(v2.1 重组:目录版本化 + 聚合 bug 修复 + 论文重建)
