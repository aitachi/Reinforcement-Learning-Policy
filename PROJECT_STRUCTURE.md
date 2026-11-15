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
│   │  v2 · 机器人控制套件(当前版本,16 算法 × 5 任务 × 4 范式)   │
│   └────────────────────────────────────────────────────────────┘
│
├── control/                        # 套件源码(纯 PyTorch,无 MuJoCo)
│   ├── envs/
│   │   ├── nav_obstacle.py         #   E1 避障导航(14/2,激光雷达)
│   │   ├── hopper_lite.py          #   E2 单足运控(5-DOF 刚体+膝弹簧 k=18)
│   │   ├── reach_grasp.py          #   E3 三连杆抓取(滞回)
│   │   ├── grasp_seq.py            #   E4 连续抓取循环(3 物体 抓-举-携-呈递)
│   │   └── walker.py               #   E5 双足行走协调(7-DOF 双腿步态)
│   ├── rewards/
│   │   └── paradigms.py            #   4 范式 × 5 任务 = 20 个带审计奖励函数
│   ├── algos/
│   │   ├── ppo_family.py           #   PPO-Clip/KL/RB/TR-RB/ESPO + PPG + A2C
│   │   ├── trpo.py                 #   TRPO(CG+回溯线搜)
│   │   ├── grpo_dapo.py            #   GRPO-C / DAPO-C
│   │   ├── sac.py                  #   SAC + SAC-TR(KL 信赖域 actor)
│   │   ├── td3.py                  #   TD3
│   │   ├── es.py                   #   ES 镜像采样无梯度基线(torch.func.vmap)
│   │   └── nets.py                 #   tanh 高斯策略 + GAE rollout
│   ├── train.py                    #   统一入口(设备自动路由+加噪评估通道)
│   ├── sweep.py                    #   全网格调度(断点续跑/--algos 设备分轨/sense 轴)
│   ├── catchup.py                  #   断点补偿工具
│   ├── analysis.py                 #   出版级图(PDF)+ LaTeX 表格(figs/)
│   ├── gen_results.py              #   headline.json + findings_draft.md(dense-only 聚合)
│   ├── configs.py                  #   公平性协议统一超参
│   ├── selftest_envs.py            #   环境物理自测
│   └── selftest_algos.py           #   16 算法冒烟(临时目录,不碰 results/)
│
├── results/                        # 实验输出(393 组)
│   ├── nav/ hopper/ grasp/         #   <algo>_<reward>_s<seed>.{csv,json,log}
│   ├── figs/                       #   论文图表(PDF)+ tables_*.tex
│   ├── headline.json               #   核心数字(dense 排名 + 奖励轴 + DAPO 证据)
│   ├── findings_draft.md           #   发现草稿
│   └── summary.csv / sweep_*.log / catchup.log
│
├── papers/                         # 【v2】两篇论文 + 正式实验报告(均含 PDF)
│   ├── paper1_algo_comparison/     #   论文1:算法对比(12 算法,6 页)
│   ├── paper2_reward_design/       #   论文2:奖励范式设计(4 页)
│   ├── experiment_report/          #   正式实验报告(中文 11 页:流程图/知识图谱/3D 对比图)
│   │   ├── main.tex                #     XeLaTeX+ctex 单源(xelatex 两遍重建)
│   │   ├── gen_report_figs.py      #     生成两幅 3D 矢量图(figs/)
│   │   └── figs/                   #     fig3d_algo_task / fig3d_paradigm
│   └── build_papers.bat            #   MiKTeX 一键重建论文1/2(零警告配方)
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
| 重建实验报告 | `python papers/experiment_report/gen_report_figs.py` 后于该目录 `xelatex main.tex` 两遍 |
| v1 四算法对比 | `cd v1-llm-post-training && python run_comparison.py` |

## 性能备注(2026-11-15 实测)

- **设备路由已内置**:同策略家族在 hopper/grasp/E4/E5 走 CPU(小内核启动开销主导);
  **off-policy(SAC/TD3/SAC-TR)走 GPU——实测快 5.8×**(批量更新主导,环境上 GPU 无碍);
- 双轨并行:`--algos` 过滤器把 off-policy 指到 GPU 轨(3 workers)与 CPU 轨(2 workers)同跑;
- `torch.set_num_threads(2)`;训练解释器:`miniconda3/envs/aitachi/python.exe`(torch 2.5.1+cu121)。

---

**作者:** Aitachi · **邮箱:** 44158892@qq.com · **GitHub:** https://github.com/aitachi/Reinforcement-Learning-Policy

**最后更新:** 2026-11-15(v2.2:E4/E5 新环境、16 算法、393 组、GPU 路由与双轨 sweep)
