# Findings draft (auto-generated, curate before use)

## E1 Nav (dense, final mean±std / AUC / wall)
- sac: 0.830±0.027 auc=0.561 wall=2847s
- ppg: 0.673±0.037 auc=0.633 wall=39s
- trpo: 0.669±0.023 auc=0.616 wall=335s
- ppo_kl: 0.658±0.035 auc=0.625 wall=338s
- ppo_rb: 0.655±0.013 auc=0.655 wall=239s
- ppo: 0.652±0.021 auc=0.663 wall=336s
- espo: 0.648±0.043 auc=0.560 wall=293s
- tr_ppo_rb: 0.625±0.016 auc=0.494 wall=312s
- grpo: 0.557±0.070 auc=0.425 wall=236s
- a2c: 0.547±0.020 auc=0.439 wall=317s
- td3: 0.434±0.202 auc=0.430 wall=967s
- dapo: 0.324±0.032 auc=0.301 wall=214s

## E2 Hopper (dense, final mean±std / AUC / wall)
- sac: 0.899±0.001 auc=0.889 wall=4403s
- ppo_kl: 0.785±0.121 auc=0.419 wall=555s
- espo: 0.755±0.035 auc=0.612 wall=759s
- tr_ppo_rb: 0.754±0.073 auc=0.546 wall=639s
- trpo: 0.681±0.086 auc=0.404 wall=986s
- ppo_rb: 0.669±0.136 auc=0.348 wall=572s
- ppo: 0.620±0.165 auc=0.329 wall=532s
- td3: 0.167±0.000 auc=0.165 wall=2364s
- a2c: 0.027±0.003 auc=0.011 wall=587s
- ppg: 0.020±0.003 auc=0.025 wall=947s
- grpo: 0.017±0.003 auc=0.013 wall=1388s
- dapo: 0.015±0.001 auc=0.012 wall=1710s

## E3 Grasp (dense, final mean±std / AUC / wall)
- trpo: 1.000±0.000 auc=0.692 wall=333s
- ppg: 0.998±0.001 auc=0.649 wall=40s
- ppo: 0.996±0.003 auc=0.640 wall=356s
- ppo_rb: 0.995±0.001 auc=0.626 wall=221s
- ppo_kl: 0.992±0.006 auc=0.616 wall=304s
- tr_ppo_rb: 0.694±0.162 auc=0.371 wall=210s
- sac: 0.629±0.229 auc=0.365 wall=3033s
- espo: 0.568±0.138 auc=0.373 wall=181s
- a2c: 0.059±0.084 auc=0.008 wall=165s
- dapo: 0.000±0.000 auc=0.000 wall=102s
- grpo: 0.000±0.000 auc=0.000 wall=93s
- td3: 0.000±0.000 auc=0.000 wall=1774s

## E1 Nav reward axis
- ppo: dense=0.652, curriculum=0.643, pbs=0.577, sparse=0.286
- trpo: dense=0.669, curriculum=0.656, pbs=0.604, sparse=0.323
- grpo: dense=0.557, curriculum=0.395, pbs=0.290, sparse=0.288

## E2 Hopper reward axis
- ppo: dense=0.620, curriculum=0.546, sparse=0.407, pbs=0.133
- trpo: dense=0.681, sparse=0.612, curriculum=0.526, pbs=0.120
- grpo: sparse=0.109, pbs=0.097, dense=0.017, curriculum=0.015

## E3 Grasp reward axis
- ppo: dense=0.996, curriculum=0.993, pbs=0.944, sparse=0.009
- trpo: dense=1.000, curriculum=0.996, pbs=0.984, sparse=0.000
- grpo: curriculum=0.000, dense=0.000, pbs=0.000, sparse=0.000
