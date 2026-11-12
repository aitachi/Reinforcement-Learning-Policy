"""Algorithm registry for the control suite."""
from __future__ import annotations

from .ppo_family import PPO, PPG, A2C
from .trpo import TRPO
from .grpo_dapo import GRPO, DAPO
from .sac import SAC
from .td3 import TD3


def make_algo(name, obs_dim, act_dim, device, cfg):
    if name == "ppo":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "clip"})
    if name == "ppo_kl":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "kl"})
    if name == "ppo_rb":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "ppo_rb"})
    if name == "tr_ppo_rb":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "tr_ppo_rb"})
    if name == "espo":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "espo"})
    if name == "ppg":
        return PPG(obs_dim, act_dim, device, cfg)
    if name == "trpo":
        return TRPO(obs_dim, act_dim, device, cfg)
    if name == "a2c":
        return A2C(obs_dim, act_dim, device, cfg)
    if name == "grpo":
        return GRPO(obs_dim, act_dim, device, cfg)
    if name == "dapo":
        return DAPO(obs_dim, act_dim, device, cfg)
    if name == "sac":
        return SAC(obs_dim, act_dim, device, cfg)
    if name == "td3":
        return TD3(obs_dim, act_dim, device, cfg)
    raise ValueError(name)


ALGO_NAMES = ["ppo", "ppo_kl", "ppo_rb", "tr_ppo_rb", "espo", "ppg",
              "trpo", "a2c", "grpo", "dapo", "sac", "td3"]

ALGO_LABELS = {
    "ppo": "PPO-Clip", "ppo_kl": "PPO-KL", "ppo_rb": "PPO-RB",
    "tr_ppo_rb": "TR-PPO-RB", "espo": "ESPO", "ppg": "PPG",
    "trpo": "TRPO", "a2c": "A2C", "grpo": "GRPO-C", "dapo": "DAPO-C",
    "sac": "SAC", "td3": "TD3",
}

ALGO_PAPERS = {
    "ppo": "Schulman et al., 2017", "ppo_kl": "Schulman et al., 2017 (KL variant)",
    "ppo_rb": "Yuan et al., 2020", "tr_ppo_rb": "Yuan et al., 2020 (trust-region)",
    "espo": "Li et al., 2023", "ppg": "Cobbe et al., 2021", "trpo": "Schulman et al., 2015",
    "a2c": "Mnih et al., 2016", "grpo": "Shao et al., 2024", "dapo": "Yu et al., 2025",
    "sac": "Haarnoja et al., 2018", "td3": "Fujimoto et al., 2018",
}
