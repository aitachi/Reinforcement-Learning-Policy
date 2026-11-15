"""Algorithm registry for the control suite."""
from __future__ import annotations

from .ppo_family import PPO, PPG, A2C
from .trpo import TRPO
from .grpo_dapo import GRPO, DAPO
from .sac import SAC, SACTR
from .td3 import TD3
from .es import ES


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
    if name == "sac_tr":
        return SACTR(obs_dim, act_dim, device, cfg)
    if name == "ppo_ent":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "clip", "ent_coef": 0.01})
    if name == "ppo_novc":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "clip", "value_clip": False})
    if name == "ppo_e01":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "clip", "clip": 0.1})
    if name == "ppo_e03":
        return PPO(obs_dim, act_dim, device, {**cfg, "method": "clip", "clip": 0.3})
    if name == "es":
        return ES(obs_dim, act_dim, device, cfg)
    raise ValueError(name)


ALGO_NAMES = ["ppo", "ppo_kl", "ppo_rb", "tr_ppo_rb", "espo", "ppg",
              "trpo", "a2c", "grpo", "dapo", "sac", "td3",
              "sac_tr", "ppo_ent", "ppo_novc", "es"]

# main-grid vs auxiliary (ablation / sensitivity) registrations
SENSE_ALGOS = ["ppo_e01", "ppo_e03"]

ALGO_LABELS = {
    "ppo": "PPO-Clip", "ppo_kl": "PPO-KL", "ppo_rb": "PPO-RB",
    "tr_ppo_rb": "TR-PPO-RB", "espo": "ESPO", "ppg": "PPG",
    "trpo": "TRPO", "a2c": "A2C", "grpo": "GRPO-C", "dapo": "DAPO-C",
    "sac": "SAC", "td3": "TD3", "sac_tr": "SAC-TR",
    "ppo_ent": "PPO-Ent", "ppo_novc": "PPO-NoVC", "es": "ES",
    "ppo_e01": "PPO-Clip(eps=0.1)", "ppo_e03": "PPO-Clip(eps=0.3)",
}

ALGO_PAPERS = {
    "ppo": "Schulman et al., 2017", "ppo_kl": "Schulman et al., 2017 (KL variant)",
    "ppo_rb": "Yuan et al., 2020", "tr_ppo_rb": "Yuan et al., 2020 (trust-region)",
    "espo": "Li et al., 2023", "ppg": "Cobbe et al., 2021", "trpo": "Schulman et al., 2015",
    "a2c": "Mnih et al., 2016", "grpo": "Shao et al., 2024", "dapo": "Yu et al., 2025",
    "sac": "Haarnoja et al., 2018", "td3": "Fujimoto et al., 2018",
    "sac_tr": "this suite (KL-TR actor, cf. Schulman et al., 2017)",
    "ppo_ent": "Schulman et al., 2017 (entropy ablation)",
    "ppo_novc": "Schulman et al., 2017 (value-clip ablation)",
    "es": "Salimans et al., 2017",
}
