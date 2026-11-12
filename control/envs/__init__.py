from .nav_obstacle import NavObstacleEnv
from .hopper_lite import HopperLiteEnv
from .reach_grasp import ReachGraspEnv

ENVS = {
    "nav": NavObstacleEnv,        # E1 obstacle-avoidance navigation
    "hopper": HopperLiteEnv,      # E2 legged locomotion
    "grasp": ReachGraspEnv,       # E3 reach-grasp-lift
}

__all__ = ["ENVS", "NavObstacleEnv", "HopperLiteEnv", "ReachGraspEnv"]
