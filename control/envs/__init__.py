from .nav_obstacle import NavObstacleEnv
from .hopper_lite import HopperLiteEnv
from .reach_grasp import ReachGraspEnv
from .grasp_seq import GraspSeqEnv
from .walker import WalkerEnv

ENVS = {
    "nav": NavObstacleEnv,        # E1 obstacle-avoidance navigation
    "hopper": HopperLiteEnv,      # E2 legged locomotion (monopod)
    "grasp": ReachGraspEnv,       # E3 reach-grasp-lift
    "graspseq": GraspSeqEnv,      # E4 continuous pick-lift-place cycles
    "walker": WalkerEnv,          # E5 whole-body biped gait coordination
}

__all__ = ["ENVS", "NavObstacleEnv", "HopperLiteEnv", "ReachGraspEnv",
           "GraspSeqEnv", "WalkerEnv"]
