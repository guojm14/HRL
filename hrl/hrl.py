import stable_baselines3.hrl.goal_space
from stable_baselines3.hrl.goal_space import Expert_Goal_Space, LESSON
import warnings
from typing import Any, Dict, Optional, Type, Union
from abc import ABC, abstractmethod
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.logger import Image
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, get_device
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

# for base algorithm
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG, DQN
_base_algo_ = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DDPG": DDPG, "DQN": DQN}

# for goal spaces

def create_hrl():
    """check the type of hrl and return the object"""
    
