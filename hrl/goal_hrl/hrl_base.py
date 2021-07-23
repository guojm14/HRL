import stable_baselines3.hrl.goal_space
from stable_baselines3.hrl.goal_space import Expert_Goal_Space, LESSON
import warnings
from typing import Any, Dict, Optional, Type, Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch as th
import time
import gym
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import logger, utils
from stable_baselines3.common.logger import Image
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, get_device
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.torch_layers import CombinedExtractor


# for base algorithm
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG, DQN
_base_algo_ = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DDPG": DDPG, "DQN": DQN}

# for goal spaces


class HRL_BASE(ABC):
    """
    :param env: The environment to learn from(if registered in Gym, can be str)
    :param high_algorithm: The type of high level policy
    :param low_algorithm: The type of low level policy
    :param high_kwargs: Kwargs for high level policy
    :param low_kwargs: Kwargs for low level policy
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param tensorboard_log: The log location for tensorboard (if None, no logging)
    :param goal_method_class: The Goal Space for hierachical reinforcement learning(origin obervation space, a function specificed by human, learning).
    :param goal_method_kwargs: Kwargs for goal space methods
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param reward_function: The fucntion for calulating reward(dense reward using Dot production or Euclidean distance, spase reward when (distance < threshold) or = when goal space is discrete)
    :param her: Whether to use hindsight experience replay
    :param her_kwargs: Kwargs for hindsight experience replay
    :param goal_correction: Whether to use goal correction in HIRO
    TO BE COMPLETED
    """

    def __init__(
        self,
        env: GymEnv,
        high_algorithm: Union[str, Type[BaseAlgorithm]],
        low_algorithm: Union[str, Type[BaseAlgorithm]],
        high_policy_kwargs: Dict[str, Any],
        low_policy_kwargs: Dict[str, Any],
        device: Union[th.device, str] = "auto",
        goal_method_class: Optional[str] = None,
        goal_method_kwargs: Optional[Dict[str, Any]] = None,
        reward_function: str = "l2",
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        max_low_steps: int = 10,
        high_rollout_steps: int = 1,
        low_policy_done_threshold: float = 0.1,
        verbose: int = 0,
        gamma: float = 1,
    ):

        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.verbose = verbose
        self.gamma = gamma
        if isinstance(high_algorithm, str):
            self.high_policy_class = _base_algo_[high_algorithm]
        else:
            self.high_policy_class = high_algorithm
        self.high_policy_kwargs = high_policy_kwargs
        high_policy_on_policy_flag = (self.high_policy_class.__base__.__name__ == "OnPolicyAlgorithm")

        if isinstance(low_algorithm, str):
            self.low_policy_class = _base_algo_[high_algorithm]
        else:
            self.low_policy_class = low_algorithm
        self.low_policy_kwargs = low_policy_kwargs
        low_policy_on_policy_flag = (self.high_policy_class.__base__.__name__ == "OnPolicyAlgorithm")
        self.low_policy_done_threshold = low_policy_done_threshold
        # self.her = her
        # self.her_kwargs = her_kwargs

        # if her:
        #     if low_policy_on_policy_flag:
        #        raise ValueError("Error: The on-policy methods don't support hindsight experience replay")

        self.goal_method_class = goal_method_class
        self.goal_method_kwargs = goal_method_kwargs
        if self.goal_method_class is not None:
            self.goal_space = self.goal_method_kwargs["goal_space"]
        else:
            self.goal_space = self.observation_space

        self.device = get_device(device)
        if verbose > 0:
            print(f"Using {self.device} device")

        self.tensorboard_log = tensorboard_log
        self.max_low_steps = max_low_steps
        self.high_rollout_steps = high_rollout_steps
        self._last_obs = None
        self.low_update_count = 0
        self.goal_method_update_count = 0

    def check_policy_extractor(self, observation_space, policy_kwargs) -> None:
        if isinstance(observation_space, gym.spaces.Dict):
            if "policy_kwargs" not in policy_kwargs:
                policy_kwargs["policy_kwargs"] = {"features_extractor_class":CombinedExtractor}
                print("features_extractor_class of policy_kwargs of policy_kwargs must be CombinedExtractor when observation_space is gym.spaces.Dict}")
            elif "features_extractor_class" not in policy_kwargs["policy_kwargs"]:
                policy_kwargs["policy_kwargs"]["features_extractor_class"] = CombinedExtractor
                print("features_extractor_class of policy_kwargs of policy_kwargs must be CombinedExtractor when observation_space is gym.spaces.Dict}")
            else:
                print("features_extractor_class of policy_kwargs of policy_kwargs must be CombinedExtractor when observation_space is gym.spaces.Dict}")
                raise

    def ensure_policy_multi_env_not_sync(self, policy_kwargs) -> None:
        if "multi_env_not_sync" not in policy_kwargs:
            policy_kwargs["multi_env_not_sync"] = True
        elif not policy_kwargs["multi_env_not_sync"]:
            print("multi_env_not_sync must be True when env not sync")
            raise

    def cat_observation_space(self, observation_space, goal_space):
        if isinstance(observation_space, gym.spaces.Dict):
            space = {}
            for k, v in observation_space.spaces.items():
                space[k] = v
            space["desired_goal"] = goal_space
        else:
            space = {"observation": observation_space, "desired_goal": goal_space}
        return gym.spaces.Dict(space)
    
    def cat_obs_and_goal(self, obs, goal):
        if not isinstance(obs, dict):
            obs = {"observation": obs, "desired_goal": goal}
        else:
            obs["desired_goal"] = goal
        return obs

    def _setup_model(self) -> None:

        self.env.action_space = self.goal_space
        self.high_policy = self.high_policy_class(
            env=self.env, tensorboard_log=self.tensorboard_log, **self.high_policy_kwargs)
        self.env.action_space = self.action_space

        if self.her:
            self.env.observation_space = {}
            self.low_policy = self.low_policy_class(env=self.env, tensorboard_log=self.tensorboard_log,
                                                    replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=self.her_kwargs, device=self.device, **self.low_policy_kwargs)
        else:
            self.low_policy = self.low_policy_class(
                env=self.env, tensorboard_log=self.tensorboard_log, device=self.device, **self.low_policy_kwargs)

        if self.goal_method_class is not None:
            self.goal_extractor = self.goal_method_class(self.goal_method_kwargs)

    def train_high(self) -> None:
        """
        Update high policy.
        """
        if self.high_policy_on_policy_flag:
            self.high_policy.tarin()
        else:
            self.high_policy.train(batch_size=self.high_policy.batch_size,
                                   gradient_steps=self.high_policy.gradient_steps)

    def train_low(self) -> None:
        """
        Update low policy.
        """
        if self.low_policy_on_policy_flag:
            self.low_policy.train()
        else:
            self.low_policy.train(batch_size=self.low_policy.batch_size,
                                  gradient_steps=self.low_policy.gradient_steps)

    def train_goal_space(self) -> None:
        """
        Update goal space.
        """

    def high_step_low_on_policy(self, obs, goal) -> None:
        active_env = range(self.num_envs)

        for _ in range(self.max_low_steps):
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

    def high_step_low_off_policy(self, goal) -> None:
        return None

    def collect_rollouts_high_on(self, env: VecEnv):
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0

    def learn(
        self,
        total_timesteps: int,
    ) -> None:
        self._setup_learn(total_timesteps)

        if self.low_policy_on_policy_flag:
            self.low_n_steps = self.low_policy_kwargs["n_steps"]
        else:
            temp = 0
        # while self.num_timesteps < total_timesteps:
        #     if x:
        #         self.train_high()
        #     if y:
        #         self.train_low()

        #     self.num_timesteps += 1

    def _setup_learn(
        self,
        total_timesteps: int,
    ) -> None:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on

        To Be Completed
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        self.num_timesteps = 0
        self.high_timesteps = 0
        self._episode_num = 0
        self._total_timesteps = total_timesteps

        if self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.num_envs), dtype=bool)

        # Configure logger's outputs
        utils.configure_logger(self.verbose, self.tensorboard_log, "high", reset_num_timesteps=True)
        self.logger_high = logger.get_logger()
        utils.configure_logger(self.verbose, self.tensorboard_log, "low", reset_num_timesteps=True)
        self.logger_low = logger.get_logger()


