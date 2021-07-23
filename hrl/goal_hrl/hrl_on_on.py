import stable_baselines3.hrl.goal_space
from stable_baselines3.hrl.goal_space import Expert_Goal_Space, LESSON
import warnings
from typing import Any, Dict, Optional, Type, Union
from abc import ABC, abstractmethod
import numpy as np
import torch as th
import gym
import time
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
from stable_baselines3.hrl.hrl_base import HRL_BASE
from stable_baselines3.common import utils
# for base algorithm
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG, DQN
_base_algo_ = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DDPG": DDPG, "DQN": DQN}

# for goal spaces


class HRL_ON_ON(HRL_BASE):
    """
    :param env: The environment to learn from(if registered in Gym, can be str)
    :param high_algorithm: The type of high level policy
    :param low_algorithm: The type of low level policy
    :param high_kwargs: Kwargs for high level policy
    :param low_kwargs: Kwargs for low level policy
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param tensorboard_log: The log location for tensorboard (if None, no logging)
    :param goal_space_method: The Goal Space for hierachical reinforcement learning(origin obervation space, a function specificed by human, learning).
    :param goal_space_kwargs: Kwargs for goal space methods
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
        high_policy_kwargs: Optional[Dict[str, Any]] =None,
        low_policy_kwargs: Optional[Dict[str, Any]]= None,
        device: Union[th.device, str] = "auto",
        goal_method_class: Optional[str] = None,
        goal_method_kwargs: Optional[Dict[str, Any]] = None,
        high_n_steps: int = 1,
        low_n_steps: int = 1000,
        goal_n_steps: int = 1000,
        high_rollout_steps: int = 1,
        reward_function: str = "l2",
        tensorboard_log: Optional[str] = None,
        max_low_steps: int = 5,
        high_gradient_steps: int = 1,
        low_gradient_steps: int = 1,
        high_batch_size: int = 256,
        low_batch_size: int = 256,
        low_policy_done_threshold: float = 0.1,
        verbose: int = 0,
        gamma: float = 1,
        # TO BE COMPLETED
    ):
        super(HRL_ON_ON,self).__init__(
            env,
            high_algorithm,
            low_algorithm,
            high_policy_kwargs,
            low_policy_kwargs,
            device,
            goal_method_class,
            goal_method_kwargs,
            reward_function,
            tensorboard_log,
            max_low_steps,
        )

        self._setup_model()

        self.goal_space_steps = 1000

    def _setup_model(self) -> None:

        # fake action space for high policy
        self.env.action_space = self.goal_space
        self.high_policy = self.high_policy_class(
            env=self.env, tensorboard_log=self.tensorboard_log, **self.high_policy_kwargs)
        self.env.action_space = self.action_space

        # fake obervations space for low level policy
        if isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("Error: The implementation doesn't support dictionary obervations")
        space = {"observation": self.observation_space, "desired_goal": self.goal_space}
        self.env.observation_space = gym.spaces.Dict(space)
        self.low_policy = self.low_policy_class(
            env=self.env, tensorboard_log=self.tensorboard_log, device=self.device, **self.low_policy_kwargs)
        self.env.observation_space = self.observation_space

        if self.goal_method_class is not None:
            self.goal_extractor = self.goal_method_class(self.goal_method_kwargs)
        self.low_n_steps = self.low_policy.n_steps
        self.high_n_steps = self.high_policy.n_steps

    def train_high(self) -> None:
        self.high_policy.tarin()

    def train_low(self) -> None:
        self.low_policy.train()

    def train_goal_space(self) -> None:
        """
        Update goal space.
        """

    def high_step(self, obs, goal):
        all_env = set(range(self.num_envs))
        active_env = set(range(self.num_envs))
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        for count in range(self.max_low_steps):
            with th.no_grad():
                obs_tensor = {"observation": self._last_obs, "desired_goal": goal}
                obs_tensor= obs_as_tensor(obs_tensor,self.device)
                #print(obs_tensor)
                actions, values, log_probs = self.low_policy.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = self.env.step(clipped_actions, active_env)
            #print(dones,infos)
            if count == 0:
                cumulative_rewards = rewards
            else:
                cumulative_rewards += rewards * self._last_episode_starts * self.gamma**count 
            for i, done in enumerate(dones):
                if done:
                    active_env -= {i}
            obs_tensor = {"observation": self._last_obs, "desired_goal": goal}
            #print(obs_tensor)
            self.low_policy.rollout_buffer.add(obs_tensor, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

            self.num_timesteps += self.env.num_envs

            if (self.num_timesteps // self.env.num_envs) % self.low_n_steps == 0:
                print("update")
                with th.no_grad():
                    obs_tensor = {"observation": self._last_obs, "desired_goal": goal}
                    obs_tensor= obs_as_tensor(obs_tensor,self.device)
                    _, values, _ = self.low_policy.policy.forward(obs_tensor)
                    self.low_policy.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
                logger.change_logger(self.logger_low)
                self.train_low()
                self.low_policy.rollout_buffer.reset()
            if (self.num_timesteps // self.env.num_envs) % self.low_n_steps == 0:
                self.train_goal_space()
            if len(active_env) == 0:
                break
        return new_obs, cumulative_rewards, dones, infos

    def collect_rollouts(self, env: VecEnv,n_rollout_steps:int):
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        self.high_policy.rollout_buffer.reset()
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.high_policy.policy.forward(obs_tensor)
            actions=actions.cpu().numpy()

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)  
            new_obs, rewards, dones, infos = self.high_step(self._last_obs,clipped_actions)
            n_steps += 1
            
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            self.high_policy.rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones
            
            """reset To be completed"""
        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.high_policy.policy.forward(obs_tensor)
        print(dones)
        self.high_policy.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
            


    def learn(
        self,
        total_timesteps: int,
    ) -> None:
        self._setup_learn(total_timesteps)


        while self.num_timesteps < total_timesteps:
            self.collect_rollouts(self.env, n_rollout_steps=self.high_n_steps)
            logger.change_logger(self.logger_high)
            self.high_policy.train()


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

        self._total_timesteps = total_timesteps

        self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
    
        # Configure logger's outputs
        utils.configure_logger(self.verbose, self.tensorboard_log, "high")
        self.logger_high = logger.get_current_logger()
        utils.configure_logger(self.verbose, self.tensorboard_log, "low")
        self.logger_low = logger.get_current_logger()
