from torch.tensor import Tensor
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
import stable_baselines3.hrl.goal_space
from stable_baselines3.hrl.goal_space import Expert_Goal_Space, LESSON
import warnings
from typing import Any, Dict, Optional, Type, Union, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import logger
from stable_baselines3.common.logger import Image
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, get_device
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.hrl.hrl_base import HRL_BASE
from stable_baselines3.common.torch_layers import CombinedExtractor
# for base algorithm
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG, DQN
_base_algo_ = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DDPG": DDPG, "DQN": DQN}

# for goal spaces

class HRL_OFF_OFF(HRL_BASE):
    """
    :param env: The environment to learn from(if registered in Gym, can be str)
    :param high_algorithm: The type of high level policy
    :param low_algorithm: The type of low level policy
    :param high_policy_kwargs: Kwargs for high level policy
    :param low_policy_kwargs: Kwargs for low level policy
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
        high_n_steps: int = 1,
        low_n_steps: int = 1000,
        goal_n_steps: int = 1000,
        high_rollout_steps: int = 1,
        reward_function: str = "l2",
        tensorboard_log: Optional[str] = None,
        max_low_steps: int = 10,
        high_gradient_steps: int = 1,
        low_gradient_steps: int = 1,
        high_batch_size: int = 256,
        low_batch_size: int = 256,
        low_policy_done_threshold: float = 0.1,
        verbose: int = 0,
        gamma: float = 1,
        # TO BE COMPLETED
    ):
        super(HRL_OFF_OFF, self).__init__(
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
            high_rollout_steps, 
            low_policy_done_threshold,
            verbose,
            gamma
        )

        self.high_n_steps = high_n_steps
        self.low_n_steps = low_n_steps
        self.goal_space_steps = goal_n_steps
        self.high_gradient_steps = high_gradient_steps
        self.low_gradient_steps = low_gradient_steps
        self.high_batch_size = high_batch_size
        self.low_batch_size = low_batch_size
        self._setup_model()

    def _setup_model(self) -> None:

        # fake action space for high policy
        self.env.action_space = self.goal_space
        self.check_policy_extractor(self.env.observation_space, self.high_policy_kwargs)
        self.high_policy = self.high_policy_class(
            env=self.env, tensorboard_log=self.tensorboard_log, **self.high_policy_kwargs)

        self.env.action_space = self.action_space

        # fake obervations space for low level policy
        self.env.observation_space = self.cat_observation_space(self.observation_space, self.goal_space)
        self.check_policy_extractor(self.env.observation_space, self.low_policy_kwargs)
        self.ensure_policy_multi_env_not_sync(self.low_policy_kwargs)
        self.low_policy = self.low_policy_class(
            env=self.env, tensorboard_log=self.tensorboard_log, device=self.device, **self.low_policy_kwargs)

        self.env.observation_space = self.observation_space

        if self.goal_method_class is not None:
            self.goal_extractor = self.goal_method_class(self.goal_method_kwargs)        

    def train_high(self) -> None:
        self.high_policy.tarin()

    def train_low(self) -> None:
        self.low_policy.train()

    def train_goal_space(self) -> None:
        """
        Update goal space.
        """
        return None

    def low_rewards(self, goal, new_obs):
        if self.goal_method_class is not None:
            with th.no_grad():
                goal = self.goal_extractor(goal)
                new_obs = self.goal_extractor(new_obs)
        reward = np.linalg.norm(goal - new_obs) 
        return reward

    def low_dones(self, goal, new_obs):
        if self.goal_method_class is not None:
            with th.no_grad():
                goal = self.goal_extractor(goal)
                new_obs = self.goal_extractor(new_obs)
        done = np.array([False if np.linalg.norm(goal - new_obs) > self.low_policy_done_threshold else True])
        return done

    def high_step(self, obs, goal):
        all_env = set(range(self.num_envs))
        active_env = set(range(self.num_envs))
        ones = np.zeros(self.num_envs)
        cumulative_rewards = np.zeros(self.num_envs)
        for count in range(self.max_low_steps):
            with th.no_grad():
                actions = self._sample_low_action(self._last_obs, goal)
                new_obs, rewards, dones, infos = self.env.step(actions, active_env)
            self.num_timesteps += len(active_env)
            cumulative_rewards += rewards * (1 - dones) * self.gamma**count
            print(actions.shape)
            for i in active_env:
                low_rewards = self.low_rewards(goal[i], new_obs[i])
                low_dones = self.low_dones(goal[i], new_obs[i])
                if not isinstance(self._last_obs, dict):
                    single_last_obs = self._last_obs[i]
                    single_new_obs = new_obs[i]
                else:
                    single_last_obs = {}
                    single_new_obs = {}
                    for k, v in self._last_obs.items():
                        single_last_obs[k] = v[i]
                        single_new_obs[k] = new_obs[k][i]
                self.low_policy.replay_buffer[i].add(
                    self.cat_obs_and_goal(single_last_obs, goal[i]),
                    self.cat_obs_and_goal(single_new_obs, goal[i]),
                    actions[i],
                    low_rewards,
                    low_dones,
                    (infos[i],),
                )

            for i, done in enumerate(dones):
                if done:
                    active_env -= {i}
                        
            self._last_obs = new_obs

            if self.num_timesteps >= (self.low_update_count + 1)* self.low_n_steps:
                self.low_policy.train(batch_size=int(self.low_batch_size/self.num_envs), gradient_steps=self.low_gradient_steps)
                self.low_update_count += 1
            if self.goal_method_class is not None:
                if self.num_timesteps >= (self.goal_method_update_count + 1) * self.goal_space_steps == 0:
                    self.train_goal_space()
                    self.goal_method_update_count += 1
            if len(active_env) == 0:
                break
        self._episode_num += dones.sum()
        return new_obs, cumulative_rewards, dones, infos

    def _sample_high_action(self, obs):
        #obs_tensor = obs_as_tensor(obs, self.device)
        actions, _ = self.high_policy.predict(obs, deterministic=False)
        #actions = actions.numpy()
        # Clip the actions to avoid out of bound error
        clipped_actions = actions
        if isinstance(self.goal_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.goal_space.low, self.goal_space.high)
        clipped_actions = th.from_numpy(clipped_actions)
        return clipped_actions

    def _sample_low_action(self, obs, goal):
        #obs_tensor = obs_as_tensor(obs, self.device)
        obs_tensor = self.cat_obs_and_goal(obs, goal)
        actions, _ = self.low_policy.predict(obs_tensor, deterministic=False)
        #actions = actions.numpy()
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        clipped_actions = th.from_numpy(clipped_actions)
        return clipped_actions

    def collect_rollouts(self, 
            env: VecEnv, 
            high_buffer: ReplayBuffer, 
            low_buffer: ReplayBuffer,
        ) -> None:
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        while n_steps < self.high_rollout_steps:
            with th.no_grad():
                actions = self._sample_high_action(self._last_obs)

            new_obs, rewards, dones, infos = self.high_step(self._last_obs, actions)
            self.high_timesteps += 1
            n_steps += 1
            self.high_policy.replay_buffer.add(
                self._last_obs,
                new_obs,
                actions,
                rewards,
                dones,
                infos,
            )
            self._last_obs = new_obs            

    def learn(
        self,
        total_timesteps: int,
    ) -> "HRL_OFF_OFF":

        self._setup_learn(
            total_timesteps
        )

        while self.num_timesteps < total_timesteps:
            self.collect_rollouts(self.env, 
                                  self.high_policy.replay_buffer, 
                                  self.low_policy.replay_buffer 
                                  )
            if self.high_timesteps % self.high_n_steps:
                self.high_policy.train(batch_size=int(self.high_batch_size/self.num_envs), gradient_steps=self.high_gradient_steps)

        return self

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

        return super()._setup_learn(
            total_timesteps
        )


