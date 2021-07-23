import warnings
from typing import Any, Dict, Optional, Type, Union
from abc import ABC, abstractmethod
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F


class Expert_Goal_Space(ABC):
    def __init__(self, expert_func):
        self.expert_func = expert_func

    def get_goal(self, state):
        return self.expert_func(state)


class Base_Learned_Goal_Space(ABC):
    def __init__(self, extractorbuffer=None):
        return None

    def get_goal(self,state):
        return None

    def train(self,buffer=None):
        if buffer is None:
            """ use separete buffer """ 
        else:
            """sample data and train"""
            
class LESSON(Base_Learned_Goal_Space):
    temp = 0

