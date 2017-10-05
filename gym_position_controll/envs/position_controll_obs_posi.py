import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

from position_controll import PositionControllEnv

class PositionControllObsPosiEnv(PositionControllEnv):
  def __init__(self):
    PositionControllEnv.__init__(self)

    self.high = np.array([ self.max_position, 0])
    self.low  = np.array([-self.max_position, 0])
    
    self.observation_space = spaces.Box(self.low, self.high)

  def _step(self, action):
    obs, reward, done, _ = PositionControllEnv._step(self, action)
    return np.array([obs[0], 0]), reward, done, {}

  def _reset(self):
    self.state = PositionControllEnv._reset(self)
    return np.array([self.state[0], 0])

