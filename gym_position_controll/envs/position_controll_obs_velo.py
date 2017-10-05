import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

from position_controll import PositionControllEnv

class PositionControllObsVeloEnv(PositionControllEnv):
  def __init__(self):
    PositionControllEnv.__init__(self)

    self.high = np.array([0,  self.max_speed])
    self.low  = np.array([0, -self.max_speed])
    
    self.observation_space = spaces.Box(self.low, self.high)

  def _step(self, action):
    obs, reward, done, _ = PositionControllEnv._step(self, action)
    return np.array([0, obs[1]]), reward, done, {}

  def _reset(self):
    self.state = PositionControllEnv._reset(self)
    return np.array([0, self.state[1]])

