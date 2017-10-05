import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

class PositionControllEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
  }

  def __init__(self):
    self.max_position = 10.0
    self.max_action = 4.0
    self.max_speed = 6

    self.high = np.array([ self.max_position,  self.max_speed])
    self.low  = np.array([-self.max_position, -self.max_speed])
    
    self.action_space = spaces.Box(-self.max_action, self.max_action, shape=(1,))
    self.observation_space = spaces.Box(self.low, self.high)

    self._seed()
    self.goal_position = 0

    self.viewer = None

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    assert self.action_space.contains(action), '%r (%s) invalid' % (action, type(action))

    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([0, 0.1])
    self.state = np.dot(A, self.state) + B * action[0]
    self.state[0] += np.random.normal(0, 0.05)
    self.state[1] += np.random.normal(0, 0.01)
    self.state = np.clip(self.state, self.low, self.high)
    if self.state[0] == self.max_position or self.state[0] == -self.max_position:
      self.state[1] = 0
    obs = self.state
    obs[0] += np.random.normal(0, 0.05)
    obs[1] += np.random.normal(0, 0.01)

    Q = np.diag([0.25, 0.1])
    R = 0.1
    reward = - np.linalg.multi_dot([self.state.T, Q, self.state]) - R * action**2
    reward = (np.exp(reward)) ** 2

    done = np.sign((self.state[0]-self.goal_position) * self._diff) == -1 
    self._diff = self.state[0] - self.goal_position

    return obs, reward, done, {}

  def _reset(self):
    init_posi = np.random.uniform(6, 8) * [-1, 1][np.random.randint(2)]
    init_velo = np.random.uniform(0, 0.5) * [-1, 1][np.random.randint(2)]
    self.state = np.array([init_posi, init_velo])
    self._diff = np.sign(self.state[0]-self.goal_position)
    return self.state

  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    screen_width = 600
    screen_height = 400
    
    world_center = (screen_width/2, screen_height/2)
    state_radius = 150
    r = 10

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)

      # state line
      xs = np.linspace(0, 1, 10)
      ys = np.zeros(10)
      xys = list(zip(xs*screen_width, ys))
      self.state_line = rendering.make_polyline(xys)
      self.state_line.add_attr(rendering.Transform(translation=[0, world_center[1]]))
      self.viewer.add_geom(self.state_line)

      # goal
      self.goal = rendering.make_circle(r)
      self.goal.set_color(255, 0, 0)
      self.goal.add_attr(rendering.Transform(translation=[self.goal_position+world_center[0], world_center[1]]))
      self.viewer.add_geom(self.goal)
      
      # agent circle
      self.agent = rendering.Transform()
      agent_circle = rendering.make_circle(r)
      agent_circle.set_color(0, 255, 0)
      agent_circle.add_attr(self.agent)
      self.viewer.add_geom(agent_circle)

    pos = world_center[0] + self.state[0] / self.max_position * world_center[0]
    self.agent.set_translation(pos, world_center[1])


    return self.viewer.render(return_rgb_array = mode=='rgb_array')


