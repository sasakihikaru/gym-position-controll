from gym.envs.registration import register

register(
  id='PositionControll-v0',
  entry_point='gym_position_controll.envs:PositionControllEnv',
)

register(
  id='PositionControll-v1',
  entry_point='gym_position_controll.envs:PositionControllObsPosiEnv',
)

register(
  id='PositionControll-v2',
  entry_point='gym_position_controll.envs:PositionControllObsVeloEnv',
)
