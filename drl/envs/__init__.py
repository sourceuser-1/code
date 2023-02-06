from gym.envs.registration import register

register(
    id='car_1D-v0',
    entry_point='drl.envs.drl_envs:CarEnv',
)
register(
    id='ExaBooster-v3',
    entry_point='drl.envs.drl_envs:ExaBooster_v3'
)
