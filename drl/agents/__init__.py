from drl.agents.registration import register, make



register(
    id='Torch_QR_EVT-v1',
    entry_point = 'drl.agents.drl_agents:Torch_QR_EVT_v1'
)


register(
    id='Torch_QR_TD3_EVT-v1',
    entry_point = 'drl.agents.drl_agents:Torch_QR_TD3_EVT_v1'
)

