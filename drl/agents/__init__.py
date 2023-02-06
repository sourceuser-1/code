from drl.agents.registration import register, make


register(
    id='TorchQR_DDPG-v0',
    entry_point = 'drl.agents.drl_agents:Torch_QR_DDPG'
)

register(
    id='Torch_QR_EVT-v1',
    entry_point = 'drl.agents.drl_agents:Torch_QR_EVT_v1'
)


register(
    id='Torch_QR_model_based_EVT-v1',
    entry_point = 'drl.agents.drl_agents:Torch_QR_model_based_EVT_v1'
)

register(
    id='TorchQR_TD3-v0',
    entry_point = 'drl.agents.drl_agents:Torch_QR_TD3'
)

register(
    id='Torch_QR_TD3_EVT-v1',
    entry_point = 'drl.agents.drl_agents:Torch_QR_TD3_EVT_v1'
)

register(
    id='Torch_QR_SAC-v1',
    entry_point = 'drl.agents.drl_agents:Torch_QR_SAC_v1'
)

register(
    id='Torch_wcpg-v1',
    entry_point = 'drl.agents.drl_agents:Torch_WCPG'
)