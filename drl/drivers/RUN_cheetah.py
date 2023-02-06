from os import system

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_EVT-v1 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_EVT-v1 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_EVT-v1 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_EVT-v1 --env HalfCheetah-v3 --trial 23 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_EVT-v1 --env HalfCheetah-v3 --trial 24 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
'''

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 23 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 24 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
'''

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based-v0 --env HalfCheetah-v3 --trial 0 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based-v0 --env HalfCheetah-v3 --trial 1 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based-v0 --env HalfCheetah-v3 --trial 2 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
'''

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_DDPG-v0 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
'''

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based_EVT-v1 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based_EVT-v1 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based_EVT-v1 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based_EVT-v1 --env HalfCheetah-v3 --trial 23 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_model_based_EVT-v1 --env HalfCheetah-v3 --trial 24 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
'''

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 23 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 24 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
'''

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 23 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 24 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
'''

'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_SAC-v1 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_SAC-v1 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_SAC-v1 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_SAC-v1 --env HalfCheetah-v3 --trial 23 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_SAC-v1 --env HalfCheetah-v3 --trial 24 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
'''


'''s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_wcpg-v1 --env HalfCheetah-v3 --trial 20 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_wcpg-v1 --env HalfCheetah-v3 --trial 21 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_wcpg-v1 --env HalfCheetah-v3 --trial 22 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3"
'''


#########################################################################################
s1 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 0 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 1 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 2 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 3 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent Torch_QR_TD3_EVT-v1 --env HalfCheetah-v3 --trial 4 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"

system(s1)
system(s2)
system(s3)
system(s4)
system(s5)

s1 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_TD3-v0 --env HalfCheetah-v3 --trial 0 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s2 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_TD3-v0 --env HalfCheetah-v3 --trial 1 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s3 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_TD3-v0 --env HalfCheetah-v3 --trial 2 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s4 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_TD3-v0 --env HalfCheetah-v3 --trial 3 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"
s5 = "python drivers/run_opt3.py --nepisodes 100 --agent TorchQR_TD3-v0 --env HalfCheetah-v3 --trial 4 --tau 0.01 --actor_lr 1e-3 --critic_lr 1e-3 --lower_quantiles 0.05 --thresh_quantile 0.95"

system(s1)
system(s2)
system(s3)
system(s4)
system(s5)