import argparse

import gym
import drl.agents
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from gym import spaces
from gym.utils import seeding
import time

import torch

class rewardWrapper_car1D(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # can check for the specific environment (more generic)
        
    def step(self, action):

        '''
        Gym wrapper for Reward: 

        '''
        next_state, reward, done, info = self.env.step(action)
        
        ind_vt_g_1 = 1 if((next_state[1])>=1.0) else 0
        r = np.random.uniform(0,1)
        #s = 1 if(r<=0.05) else 0
        s = 1 if(r<=0.1) else 0

        reward[0] = reward[0] + -20*(s*ind_vt_g_1)  #(10*ind_vt_g_1*np.random.randn())#+ -85*(s*ind_vt_g_1) 

        info = [next_state[1], ind_vt_g_1]
        
        return next_state, reward, done, info

class rewardWrapper_Cheetah(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # can check for the specific environment (more generic)
        
    def step(self, action):

        '''
        Gym wrapper for Reward: 

        '''
        next_state, reward, done, info = self.env.step(action)

        vel = info['x_velocity']        
        ind_vt_g_1 = 1 if(vel>=0.8) else 0
        #ind_vt_g_1 = 1 if(vel>=1.2) else 0
        r = np.random.uniform(0,1)
        s = 1 if(r<=0.05) else 0
        #s = 1 if(r<=0.1) else 0

        reward = np.array([reward])

        #if (ind_vt_g_1==1 and s==1):
            #print('True') 
        
        #print(reward)
        reward[0] = reward[0] + -100*(s*ind_vt_g_1) # -100*(s*ind_vt_g_1)#(10*ind_vt_g_1*np.random.randn())#+ -85*(s*ind_vt_g_1) #+ s*1*np.random.randn()#+ -120*(s*ind_vt_g_1) 
        
        info = [vel, ind_vt_g_1]
        
        return next_state, reward, done, info

class rewardWrapper_Hopper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # can check for the specific environment (more generic)
        
    def step(self, action):

        '''
        Gym wrapper for Reward: 

        '''
        next_state, reward, done, info = self.env.step(action)

        reward = np.array([reward])

        z, angle = self.env.sim.data.qpos[1:3]
        angle_greater = (angle<-0.03) or ( angle > 0.03)

        r = np.random.uniform(0,1)
        s = 1 if(r<=0.05) else 0
        #s = 1 if(r<=0.1) else 0

        reward[0] = reward[0] + -50*(s*angle_greater) 

        info = [angle, angle_greater]
        
        return next_state, reward, done, info

class rewardWrapper_Walker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # can check for the specific environment (more generic)
        
    def step(self, action):

        '''
        Gym wrapper for Reward: 

        '''
        next_state, reward, done, info = self.env.step(action)

        reward = np.array([reward])

        z, angle = self.env.sim.data.qpos[1:3]
        angle_greater = (angle<-0.2) or ( angle > 0.2)

        r = np.random.uniform(0,1)
        s = 1 if(r<=0.05) else 0
        #s = 1 if(r<=0.1) else 0

        reward[0] = reward[0] + -30*(s*angle_greater) 

        info = [angle, angle_greater]
        
        return next_state, reward, done, info
        
class CarEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.envname = 'car'
        self.dt = 0.1
        self.xg = 2.5#2.5
        self.max_acc = 1
        self.max_x = 3
        self.max_v = 3
        self.viewer = None
        
        high = np.array([1.0, self.max_v], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_acc, high=self.max_acc, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed() 

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        x = self.state[0]
        v = self.state[1]

        dt = self.dt

        u = np.clip(u, -self.max_acc, self.max_acc)[0]
        self.last_u = u  # for rendering

        
        #####################################
        # Introducing state dependent process noise
        x = x + (v*dt) + 0.5*(u* (dt)**2) 
        v = v + (u*dt)

        x = np.clip(x, -self.max_x, self.max_x)
        v = np.clip(v, -self.max_v, self.max_v)

        ind_xt_xg = 1 if((x <= self.xg + 0.15) and (x>= self.xg - 0.15)) else 0
        done = True if((ind_xt_xg==1) or (self.counter>=200)) else False

        if(done):
            pass
            #print('DONE', 'x:', x, 'v:', v)

        costs = v #+ 350*(ind_xt_xg) 
        self.counter+=1

        #####################################

        self.state = np.array([x, v])

        return self.state, np.array([costs]), done, {}

    def reset(self):
        x = np.random.uniform(-0.25,0.25)
        v = np.random.uniform(-0.25,0.25)
        self.state = np.array([x, v])
        self.counter=0
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def eval_policy(policy, env_name, seed, eval_episodes=1):
    if('car' in env_name):
        eval_env = rewardWrapper_car1D(CarEnv()) 
    elif('Cheetah' in env_id):
        eval_env = rewardWrapper_Cheetah(gym.make(env_name))
    elif('Hopper' in env_id):
        eval_env = rewardWrapper_Hopper(gym.make(env_name))
    elif('Walker' in env_id):
        eval_env = rewardWrapper_Walker(gym.make(env_name))
    else:
        eval_env = gym.make(env_name)
    #eval_env.seed(seed + 100)

    end_sim = 200

    avg_reward = 0.0
    avg_std = []
    avg_qty = []
    failures = []
    cvar = []

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = torch.Tensor([state])
        steps = 0
        while ((not done) and (steps<=end_sim)):
            action, noise = policy.action(state, 0.0)
            #avg_std.append(policy.variation_in_Q(state))
            #avg_std.append(policy.calc_variance(state, torch.Tensor([action]) ).cpu()[0,0])
            avg_std.append(0)
            #n_state, reward, done, _ = eval_env.step(action[0])
            if(('Cheetah' in env_name) or ('Hopper' in env_name) or ('Walker' in env_name) or ('Swimmer' in env_name) or ('Ant' in env_name) or ('Reacher' in env_name)):
                action  = action[0]
            n_state, reward, done, info = eval_env.step(action)
            avg_qty.append(info[0])
            failures.append(info[1])
            cvar.append(policy.CVaR(state, 0.0).item())
            avg_reward += reward
            state = torch.Tensor([n_state])
            steps+=1

    #avg_reward /= eval_episodes
    avg_reward /= steps
    avg_std = np.array(avg_std)
    final_std_dev = avg_std.mean()
    avg_qty_num = np.array(avg_qty).mean()
    percentage_failures = (np.array(failures)).mean()
    cvar = np.array(cvar).mean()

    print("---------------------------------------")
    #print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} : {final_std_dev} : {policy.reward_loss}")
    print(f"Evaluation over {eval_episodes} episodes: {np.round(avg_reward,3)} : {final_std_dev} : {np.round(policy.reward_loss,3)}: {np.round(avg_qty_num,3)} : {np.round(percentage_failures,3)}")
    #print(f"Evaluation over {eval_episodes} episodes: {avg_reward} : {final_std_dev} :  {avg_qty_num} : {percentage_failures}")
    print("---------------------------------------")
    return avg_reward, 0, avg_qty, percentage_failures, cvar #final_std_dev

def run_opt(nepisodes, agent_id, env_id, trial, critic_lr, actor_lr, gamma, tau, lower_quantiles, thresh_quantile):

    total_episodes = nepisodes
    file_name = 'saved_rewards/' + env_id + '_' + agent_id + '_' + str(trial)

    # Environment
    if('Mountain' in env_id):
        env = gym.make(env_id)
        end_sim=100
    elif('Pendulum' in env_id):
        env = gym.make(env_id)
        end_sim=100
    elif('Cheetah' in env_id):
        env = rewardWrapper_Cheetah(gym.make(env_id))
        #env = gym.make(env_id)
        end_sim=200
    elif('car' in env_id):
        env = rewardWrapper_car1D(CarEnv())  
        end_sim=200  
    elif('Hopper' in env_id):
        env = rewardWrapper_Hopper(gym.make(env_id))
        #env = gym.make(env_id)
        end_sim=200
    elif('Walker' in env_id):
        env = rewardWrapper_Walker(gym.make(env_id))
        #env = gym.make(env_id)
        end_sim=200
    elif('Swimmer' in env_id):
        #env = rewardWrapper_Walker(gym.make(env_id))
        env = gym.make(env_id)
        end_sim=1000

    elif('Ant' in env_id):
        #env = rewardWrapper_Walker(gym.make(env_id))
        env = gym.make(env_id)
        end_sim=1000

    elif('Reacher' in env_id):
        #env = rewardWrapper_Walker(gym.make(env_id))
        env = gym.make(env_id)
        end_sim=1000

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    # Agent
    agent = drl.agents.make(agent_id, env=env, critic_lr=critic_lr, actor_lr=actor_lr, tau=tau, \
    gamma=gamma, thresh_quantile = thresh_quantile, lower_quantiles=lower_quantiles )

    # To store reward history of each episode
    ep_reward_list = []
    inference_reward_list=[]
    inference_std_dev_list = []
    inference_reward_loss_list = []
    inference_avg_qty = []
    inference_per_failure = []
    inference_cvar = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    nsteps = 0
    while(nsteps<=30000):

        counter=0
        prev_state = env.reset()
        episodic_reward = 0
        while True:

            if 'Torch' in agent_id:
                tf_prev_state = torch.Tensor([prev_state])
                start_time = time.time()
                if(nsteps <= 1000):
                    action = env.action_space.sample()
                else:
                    action, noise = agent.action(tf_prev_state, 1.0)
                #print('Action time:', time.time()-start_time)
            else:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action, noise = agent.action(tf_prev_state)


            if(('Cheetah' in env_id) or ('Hopper' in env_id) or ('Walker' in env_id) or ('Swimmer' in env_id) or ('Ant' in env_id) or ('Reacher' in env_id)):
                action  = action[0]

            state, reward, done, info = env.step(action)
            '''if('Pendulum' in env_id or 'Mountain' in env_id):
                print('ep:', ep, 'steps:', counter, 'reward:', reward, 'state0:', state[0], 'state1:', state[1], 'action:', action)
            elif('Cheetah' in env_id):
                print('ep:', ep, 'steps:', counter, 'pos:', state[0], 'vel:', info['x_velocity'], 'reward:', reward)#, 'noise:', epistemic_noise_scale)
            elif('car' in env_id):
                print('ep:', ep, 'steps:', counter, 'pos:', state[0], 'vel:', state[1], 'reward:', reward)#, 'noise:', epistemic_noise_scale)
            #elif(('Hopper' in env_id) or ('Walker' in env_id)):
                #print('ep:', ep, 'steps:', counter, 'pos:', state[0], 'angle:', env.sim.data.qpos[2] , 'reward:', reward)#, 'noise:', epistemic_noise_scale)
            elif(('Swimmer' in env_id) or ('Ant' in env_id)):
                print('ep:', ep, 'steps:', counter, 'reward:', reward)#, 'noise:', epistemic_noise_scale)'''

            if((nsteps+1)%1000==0):
                    avg_return, final_std_dev, avg_qty, per_failure, cvar = eval_policy(agent, env_id, nsteps)
                    inference_reward_list.append(avg_return)
                    inference_std_dev_list.append(final_std_dev)
                    inference_reward_loss_list.append(agent.reward_loss)
                    inference_avg_qty.append(avg_qty)
                    inference_per_failure.append(per_failure)
                    inference_cvar.append(cvar)
                    
                    np.save(file_name, inference_reward_list)
                    #file_name_std_dev = 'saved_rewards/' + 'std_dev_' + env_id + '_' + agent_id + '_' + str(trial)
                    #np.save(file_name_std_dev, inference_std_dev_list)
                    #file_name_reward_loss = 'saved_rewards/' + 'reward_loss_' + env_id + '_' + agent_id + '_' + str(trial)
                    #np.save(file_name_reward_loss, inference_reward_loss_list)
                    file_name_reward_loss = 'saved_rewards/' + 'avg_qty_' + env_id + '_' + agent_id + '_' + str(trial)
                    np.save(file_name_reward_loss, inference_avg_qty)
                    file_name_reward_loss = 'saved_rewards/' + 'per_failure_' + env_id + '_' + agent_id + '_' + str(trial)
                    np.save(file_name_reward_loss, inference_per_failure)
                    file_name_cvar = 'saved_rewards/' + 'cvar_' + env_id + '_' + agent_id + '_' + str(trial)
                    np.save(file_name_cvar, inference_cvar)

                    agent.save(env_id, trial)

            nsteps += 1
            episodic_reward += reward

            agent.memory((prev_state, action, reward, state, 1-float(done)))

            start_time= time.time()
            agent.train()
            #print('Train_time:', time.time()-start_time)
            counter+=1

            # End this episode when `done` is True
            if (done or (counter>end_sim)):
                print("Num_samples : {} ;   Steps: {};   Episodic Reward is : {}".format(nsteps, counter, episodic_reward ))
                break

            prev_state = state

    agent.save(env_id, trial)
    
    #avg_reward_list = np.array(avg_reward_list)
    #save_path = 'saved_rewards/' + env_id + '_' + agent_id + '_' + str(trial) + '.txt'
    #np.savetxt(save_path, avg_reward_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=500)
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG-v0')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG_ensemble-v1')
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG_UP-v2')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='car_1D-v0')
    parser.add_argument("--env", help="Environment used for RL", type=str, default='HalfCheetah-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Walker2d-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Hopper-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='MountainCarContinuous-v0')
    parser.add_argument("--trial", help="Trial num of algo run", type=int, default=0)
    parser.add_argument("--critic_lr", help="Learning rate of critic", type=float, default=0.001)
    parser.add_argument("--actor_lr", help="Learning rate of actor", type=float, default=0.001)
    parser.add_argument("--gamma", help="Gamma", type=float, default=0.99)
    parser.add_argument("--tau", help="Tau", type=float, default=0.05)
    parser.add_argument("--lower_quantiles", help="Tau", type=float, default=0.1)
    parser.add_argument("--thresh_quantile", help="Tau", type=float, default=0.9)

    # Get input arguments
    args = parser.parse_args()
    nepisodes = args.nepisodes
    agent_id = args.agent
    env_id = args.env
    trial = args.trial
    lower_quantiles = args.lower_quantiles
    thresh_quantile = args.thresh_quantile

    # Print input settings
    run_opt(nepisodes,agent_id,env_id, trial, args.critic_lr, args.actor_lr, args.gamma, args.tau, lower_quantiles, thresh_quantile)

