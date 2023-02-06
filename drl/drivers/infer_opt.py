import argparse
from pyexpat.errors import XML_ERROR_INVALID_TOKEN

import gym
import drl.agents
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from gym import spaces
from gym.utils import seeding

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

        reward[0] = reward[0] + -50*(s*ind_vt_g_1)  #(10*ind_vt_g_1*np.random.randn())#+ -85*(s*ind_vt_g_1) 

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
        reward[0] = reward[0] + -100*(s*ind_vt_g_1)#(10*ind_vt_g_1*np.random.randn())#+ -85*(s*ind_vt_g_1) #+ s*1*np.random.randn()#+ -120*(s*ind_vt_g_1) 
        
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

#def infer_opt(nepisodes, agent_id, env_id):
def infer_opt(nepisodes, agent_id, env_id, trial, critic_lr, actor_lr, gamma, tau, lower_quantiles, thresh_quantile):

    overshoot=[]
    violations=[]
    for trial in range(1):

        total_episodes = nepisodes

        # Environment
        if('Cheetah' in env_id):
            env = rewardWrapper_Cheetah(gym.make(env_id))
            threshold = 0.8
        if('car' in env_id):
            env = rewardWrapper_car1D(CarEnv())  
            threshold = 1.0  
        if('Hopper' in env_id):
            env = rewardWrapper_Hopper(gym.make(env_id))
            threshold = 0.03
        if('Walker' in env_id):
            env = rewardWrapper_Walker(gym.make(env_id))
            threshold = 0.2

        num_states = env.observation_space.shape[0]
        print("Size of State Space ->  {}".format(num_states))
        num_actions = env.action_space.shape[0]
        print("Size of Action Space ->  {}".format(num_actions))

        upper_bound = env.action_space.high[0]
        lower_bound = env.action_space.low[0]

        print("Max Value of Action ->  {}".format(upper_bound))
        print("Min Value of Action ->  {}".format(lower_bound))

        # Agent
        #agent = dnc2s_rl.agents.make(agent_id, env=env, critic_lr=1e-3, actor_lr=1e-3, tau=0.01, gamma=0.99)
        agent = drl.agents.make(agent_id, env=env, critic_lr=critic_lr, actor_lr=actor_lr, tau=tau, \
            gamma=gamma, thresh_quantile = thresh_quantile, lower_quantiles=lower_quantiles )
        agent.load(env_id, agent_id, trial+2)

        #local overshoot & violations list
        quant_of_interest=[]
        
        nsteps = 0
        
        velocity_arr = []
        position_arr = []
        for ep in range(1):

            counter=0
            prev_state = env.reset()
            episodic_reward = 0
            while True:
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                #env.render()

                if 'Torch' in agent_id:
                    tf_prev_state = torch.Tensor([prev_state])
                    action, noise = agent.action(tf_prev_state, 0.0)
                else:
                    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                    action, noise = agent.action(tf_prev_state)

                # action, noise = agent.action(tf_prev_state)
                # Receive state and reward from environment.

                if(('Cheetah' in env_id) or ('Hopper' in env_id) or ('Walker' in env_id)):
                    action  = action[0]
                # Add noise from epsitemic uncty
                #epistemic_noise_scale = 0.5*(agent.epistemic(tf_prev_state)).cpu().detach().numpy()
                #print(epistemic_noise_scale, action)
                #action = action + epistemic_noise_scale*np.random.randn(action.shape[0])
                #action = action + epistemic_noise_scale.reshape()*noise
                #print(action, epistemic_noise_scale, noise)

                state, reward, done, info = env.step(action)
                if('Cheetah' in env_id):
                    #print('ep:', ep, 'steps:', counter, 'pos:', state[0], 'vel:', info['x_velocity'], 'reward:', reward)#, 'noise:', epistemic_noise_scale)
                    #quant_of_interest.append(info['x_velocity'])
                    quant_of_interest.append(info[0])
                if('car' in env_id):
                    #print('ep:', ep, 'steps:', counter, 'pos:', state[0], 'vel:', state[1], 'reward:', reward)#, 'noise:', epistemic_noise_scale)
                    quant_of_interest.append(state[1])
                if(('Hopper' in env_id) or ('Walker' in env_id)):
                    #print('ep:', ep, 'steps:', counter, 'pos:', state[0], 'angle:', env.sim.data.qpos[2] , 'reward:', reward)#, 'noise:', epistemic_noise_scale)
                    #quant_of_interest.append(env.sim.data.qpos[2])
                    quant_of_interest.append(abs(info[0]))
                
                #print(abs(info[0]))

                episodic_reward += reward
                #velocity_arr.append(info['x_velocity'])
                velocity_arr.append(state[1])
                position_arr.append(state[0])


                counter+=1

                # End this episode when `done` is True
                if (done or (counter>=200)):
                    break

                # break
                prev_state = state

            quant_of_interest = np.array(quant_of_interest)

            # Find percent violations
            percent_failure = np.mean(((np.abs(quant_of_interest) - threshold)>0).astype(float))
            violations.append(percent_failure)
            print(percent_failure)

            # Find mean overshoot
            ov = np.abs(quant_of_interest[(np.abs(quant_of_interest) - threshold)>0]) - threshold
            if(np.isnan(np.mean(ov))):
                overshoot.append(0)
            else:
                overshoot.append(np.mean(ov))
            print(np.mean(ov))

    overshoot = np.array(overshoot)
    violations = np.array(violations)

    print('Mean-Overshoot:', np.round(np.mean(overshoot),2), '+-', np.round(np.std(overshoot),2))
    print('Violations:', np.round(np.mean(violations),2), '+-', np.round(np.std(violations),2))

    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=500)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG-v0')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG_ensemble-v3')
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='Torch_wcpg-v1')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='car_1D-v0')
    parser.add_argument("--env", help="Environment used for RL", type=str, default='HalfCheetah-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Walker2d-v3')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Hopper-v3')
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

    # Print input settings
    #infer_opt(nepisodes,agent_id,env_id)
    infer_opt(nepisodes,agent_id,env_id, trial, args.critic_lr, args.actor_lr, args.gamma, args.tau, args.lower_quantiles, args.thresh_quantile)
