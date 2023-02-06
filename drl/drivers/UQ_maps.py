import argparse

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
        
    def step(self, action, state):

        '''
        Gym wrapper for Reward: 

        '''
        next_state, reward, done, info = self.env.step(action, state)
        
        ind_vt_g_1 = 1 if(abs(next_state[1])>1.0) else 0
        r = np.random.uniform(0,1)
        s = 1 if(r<=0.07) else 0

        reward[0] = reward[0] + -85*(s*ind_vt_g_1) 
        
        return next_state, reward, done, {}
        
class CarEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.envname = 'car'
        self.dt = 0.1
        self.xg = 2.5
        self.max_acc = 1
        self.max_x = 5
        self.max_v = 3
        self.viewer = None
        
        high = np.array([1.0, self.max_v], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_acc, high=self.max_acc, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed() 
        self.counter=0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u, state):

        self.state = state

        x = self.state[0,0]
        v = self.state[0,1]

        dt = self.dt

        u = np.clip(u, -self.max_acc, self.max_acc)[0]
        self.last_u = u  # for rendering

        
        #####################################
        # Introducing state dependent process noise
        x = x + (v*dt) + 0.5*(u* (dt)**2) 
        v = v + (u*dt)

        x = np.clip(x, -self.max_x, self.max_x)
        v = np.clip(v, -self.max_v, self.max_v)

        ind_xt_xg = 1 if((x <= self.xg + 0.1) and (x>= self.xg - 0.1)) else 0
        done = True if((ind_xt_xg==1) or (self.counter>=100)) else False

        if(done):
            print('DONE', 'x:', x, 'v:', v)

        costs = -10 + 350*(ind_xt_xg) 
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

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



def run_opt(nepisodes, agent_id, env_id, trial):

    total_episodes = nepisodes

    # Environment
    #env = rewardWrapper_car1D(gym.make(env_id))
    env = rewardWrapper_car1D(CarEnv())    

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    # Agent
    agent = drl.agents.make(agent_id, env=env)
    agent.load(env_id, agent_id, trial)

    '''# To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    state_0 = []
    state_1 = []

    # Takes about 4 min to train
    #plt.ion()
    # PLot results
    fig, axs = plt.subplots(2)
    fig.suptitle('RL results')
    nsteps = 0
    for ep in range(1):

        prev_state = env.reset()
        episodic_reward = 0
        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            #env.render()

            if 'Torch' in agent_id:
                tf_prev_state = torch.Tensor([prev_state])
                action, noise = agent.action(tf_prev_state)
            else:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action, noise = agent.action(tf_prev_state)

            # action, noise = agent.action(tf_prev_state)
            # Receive state and reward from environment.
            state, reward, done, info = env.step(action)
            print('pos:', state[0], 'vel:', state[1], 'reward:', reward)
            nsteps += 1
            episodic_reward += reward

            state_0.append(state[0])
            state_1.append(state[1])

            # End this episode when `done` is True
            if done:
                break

            # break
            prev_state = state

    axs[0].plot(state_0)
    axs[0].set_ylabel("Position")
    axs[1].plot(state_1)
    axs[1].set_ylabel("Velocity")

    plt.show(block=True)
    # # Plotting graph
    # # Episodes versus Avg. Rewards
    # plt.plot(avg_reward_list)
    # plt.xlabel("Episode")
    # plt.ylabel("Avg. Epsiodic Reward")
    # plt.show()
    # plt.savefig('Pendulum_reward.png')
    # sys.exit(0)'''

    POS=[]
    VEL=[]
    
    ALEA=[]
    EPIST=[]
    CVAR=[]
    SURR=[]

    for trial in range(1000):

        print(trial)

        velocity = np.random.uniform(0.0, 1.7)
        position = np.random.uniform(0.0, 2.5)
        #velocity = np.random.uniform(-1.0, 3.0)
        #position = np.random.uniform(-0.5, 3.0)

        VEL.append(velocity)
        POS.append(position)
        
        tr = 0

        #worker = worker_module(args, agent, env)

        state = torch.Tensor([[position, velocity]])
        count=0

        action, nosie = agent.action(state)
        next_state, reward, done, _ = env.step(action, state.numpy())
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([[reward]])

        cvar = agent.CVaR(state, 0.0).item()
        aleatoric = agent.aleatoric(state).item()
        #epistemic = agent.epistemic(state).item()

        if('v3' in agent_id):
            surrogate = agent.inst_reward_est(state).item()
            SURR.append(1.0*surrogate-0.0*aleatoric+0.0*cvar)

        CVAR.append(cvar)
        ALEA.append(aleatoric)
        #EPIST.append(epistemic)

    '''plt.figure()
    #cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=np.min(np.array(CVAR)), vmax=np.max(np.array(CVAR) ))
    cmap = "jet"
    #z = cmap(norm(CVAR_array[:,2]))
    z = np.array(CVAR)
    plt.scatter(np.array(POS), np.array(VEL), c = z, norm = norm, cmap = cmap )
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('CVAR')
    plt.show(block=True)
    #plt.savefig('Aleatoric.png')'''
    
    plt.figure()
    #cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=np.min(np.array(ALEA)), vmax=np.max(np.array(ALEA)) )
    cmap = "jet"
    #z = cmap(norm(CVAR_array[:,2]))
    z = np.array(ALEA)
    plt.scatter(np.array(POS), np.array(VEL), c = z, norm = norm, cmap = cmap )
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Uncertainty from estimator distribution')
    #plt.show(block=True)
    plt.savefig('alea.eps')

    if('v3' in agent_id):
        plt.figure()
        #cmap = plt.cm.Spectral
        norm = plt.Normalize(vmin=np.min(np.array(SURR)), vmax=np.max(np.array(SURR)) )
        cmap = "jet"
        #z = cmap(norm(CVAR_array[:,2]))
        z = np.array(SURR)
        plt.scatter(np.array(POS), np.array(VEL), c = z, norm = norm, cmap = cmap )
        plt.colorbar()
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Uncertainty from reward surrogate distribution')
        #plt.show(block=True)
        plt.savefig('surr.eps')

    
    '''plt.figure()
    #cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=np.min(np.array(EPIST)), vmax=np.max(np.array(EPIST)) )
    cmap = "jet"
    #z = cmap(norm(CVAR_array[:,2]))
    z = np.array(EPIST)
    plt.scatter(np.array(POS), np.array(VEL), c = z, norm = norm, cmap = cmap )
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Epistemic Uncty')
    plt.show(block=True) '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=40)
    #parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasTD3-v0')
    #parser.add_argument("--env", help="Environment used for RL", type=str, default='Pendulum-v1')
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='TorchQR_DDPG_ensemble-v3')
    parser.add_argument("--env", help="Environment used for RL", type=str, default='car_1D-v0')
    parser.add_argument("--trial", help="Trial num of algo run", type=int, default=0)


    # Get input arguments
    args = parser.parse_args()
    nepisodes = args.nepisodes
    agent_id = args.agent
    env_id = args.env
    trial = args.trial

    # Print input settings
    run_opt(nepisodes,agent_id,env_id, trial)

