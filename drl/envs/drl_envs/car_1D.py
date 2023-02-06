import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class rewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # can check for the specific environment (more generic)
        
    def step(self, action):

        '''
        Gym wrapper for Reward: 

        '''
        next_state, reward, done, info = self.env.step(action)
        
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
        self.noise = noise

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

        ind_xt_xg = 1 if((x <= self.xg + 0.1) and (x>= self.xg - 0.1)) else 0
        done = True if((ind_xt_xg==1) or (self.counter>=100)) else False

        if(done):
            print('DONE', 'x:', x, 'v:', v)

        costs = -10 + 350*(ind_xt_xg) 
        sel.counter+=1

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

