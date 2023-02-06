import os
import random
from datetime import datetime
import numpy as np


class OUActionNoise:
    def __init__(self, mean=0.0, std_deviation=0.3, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = None
        self.reset()

    def __call__(self):
        # random.seed(datetime.now())
        # random_data = os.urandom(4)
        # np.random.seed(int.from_bytes(random_data, byteorder="big"))
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class GreedyGaussianNoise:
    def __init__(self, action_dim, noise=0.1, epsilon_decay=0.99999, min_epsilon=0.01, max_epsilon=1.0) -> object:
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
        self.epsilon_decay = epsilon_decay
        self.noise = noise

    def __call__(self, lower_bound, upper_bound):
        epsilon = max(self.min_epsilon, self.epsilon)
        if random.random() < self.epsilon:
            noise = np.random.normal(0, self.noise, self.action_dim)
            self.epsilon = self.epsilon_decay * self.epsilon
            return noise * (upper_bound-lower_bound)/2.0#*self.epsilon
        else:
            return 0


class GreedyNoise:
    def __init__(self, action_dim, epsilon_decay=0.99999, min_epsilon=0.01, max_epsilon=1.0):
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
        self.epsilon_decay = epsilon_decay

    def __call__(self, lower_bound, upper_bound):
        epsilon = max(self.min_epsilon, self.epsilon)
        if random.random() < self.epsilon:
            noise = np.random.uniform(lower_bound, upper_bound, self.action_dim)
            self.epsilon = self.epsilon_decay * self.epsilon
            return noise
        else:
            return 0