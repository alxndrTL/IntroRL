"""
Dummy env to check that your computation of returns, rtg, gae is correct.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DebugEnv(gym.Env):
    def __init__(self):
        super(DebugEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1000)
        self.state = 0
        self.done = False

    def reset(self, seed=None, options=None):
        self.state = 0
        self.done = False
        return self.state, {}

    def step(self, action):
        self.state += 1
        reward = np.random.choice([0, 1])
        
        if np.random.rand() > 0.9 and self.state>=5:
            self.done = True
            reward = 10

        if self.state==1:
            reward = 2
        
        # obs, reward, done, truncated, info
        return self.state, reward, self.done, False, {}

    def render(self):
        print(f"State: {self.state}")

gym.envs.registration.register(
    id='DebugEnv-v0',
    entry_point=DebugEnv,
)
