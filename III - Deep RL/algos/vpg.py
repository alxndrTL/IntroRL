from dataclasses import dataclass
import tyro

import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import gymnasium as gym

# todo : multi env (shapes)


"""
env = gym.make('CartPole-v1', render_mode='human')

observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        observation, info = env.reset()

env.close()
"""

@dataclass
class Config:
    env_id: str = "CartPole-v1"

    total_timesteps: int = 250_000
    """ total number of timesteps collected for the training """
    num_steps: int = 5_000
    """ number of steps per rollout (between updates) """

    pi_lr: float = 1e-2


class Agent(nn.Module):
    def __init__(self, env: gym.vector.SyncVectorEnv):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 32),
            nn.Tanh(),
            nn.Linear(32, env.single_action_space.n)
        )

    def get_action(self, obs):
        # x: (obs_dim_flatten,)
        # > action: ()
        # > logprobs: ()
        logits = self.policy(obs)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)

config = Config(env_id='CartPole-v1', total_timesteps=250_000, num_steps=5_000, pi_lr=1e-2)
device = "cpu"

env = gym.make(config.env_id)
agent = Agent(env)


# def rollout(), def update() puis le corps

def rollout():
    batch_obs = torch.zeros((config.num_steps) + env.observation_space.shape).to(device) # not used
    batch_logprobs = torch.zeros(config.num_steps).to(device)
    batch_rewards = torch.zeros(config.num_steps).to(device)
    batch_dones = torch.zeros(config.num_steps).to(device)

    observation, _ = env.reset()
    observation = torch.Tensor(observation).to(device)
    done = torch.ones(1)

    for t in range(config.num_steps):
        batch_obs[t] = observation
        batch_dones[t] = done

        with torch.no_grad():
            action, logprob = agent.get_action(observation)

        # env step
        observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        observation, done = torch.Tensor(observation).to(device), torch.Tensor(done).to(device)

        batch_logprobs[t] = logprob
        batch_rewards[t] = reward

    # compute return
    batch_returns = torch.zeros(config.num_steps)
    curr_ret = 0
    last_ep_idx = config.num_steps-1
    for t in reversed(range(config.num_steps)):
        curr_ret += batch_rewards[t]

        if t < config.num_steps-1 and batch_dones[t+1]:
            batch_returns[t:last_ep_idx] = curr_ret # +- 1
            curr_ret = 0
            last_ep_idx = t # +- 1












