"""
REINFORCE with baseline algorithm

only works with 1 env
"""

from dataclasses import dataclass
import wandb
import numpy as np

import gymnasium as gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

@dataclass
class Config:
    env_id: str = "CartPole-v1"

    total_timesteps: int = 250_000
    """ total number of timesteps collected for the training """
    num_steps: int = 5_000
    """ number of steps per rollout (between updates) """

    lr: float = 3e-4
    """ learning rate (policy and value function) """

    vf_coef: float = 0.5
    """ coefficient of the value function in the total loss """

    gamma: float = 0.99
    """ discount factor """

    log_wandb: bool = False

    device: str = "cpu"

class Agent(nn.Module):
    def __init__(self, env: gym.vector.SyncVectorEnv):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 32),
            nn.Tanh(),
            nn.Linear(32, env.single_action_space.n)
        )

        self.value_func = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def get_value(self, obs):
        return self.value_func(obs)
    
    def get_action(self, obs, action=None):
        """
        Two modes:
        -rollout mode (action=None): an action is sampled (B=num_envs)
        -training mode (action!=None): logp(a|o) is given (B=num_steps*num_envs)

        obs: (B, obs_dim)
        action: (B, action_dim)
        > action: (B, action_dim)
        > logp: (B)
        """

        logits = self.policy(obs)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
            logp = None
        else:
            logp = probs.log_prob(action)
            action = None
        
        return action, logp

def rollout():
    """
    collect num_steps timesteps of experience in the environment
    """

    b_observations = torch.zeros((config.num_steps,) + env.single_observation_space.shape).to(config.device)
    b_actions = torch.zeros((config.num_steps,) + env.single_action_space.shape).to(config.device)
    b_rewards = torch.zeros(config.num_steps).to(config.device)
    b_dones = torch.zeros(config.num_steps).to(config.device)

    # observation : (1, obs_dim)
    # action : (1, action_dim)
    # reward : (1,)
    # done : (1,)

    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(config.device)
    next_done = torch.ones(1)

    for t in range(config.num_steps):
        b_observations[t] = next_obs
        b_dones[t] = next_done

        with torch.no_grad():
            action, _ = agent.get_action(next_obs)

        # env step
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        next_done = np.logical_or(terminated, truncated)
        next_obs, next_done = torch.Tensor(next_obs).to(config.device), torch.Tensor(next_done).to(config.device)

        b_actions[t] = action
        b_rewards[t] = torch.tensor(reward).to(config.device)

    b_rtg = torch.zeros(config.num_steps)
    rtg = 0
    returns = [] # for logging, one per traj
    ret = 0
    for t in reversed(range(config.num_steps)):
        rtg += b_rewards[t]
        ret += b_rewards[t]
        b_rtg[t] = rtg
        rtg *= config.gamma

        if b_dones[t]:
            returns.append(ret)
            rtg = 0
            ret = 0

    return b_observations, b_actions, b_rtg, returns

def update(obs, actions, rtg):
    """
    obs: (B, obs_dim)
    actions: (B, action_dim)
    rtg: (B,)
    """

    values = agent.get_value(obs)
    _, logp = agent.get_action(obs, action=actions)
    
    # policy loss
    loss_pi = -torch.mean(logp * (rtg - values))

    # value 
    loss_vf = torch.mean((rtg - values)**2)

    # different choices here.
    # could have gone with two different LRs
    # could also have done multiple update on loss_vf

    # total update
    loss = loss_pi + config.vf_coef * loss_vf
    loss.backward()
    optim.step()
    optim.zero_grad()

if __name__ == "__main__":
    config = Config(env_id='CartPole-v1', total_timesteps=100_000, num_steps=500, lr=2**(-6), vf_coef=0.5, gamma=0.99, log_wandb=True, device="cpu")
    
    if config.log_wandb:
        wandb.init(project="reinforce", config={
            "algo": "reinforce_baseline",
            "env_id": config.env_id,
            "total_timesteps": config.total_timesteps,
            "step_per_rollout": config.num_steps,
            "lr": config.lr,
            "vf_coef": config.vf_coef,
            "gamma": config.gamma
        })

    env = gym.vector.SyncVectorEnv([lambda: gym.make(config.env_id)])

    agent = Agent(env)
    optim = torch.optim.Adam(agent.parameters(), lr=config.lr)
    
    total_steps = config.total_timesteps // config.num_steps

    for step in range(total_steps):
        obs, actions, rtg, rets = rollout()
        update(obs, actions, rtg)

        num_digits = len(str(total_steps))
        formatted_iter = f"{step:0{num_digits}d}"
        print(f"Step: {formatted_iter}/{total_steps}. Avg return: {np.mean(rets):.2f}")

        if config.log_wandb:
            wandb.log({"returns": np.mean(rets)}, step=(step+1)*config.num_steps)
