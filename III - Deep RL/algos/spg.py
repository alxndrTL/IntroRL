"""
Simple Policy Gradient algorithm

only works with 1 env
"""

from dataclasses import dataclass
import tyro
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

    pi_lr: float = 1e-2
    """ learning rate of the policy """

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
    done = torch.ones(1)

    for t in range(config.num_steps):
        b_observations[t] = next_obs
        b_dones[t] = done

        with torch.no_grad():
            action, _ = agent.get_action(next_obs)

        # env step
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        next_done = np.logical_or(terminated, truncated)
        next_obs, next_done = torch.Tensor(next_obs).to(config.device), torch.Tensor(next_done).to(config.device)

        b_actions[t] = action
        b_rewards[t] = torch.tensor(reward).to(config.device)

    b_returns = torch.zeros(config.num_steps)
    ep_returns = [] # for logging, one per traj
    curr_ret = 0
    last_ep_idx = config.num_steps-1
    for t in reversed(range(config.num_steps)):
        curr_ret += b_rewards[t]

        if b_dones[t]:
            b_returns[t:last_ep_idx+1] = curr_ret
            ep_returns.append(curr_ret)

            curr_ret = 0
            last_ep_idx = t-1

    return b_observations, b_actions, b_returns, ep_returns

def update(obs, actions, returns):
    """
    obs: (B, obs_dim)
    actions: (B, action_dim)
    returns: (B,)
    """

    _, logp = agent.get_action(obs, action=actions)
    
    loss_pi = -torch.mean(logp * returns) # technically not correct, the mean is over the traj only, not timesteps
    loss_pi.backward()
    optim.step()
    optim.zero_grad()

if __name__ == "__main__":
    config = Config(env_id='CartPole-v1', total_timesteps=100_000, num_steps=500, pi_lr=2**(-6), log_wandb=True, device="cpu")
    
    if config.log_wandb:
        wandb.init(project="spg", config={
            "algo": "spg",
            "env_id": config.env_id,
            "total_timesteps": config.total_timesteps,
            "step_per_rollout": config.num_steps,
            "pi_lr": config.pi_lr
        })

    env = gym.vector.SyncVectorEnv([lambda: gym.make(config.env_id)])

    agent = Agent(env)
    optim = torch.optim.Adam(agent.parameters(), lr=config.pi_lr)

    total_steps = config.total_timesteps // config.num_steps

    for step in range(total_steps):
        obs, actions, returns, ep_rets = rollout()
        update(obs, actions, returns)

        num_digits = len(str(total_steps))
        formatted_iter = f"{step:0{num_digits}d}"
        print(f"Step: {formatted_iter}/{total_steps}. Avg return: {np.mean(ep_rets):.2f}")

        if config.log_wandb:
            wandb.log({"returns": np.mean(ep_rets)}, step=(step+1)*config.num_steps)
