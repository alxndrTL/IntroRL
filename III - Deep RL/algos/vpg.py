"""
Vanilla Policy Gradient algorithm
with Generalized Advantage Estimation (GAE)

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

    gae_gamma: float = 0.99
    """ gamma for advantage computation (technically, discount factor is 1) """
    gae_lambda: float = 0.95
    """ lambda for advantage computation """

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
    
    def get_action_value(self, obs, action=None):
        """
        Two modes:
        -rollout mode (action=None): an action is sampled (B=num_envs)
        -training mode (action!=None): logp(a|o) is given (B=num_steps*num_envs)

        obs: (B, obs_dim)
        action: (B, action_dim)
        > action: (B, action_dim)
        > logprobs: (B)
        """

        logits = self.policy(obs)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
            logp = None
        else:
            logp = probs.log_prob(action)
            action = None
        
        return action, logp, self.value_func(obs)

def rollout():
    """
    collect num_steps timesteps of experience in the environment
    """

    b_observations = torch.zeros((config.num_steps,) + env.single_observation_space.shape).to(config.device)
    b_actions = torch.zeros((config.num_steps,) + env.single_action_space.shape).to(config.device)
    b_rewards = torch.zeros(config.num_steps).to(config.device)
    b_dones = torch.zeros(config.num_steps).to(config.device)
    b_values = torch.zeros(config.num_steps).to(config.device)

    # next_obs  : (1, obs_dim)
    # next_done : (1,)
    # action    : (1, action_dim)
    # reward    : (1,)

    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(config.device)
    next_done = torch.ones(1)

    for t in range(config.num_steps):
        b_observations[t] = next_obs
        b_dones[t] = next_done

        with torch.no_grad():
            action, _, value = agent.get_action_value(next_obs)

        # env step (if done, next_obs is the obs of the new episode)
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        next_done = np.logical_or(terminated, truncated)
        next_obs, next_done = torch.Tensor(next_obs).to(config.device), torch.Tensor(next_done).to(config.device)

        b_actions[t] = action
        b_rewards[t] = torch.tensor(reward).to(config.device)
        b_values[t] = value

    with torch.no_grad():
        next_value = agent.get_value(next_obs)
    
    """
    GAE advantage computation : A_t = sum_{l=0} (gamma*lambda)^l delta_{t+l}
    (where delta_t = r_t + gamma V(s_{t+1}) - V(s_t))
    A nice way to compute it is to start from the end by first computing the last delta_T
    and then add to it delta_{T-1}, delta_{T-2} etc, but each time decaying the advantage by gamma*lambda
    """
    b_adv = torch.zeros(config.num_steps)
    gae = 0
    returns = [] # for logging, one per traj
    ret = 0

    next_value = next_value
    next_nonterminal = 1.0 - next_done
    for t in reversed(range(config.num_steps)):
        delta = b_rewards[t] + next_nonterminal * config.gae_gamma * next_value - b_values[t]
        gae = delta + next_nonterminal * config.gae_gamma * config.gae_lambda * gae
        b_adv[t] = gae

        ret += b_rewards[t]
        if b_dones[t]:
            returns.append(ret)
            ret = 0

        next_value = b_values[t]
        next_nonterminal = 1.0 - b_dones[t]

    b_returns = b_adv + b_values # adv = q - v so adv + v = q (with q being a mix of n-step returns)

    return b_observations, b_actions, b_adv, b_returns, returns

def update(obs, actions, adv, rets):
    """
    obs: (B, obs_dim)
    actions: (B, action_dim)
    adv: (B,)
    q: (B,)
    """

    _, logp, values = agent.get_action_value(obs, action=actions)
    
    # policy loss
    loss_pi = -torch.mean(logp * adv)

    # value 
    loss_vf = torch.mean((rets - values)**2)

    # different choices here.
    # could have gone with two different LRs
    # could also have done multiple update on loss_vf

    # total update
    loss = loss_pi + config.vf_coef * loss_vf
    loss.backward()
    optim.step()
    optim.zero_grad()

if __name__ == "__main__":
    # 2_000/2^-5, 500/2^-6
    config = Config(env_id='CartPole-v1', total_timesteps=100_000, num_steps=2_000,
                    lr=2**(-5), vf_coef=0.5, gae_gamma=0.99, gae_lambda=0.97,
                    log_wandb=False, device="cpu")
    
    if config.log_wandb:
        wandb.init(project="vpg", config={
            "algo": "vpg",
            "env_id": config.env_id,
            "total_timesteps": config.total_timesteps,
            "step_per_rollout": config.num_steps,
            "lr": config.lr,
            "vf_coef": config.vf_coef,
            "gae_gamma": config.gae_gamma,
            "gae_lambda": config.gae_lambda,
        })

    env = gym.vector.SyncVectorEnv([lambda: gym.make(config.env_id)])

    agent = Agent(env)
    optim = torch.optim.Adam(agent.parameters(), lr=config.lr)
    
    total_steps = config.total_timesteps // config.num_steps

    for step in range(total_steps):
        obs, actions, adv, rets, ep_rets = rollout()
        update(obs, actions, adv, rets)

        num_digits = len(str(total_steps))
        formatted_iter = f"{step:0{num_digits}d}"
        print(f"Step: {formatted_iter}/{total_steps}. Avg return: {np.mean(ep_rets):.2f}")

        if config.log_wandb:
            wandb.log({"returns": np.mean(ep_rets)}, step=(step+1)*config.num_steps)
