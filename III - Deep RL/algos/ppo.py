"""
Proximal Policy Optimization (PPO) algorithm
with Generalized Advantage Estimation (GAE)

TODOS:
-Migration to gymnasium==1.0.0

Notes:
-works with gymnasium==0.29.1
-this implementations supposes than environments terminate but not truncate

implementation details:
https://arxiv.org/abs/2005.12729
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

PPO check : 400 episodic return in breakout
"""

# TODO
# pistes d'améliorations:
# leanrl version (torch compile & cuda graphs)
# re comparer avec ppo_leanrl avec des HPs différents (adv norm, clip VF loss...)

import os
from dataclasses import dataclass
from typing import Optional
import string
from collections import deque
import time
import wandb
import tyro
import random
import numpy as np

import gymnasium as gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from misc import format_time

@dataclass
class Config:
    env_id: str = "CartPole-v1"

    seed: int = 3

    total_timesteps: int = 100_000
    """ total number of timesteps collected for the training """
    num_steps: int = 2_000
    """ number of steps per rollout (between updates) """
    num_envs: int = 1
    """ number of parallel envs used at each rollout """
    
    num_epochs: int = 10
    """ number of passes over each rollout experience (it is then discarded) """
    batch_size: int = 64
    """ batch size of each update """

    normalize_adv: bool = False
    """ whether or not to normalize the advantage in the policy update """

    clip_ratio: float = 0.2
    """ clipping ratio between old and new policy probs """
    clip_loss_vf: bool = False
    """ whether or not to apply a clipping strategy on the vf loss (done with pi_loss) """
    vf_coef: float = 0.5
    """ coefficient of the value function in the total loss """
    ent_coef: float = 0.
    """ coefficient of the policy entropy in the total loss """

    lr: float = 3e-4
    """ learning rate (policy and value function) """
    anneal_lr: bool = True
    """ whether or not to anneal the LR throughout training """

    max_kl: Optional[float] = None
    """ threshold for the KL div between old and new policy. Above this, the current update stops. """

    gae_gamma: float = 0.99
    """ gamma for advantage computation (technically, discount factor is 1)
        a good heuristic is knowing that the 1/(1-gamma) is the "sight" of the agent"""
    gae_lambda: float = 0.95
    """ lambda for advantage computation """

    max_grad_norm: float = 0.5
    """ max grad norm during training """

    device: str = "cpu"

    measure_burnin: int = 3
    """ number of steps to skip at the beginning before tracking speed """

    log_wandb: bool = False
    
    save_ckpt: bool = False
    """ whether or not to save the agent in a .pth file at the end of training """

    capture_video: bool = False
    """ whether or not to record interactions during training """
    capture_interval: int = 20_000
    """ number of total timesteps in between the recordings """
    capture_length: int = 0
    """ video length for each recording """

def make_env(idx):
    def thunk():
        if config.capture_video and idx == 0:
            env = gym.make(config.env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder=f"videos/{run_name}", name_prefix="training", 
                                           step_trigger=lambda i: i % (config.capture_interval//config.num_envs) == 0,
                                           video_length=config.capture_length, disable_logger=True)
        else:
            env = gym.make(config.env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = gym.wrappers.ClipAction(env) # only for Box action spaces
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=config.gae_gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()

        self.policy = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, envs.single_action_space.n), gain=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), gain=1.0)
        )

    def get_value(self, obs):
        return self.critic(obs)
    
    def get_action_value(self, obs, action=None):
        """
        Two modes:
        -rollout mode (action=None): an action is sampled (B=num_envs)
        -training mode (action!=None): logp(a|o) and entropy is given (B=num_steps*num_envs)

        obs: (B, obs_dim)
        action: (B, action_dim)
        > action: (B, action_dim)
        > logp: (B)
        """

        logits = self.policy(obs)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
            logp = probs.log_prob(action)
            ent = None
        else:
            logp = probs.log_prob(action)
            ent = probs.entropy()
            action = None
        
        return action, logp, ent, self.critic(obs)

def rollout(next_obs=None, next_done=None, avg_returns=None, avg_lengths=None):
    """
    collect num_steps timesteps of experience in the environment
    """

    b_observations = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(config.device)
    b_actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(config.device)
    b_logp = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    b_rewards = torch.zeros(config.num_steps, config.num_envs).to(config.device)
    b_dones = torch.zeros(config.num_steps, config.num_envs).to(config.device)
    b_values = torch.zeros(config.num_steps, config.num_envs).to(config.device)

    # next_obs  : (num_envs, obs_dim)
    # next_done : (num_envs,)
    # action    : (num_envs, action_dim) (with action_dim=() possible)
    # reward    : (num_envs,)

    if next_obs is None:
        next_obs, _ = envs.reset(seed=config.seed)
        next_obs = torch.Tensor(next_obs).to(config.device)
        next_done = torch.ones(config.num_envs).to(config.device)
        avg_returns = deque(maxlen=20)
        avg_lengths = deque(maxlen=20)

    for t in range(config.num_steps):
        b_observations[t] = next_obs
        b_dones[t] = next_done

        with torch.no_grad():
            action, logp, _, value = agent.get_action_value(next_obs)

        # env step (if done, next_obs is the obs of the new episode)
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminated, truncated)
        next_obs, next_done = torch.Tensor(next_obs).to(config.device), torch.Tensor(next_done).to(config.device)

        b_actions[t] = action
        b_logp[t] = logp
        b_rewards[t] = torch.tensor(reward).to(config.device)
        b_values[t] = value.flatten() # value was (num_envs, 1)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    r = float(info["episode"]["r"].reshape(()))
                    l = float(info["episode"]["l"].reshape(()))
                    avg_returns.append(r)
                    avg_lengths.append(l)

    # bootstrap final step
    with torch.no_grad():
        next_value = agent.get_value(next_obs)
        next_value = next_value.flatten()
    
    """
    GAE advantage computation : A_t = sum_{l=0} (gamma*lambda)^l delta_{t+l}
    (where delta_t = r_t + gamma V(s_{t+1}) - V(s_t))
    A nice way to compute it is to start from the end by first computing the last delta_T
    and then add to it delta_{T-1}, delta_{T-2} etc, but each time decaying the advantage by gamma*lambda
    """
    b_adv = torch.zeros(config.num_steps, config.num_envs).to(config.device)
    gae = 0

    next_value = next_value
    next_nonterminal = 1.0 - next_done
    for t in reversed(range(config.num_steps)):
        delta = b_rewards[t] + next_nonterminal * config.gae_gamma * next_value - b_values[t]
        gae = delta + next_nonterminal * config.gae_gamma * config.gae_lambda * gae
        b_adv[t] = gae

        next_value = b_values[t]
        next_nonterminal = 1.0 - b_dones[t]

    b_returns = b_adv + b_values # adv = q - v so adv + v = q (with q being a mix of n-step returns)

    # todo : view?
    # flatten parallel env data into a single stream
    b_observations = b_observations.reshape((-1,) + envs.single_observation_space.shape)
    b_actions = b_actions.reshape((-1,) + envs.single_action_space.shape)
    b_logp = b_logp.reshape(-1)
    b_adv = b_adv.reshape(-1)
    b_values = b_values.reshape(-1)
    b_returns = b_returns.reshape(-1)

    return next_obs, next_done, b_observations, b_actions, b_logp, b_adv, b_values, b_returns, avg_returns, avg_lengths

def update(obs, actions, old_logp, adv, old_values, rets):
    """
    obs: (B, obs_dim)
    actions: (B, action_dim)
    old_logp: (B,)
    adv: (B,)
    old_values: (B,)
    rets: (B,)

    all these are supposed to be collected in a rollout phase
    here, "old_" means that it has been computed by the previous policy (theta_old) (and there is no gradient attached to it)
    """

    #ent = []
    clipfracs = []
    #kls = []

    for _ in range(config.num_epochs):
        indices = torch.randperm(config.num_steps)
        for start in range(0, 1 * config.num_steps, config.batch_size):
            end = start + config.batch_size
            idx_batch = indices[start:end]

            b_obs = obs[idx_batch]
            b_actions = actions[idx_batch]
            b_old_logp = old_logp[idx_batch]
            b_rets = rets[idx_batch]
            b_adv = adv[idx_batch]
            b_old_values = old_values[idx_batch]
        
            _, b_logp, b_entropy, b_values = agent.get_action_value(b_obs, action=b_actions)

            if config.normalize_adv:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # policy loss
            log_ratio = b_logp - b_old_logp
            ratio = torch.exp(log_ratio)
            clip_adv = torch.clamp(ratio, 1-config.clip_ratio, 1+config.clip_ratio) * b_adv
            loss_pi = -torch.min(ratio * b_adv, clip_adv).mean()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean() # (http://joschu.net/blog/kl-approx.html)
                clipfracs.append(((ratio - 1.0).abs() > config.clip_ratio).float().mean().item())
            
            # value loss (0.5 is to ensure same vf_coef as with other implementations)
            b_values = b_values.view(-1)
            if config.clip_loss_vf:
                loss_vf_unclipped = (b_rets - b_values)**2

                b_values_clipped = b_old_values + torch.clamp(b_values - b_old_values, -config.clip_ratio, +config.clip_ratio)
                loss_vf_clipped = (b_rets - b_values_clipped)**2

                loss_vf = 0.5 * torch.max(loss_vf_unclipped, loss_vf_clipped).mean()
            else:
                loss_vf = 0.5 * ((b_rets - b_values)**2).mean()

            # entropy loss
            loss_ent = b_entropy.mean()
            
            # total update
            loss = loss_pi + config.vf_coef * loss_vf - config.ent_coef * loss_ent
            loss.backward()
            _ = nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optim.step()
            optim.zero_grad()

            #ent.append(loss_ent.detach()) # TODO warning with torch.compile/cuda graphs
            #kls.append(approx_kl) # TODO warning with torch.compile/cuda graphs

        if config.max_kl is not None and approx_kl > config.max_kl:
            break
            
    y_pred, y_true = old_values.cpu().numpy(), rets.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return explained_var, np.mean([0]), clipfracs

if __name__ == "__main__":
    config = tyro.cli(Config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    
    if config.log_wandb:
        wandb.init(project="ppo", config={"algo": "ppo", **vars(config)},)
        run_name = wandb.run.name
    else:
        run_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))

    save_dir = os.path.join("runs/", run_name)
    os.makedirs(save_dir, exist_ok=True)

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(config.num_envs)])

    agent = Agent(envs).to(config.device)
    optim = torch.optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    print(f"Run {run_name} starting.")
    
    total_steps = config.total_timesteps // (config.num_steps*config.num_envs)

    next_obs, next_done = None, None
    avg_returns, avg_lengths = None, None

    start_time = time.time()

    for step in range(total_steps):
        t0 = time.time()

        next_obs, next_done, obs, actions, old_logp, adv, old_values, rets, avg_returns, avg_lengths = rollout(next_obs, next_done, avg_returns, avg_lengths)
        explained_var, mean_kl, clipfracs = update(obs, actions, old_logp, adv, old_values, rets)

        t1 = time.time()

        # lr annealing
        if config.anneal_lr:
            frac = 1.0 - ((step+1) / total_steps)
            lr = frac * config.lr
            optim.param_groups[0]["lr"] = lr

        # printing and logging
        uptime = time.time() - start_time
        total_time = ((total_steps*config.num_steps) * uptime) / ((step+1) * config.num_steps)
        eta = total_time - uptime

        timesteps_per_s = config.num_steps*config.num_envs / (t1-t0)

        num_digits = len(str(total_steps))
        formatted_iter = f"{step+1:0{num_digits}d}"
        print(f"Step: {formatted_iter}/{total_steps}. Avg episode return: {np.mean(avg_returns):.2f}. Avg episode length: {np.mean(avg_lengths):.2f}. Timesteps/s: {timesteps_per_s:.0f}. ETA: {format_time(eta)}.")

        if config.log_wandb:
            wandb.log({"returns": np.mean(avg_returns), "lengths": np.mean(avg_lengths),
                       "returns_std": np.std(avg_returns), "lengths_std": np.std(avg_lengths),
                       "explained_var": explained_var, "mean_kl": mean_kl, "clipfracs": np.mean(clipfracs),
                       "lr": optim.param_groups[0]["lr"],
                       "timesteps_per_s": timesteps_per_s},
                       step=(step+1)*config.num_steps*config.num_envs)
    
    envs.close()

    if config.save_ckpt:
        torch.save({"model": agent.state_dict(), "config": vars(config)}, os.path.join(save_dir, "agent.pth"))
