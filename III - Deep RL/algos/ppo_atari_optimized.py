"""
Proximal Policy Optimization (PPO) algorithm
with Generalized Advantage Estimation (GAE)

implementation details:
https://arxiv.org/abs/2005.12729
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

PPO check : 400 episodic return in breakout
"""

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

import envpool
import gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical, Distribution

import tensordict
from tensordict import from_module
from tensordict.nn import CudaGraphModule

from misc import format_time

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")

@dataclass
class Config:
    env_id: str = "Breakout-v5"

    seed: int = 2

    total_timesteps: int = 10_000_000
    """ total number of timesteps collected for the training """
    num_steps: int = 128
    """ number of steps per rollout (between updates) """
    num_envs: int = 8
    """ number of parallel envs used at each rollout """
    
    num_epochs: int = 4
    """ number of passes over each rollout experience (it is then discarded) """
    batch_size: int = 256
    """ batch size of each update """

    normalize_adv: bool = True
    """ whether or not to normalize the advantage in the policy update """

    clip_ratio: float = 0.1
    """ clipping ratio between old and new policy probs """
    clip_loss_vf: bool = True
    """ whether or not to apply a clipping strategy on the vf loss (done with pi_loss) """
    vf_coef: float = 0.5
    """ coefficient of the value function in the total loss """
    ent_coef: float = 0.01
    """ coefficient of the policy entropy in the total loss """

    lr: float = 2.5e-4
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

    device: str = "cuda"
    compile: bool = False
    cudagraphs: bool = True

    log_wandb: bool = False
    
    ckpt_interval: int = 2_000_000
    """ interval between two checkpoints (in number of timesteps) """
    save_ckpt: bool = False
    """ whether or not to save the agent in a .pth file at the end of training (independent of ckpt_interval) """

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, device=None):
        super().__init__()

        self.core = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, device=device)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512, device=device)),
            nn.ReLU(),
        )
        self.policy = layer_init(nn.Linear(512, envs.single_action_space.n, device=device), gain=0.01)
        self.critic = layer_init(nn.Linear(512, 1, device=device), gain=1.0)

    def get_value(self, obs):
        return self.critic(self.core(obs / 255.))
    
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

        hidden = self.core(obs / 255.)

        logits = self.policy(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
            logp = probs.log_prob(action)
            ent = None
        else:
            logp = probs.log_prob(action)
            ent = probs.entropy()
            action = None
        
        return action, logp, ent, self.critic(hidden)

def rollout(obs=None, done=None, avg_returns=None, avg_lengths=None, max_ret=0):
    """
    collect num_steps timesteps of experience in the environment
    """

    ts = []
    #b_observations = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(config.device)
    #b_actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(config.device)
    #b_logp = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    #b_rewards = torch.zeros(config.num_steps, config.num_envs).to(config.device)
    #b_dones = torch.zeros(config.num_steps, config.num_envs).to(config.device)
    #b_values = torch.zeros(config.num_steps, config.num_envs).to(config.device)

    # next_obs  : (num_envs, obs_dim)
    # next_done : (num_envs,)
    # action    : (num_envs, action_dim) (with action_dim=() possible)
    # reward    : (num_envs,)

    if obs is None:
        obs = torch.tensor(envs.reset(), device=config.device, dtype=torch.uint8)
        done = torch.ones(config.num_envs, device=config.device, dtype=torch.bool)
        avg_returns = deque(maxlen=20)
        avg_lengths = deque(maxlen=20)

    for _ in range(config.num_steps):
        action, logp, _, value = policy(obs)

        # env step (if done, next_obs is the obs of the new episode)
        next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
        next_obs, reward, next_done = torch.as_tensor(next_obs), torch.as_tensor(reward), torch.as_tensor(next_done)

        for idx, d in enumerate(next_done):
            if d and infos["lives"][idx] == 0:
                r = float(infos["r"][idx])
                max_ret = max(r, max_ret)
                l = float(infos["l"][idx])
                avg_returns.append(r)
                avg_lengths.append(l)
    
        ts.append(tensordict.TensorDict._new_unsafe(
            b_obs=obs,
            b_actions=action,
            dones=done,
            b_old_logp=logp,
            rewards=reward,
            b_old_values=value.flatten(),
            batch_size=(config.num_envs,),
        ))

        obs = next_obs = next_obs.to(config.device, non_blocking=True)
        done = next_done.to(config.device, non_blocking=True)
    
    container = torch.stack(ts, 0).to(config.device)
    return next_obs, done, container, avg_returns, avg_lengths, max_ret

def gae(next_obs, next_done, container):
    """
    GAE advantage computation : A_t = sum_{l=0} (gamma*lambda)^l delta_{t+l}
    (where delta_t = r_t + gamma V(s_{t+1}) - V(s_t))
    A nice way to compute it is to start from the end by first computing the last delta_T
    and then add to it delta_{T-1}, delta_{T-2} etc, but each time decaying the advantage by gamma*lambda
    """

    # bootstrap final step
    next_value = get_value(next_obs).flatten()

    b_nextnonterminals = (~container['dones']).float().unbind(0) # tuple of num_steps (num_envs,) tensors
    b_values = container['b_old_values'] # (num_steps, num_envs)
    b_values_unbind = b_values.unbind(0) # tuple of num_steps (num_envs,) tensors
    b_rewards = container['rewards'].unbind(0) # tuple of num_steps (num_envs,) tensors

    b_adv = []
    gae = 0

    next_value = next_value
    next_nonterminal = (~next_done).float()
    for t in reversed(range(config.num_steps)):
        delta = b_rewards[t] + next_nonterminal * config.gae_gamma * next_value - b_values_unbind[t]
        gae = delta + next_nonterminal * config.gae_gamma * config.gae_lambda * gae
        b_adv.append(gae)

        next_value = b_values_unbind[t]
        next_nonterminal = b_nextnonterminals[t]
        
    b_adv = container['b_adv'] = torch.stack(list(reversed(b_adv)))
    container['b_rets'] = b_adv + b_values # adv = q - v so adv + v = q (with q being a mix of n-step returns)

    return container

def update(b_obs, b_actions, b_old_logp, b_adv, b_old_values, b_rets):
    """
    perform one step of update on both the actor and the critic

    b_obs: (B, obs_dim)
    b_actions: (B, action_dim)
    b_old_logp: (B,)
    b_adv: (B,)
    b_old_values: (B,)
    b_rets: (B,)

    here B is the batch size.
    this should be called inside a double for loop (over epochs and batches)
    """

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
        clipfrac = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
            
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

    return approx_kl, clipfrac, loss_ent.detach()

update = tensordict.nn.TensorDictModule(
    update,
    in_keys=['b_obs', 'b_actions', 'b_old_logp', 'b_adv', 'b_old_values', 'b_rets'],
    out_keys=['approx_kl', 'clipfrac', 'loss_ent']
)

if __name__ == "__main__":
    config = tyro.cli(Config)

    seed = 123456789 + config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    if config.log_wandb:
        wandb.init(project="ppo_atari", config={**vars(config)},)
        run_name = wandb.run.name
    else:
        run_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))

    save_dir = os.path.join("runs/", run_name)
    os.makedirs(save_dir, exist_ok=True)

    envs = envpool.make(config.env_id,
                        env_type="gym",
                        num_envs=config.num_envs,
                        episodic_life=True,
                        reward_clip=True,
                        seed=config.seed)
    envs.num_envs = config.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, device=config.device)

    agent_inference = Agent(envs, device=config.device)
    agent_inference_params = from_module(agent).data
    agent_inference_params.to_module(agent_inference)

    optim = torch.optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5, capturable=config.cudagraphs and not config.compile)

    # define executables
    policy = agent_inference.get_action_value
    get_value = agent_inference.get_value

    if config.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)
    
    if config.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    print(f"Run {run_name} starting.")
    
    total_steps = config.total_timesteps // (config.num_steps*config.num_envs)

    next_obs, next_done = None, None
    avg_returns, avg_lengths = None, None

    start_time = time.time()

    max_ret = -999999

    for step in range(total_steps):
        t0 = time.time()

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container, avg_returns, avg_lengths, max_ret = rollout(next_obs, next_done, avg_returns, avg_lengths, max_ret)

        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1) # flatten num_steps and num_envs

        clipfracs = []
        ents = []
        kls = []

        for _ in range(config.num_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=config.device).split(config.batch_size)
            for b in b_inds:
                container_local = container_flat[b]
                out = update(container_flat, tensordict_out=tensordict.TensorDict())

                clipfracs.append(out['clipfrac'])
                ents.append(out['loss_ent'])
                kls.append(out['approx_kl'])
            
            if config.max_kl is not None and out['approx_kl'] > config.max_kl:
                break
        
        y_pred, y_true = container_flat['b_old_values'].cpu().numpy(), container_flat['b_rets'].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        t1 = time.time()

        # lr annealing
        if config.anneal_lr:
            frac = 1.0 - ((step+1) / total_steps)
            lr = frac * config.lr
            optim.param_groups[0]["lr"] = lr

        # printing and logging
        uptime = time.time() - start_time
        total_time = ((total_steps*config.num_steps*config.num_envs) * uptime) / ((step+1) * config.num_steps*config.num_envs)
        eta = total_time - uptime

        timesteps_per_s = config.num_steps*config.num_envs / (t1-t0)

        num_digits = len(str(total_steps))
        formatted_iter = f"{step+1:0{num_digits}d}"
        print(f"Step: {formatted_iter}/{total_steps}. Avg episode return: {np.mean(avg_returns):.2f}. Max episode return: {max_ret}. Avg episode length: {np.mean(avg_lengths):.2f}. Timesteps/s: {timesteps_per_s:.0f}. ETA: {format_time(eta)}.")

        if config.log_wandb:
            wandb.log({"returns": np.mean(avg_returns), "lengths": np.mean(avg_lengths),
                       "returns_std": np.std(avg_returns), "lengths_std": np.std(avg_lengths),
                       "explained_var": explained_var, "mean_kl": np.mean(kls), "clipfracs": np.mean(clipfracs), "entropy": np.mean(ents),
                       "lr": optim.param_groups[0]["lr"],
                       "timesteps_per_s": timesteps_per_s},
                       step=(step+1)*config.num_steps*config.num_envs)
            
        if ((step+1)*config.num_steps*config.num_envs % config.ckpt_interval) == 0:
            dirname = f"ckpt_{iter}/"
            os.makedirs(os.path.join(save_dir, dirname), exist_ok=True)
            checkpoint = {"model": agent_inference.state_dict(), "config": vars(config)}
            torch.save(checkpoint, os.path.join(save_dir, dirname, "model.pth"))
    
    envs.close()

    if config.save_ckpt:
        torch.save({"model": agent.state_dict(), "config": vars(config)}, os.path.join(save_dir, "agent.pth"))
        # todo : agent inference
