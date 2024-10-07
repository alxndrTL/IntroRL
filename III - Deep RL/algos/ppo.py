"""
Proximal Policy Optimization (PPO) algorithm
with Generalized Advantage Estimation (GAE)

only works with 1 env
"""

# TODO
# pistes d'améliorations:
# multi envs?
# script de test d'un agent entraîné (pas forcément en rendering, paramètre)
# mettre en place pipeline de test (avec notamment tyro pour lancer depuis la cmd line) et reflechir a comment mettre en place cela (wandb? etc) (commun à tous les algos!!)
# leanrl version (torch compile & cuda graphs)

# reprendre wandb typo de leanRL

# lire papier implementations details of ppo + intégrer notes lec6

# re comparer avec ppo_leanrl avec des HPs différents (adv norm, clip VF loss...)

from dataclasses import dataclass
import wandb
import random
import numpy as np

import gymnasium as gym

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

@dataclass
class Config:
    env_id: str = "CartPole-v1"

    seed: int = 3

    # todo : remettre valeurs : total_timesteps : 250_000, num_steps : 5_000
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
    anneal_lr: bool = False
    """ whether or not to anneal the LR throughout training """

    max_kl: float = None
    """ threshold for the KL div between old and new policy. Above this, the current update stops. """

    gae_gamma: float = 0.99
    """ gamma for advantage computation (technically, discount factor is 1) """
    gae_lambda: float = 0.95
    """ lambda for advantage computation """

    max_grad_norm: float = 0.5
    """ max grad norm during training """

    log_wandb: bool = True

    device: str = "cpu"

def make_env(env_id, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = gym.wrappers.ClipAction(env) # only for Box action spaces
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

class Agent(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 32),
            nn.Tanh(),
            nn.Linear(32, envs.single_action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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

def rollout():
    """
    collect num_steps timesteps of experience in the environment
    """

    b_observations = torch.zeros((config.num_steps,) + envs.single_observation_space.shape).to(config.device)
    b_actions = torch.zeros((config.num_steps,) + envs.single_action_space.shape).to(config.device)
    b_logp = torch.zeros((config.num_steps,)).to(config.device)
    b_rewards = torch.zeros(config.num_steps).to(config.device)
    b_dones = torch.zeros(config.num_steps).to(config.device)
    b_values = torch.zeros(config.num_steps).to(config.device)
    returns = []

    # next_obs  : (1, obs_dim)
    # next_done : (1,)
    # action    : (1, action_dim)
    # reward    : (1,)

    seed = config.seed if step==0 else None

    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(config.device)
    next_done = torch.ones(1)

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
        b_values[t] = value

        if "final_info" in infos:
            for info in infos["final_info"]:
                r = float(info["episode"]["r"].reshape(()))
                returns.append(r)

    # bootstrap final step
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

    next_value = next_value
    next_nonterminal = 1.0 - next_done
    for t in reversed(range(config.num_steps)):
        delta = b_rewards[t] + next_nonterminal * config.gae_gamma * next_value - b_values[t]
        gae = delta + next_nonterminal * config.gae_gamma * config.gae_lambda * gae
        b_adv[t] = gae

        next_value = b_values[t]
        next_nonterminal = 1.0 - b_dones[t]

    b_returns = b_adv + b_values # adv = q - v so adv + v = q (with q being a mix of n-step returns)

    return b_observations, b_actions, b_logp, b_adv, b_values, b_returns, returns

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
            loss_pi = -torch.mean(torch.min(ratio * b_adv, clip_adv))

            # KL computations (http://joschu.net/blog/kl-approx.html)
            with torch.no_grad():
                approx_kl = torch.mean(((ratio - 1) - log_ratio))
            
            # value loss (0.5 is to ensure same vf_coef as with other implementations)
            b_values = b_values.view(-1)
            if config.clip_loss_vf:
                loss_vf_unclipped = (b_rets - b_values)**2

                b_values_clipped = b_old_values + torch.clamp(b_values - b_old_values, -config.clip_ratio, +config.clip_ratio)
                loss_vf_clipped = (b_rets - b_values_clipped)**2

                loss_vf = 0.5 * torch.mean(torch.max(loss_vf_unclipped, loss_vf_clipped))
            else:
                loss_vf = 0.5 * torch.mean((b_rets - b_values)**2)

            # entropy loss
            loss_ent = b_entropy.mean()

            # total update
            loss = loss_pi + config.vf_coef * loss_vf - config.ent_coef * loss_ent
            loss.backward()
            _ = nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optim.step()
            optim.zero_grad()
        
        if config.max_kl is not None and approx_kl > config.max_kl:
            break
            
    y_pred, y_true = old_values.cpu().numpy(), rets.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return explained_var

if __name__ == "__main__":
    # 2_000/2^-5, 500/2^-6
    config = Config()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    
    if config.log_wandb:
        wandb.init(project="ppo", config={
            "algo": "ppo",
            "source": "mine",
            "env_id": config.env_id,
            "total_timesteps": config.total_timesteps,
            "step_per_rollout": config.num_steps,
            "lr": config.lr,
            "vf_coef": config.vf_coef,
            "gae_gamma": config.gae_gamma,
            "gae_lambda": config.gae_lambda,
        })

    envs = gym.vector.SyncVectorEnv([make_env(config.env_id, config.gae_gamma) for i in range(config.num_envs)])

    agent = Agent(envs)
    optim = torch.optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)
    
    total_steps = config.total_timesteps // config.num_steps

    for step in range(total_steps):
        obs, actions, old_logp, adv, old_values, rets, ep_rets = rollout()
        explained_var = update(obs, actions, old_logp, adv, old_values, rets)

        if config.anneal_lr:
            frac = 1.0 - (step / total_steps)
            lr = frac * config.lr
            optim.param_groups[0]["lr"] = lr

        num_digits = len(str(total_steps))
        formatted_iter = f"{step+1:0{num_digits}d}"
        print(f"Step: {formatted_iter}/{total_steps}. Avg return: {np.mean(ep_rets):.2f}.")

        if config.log_wandb:
            wandb.log({"returns": np.mean(ep_rets), "explained_var": explained_var, "lr": optim.param_groups[0]["lr"]}, step=(step+1)*config.num_steps)
