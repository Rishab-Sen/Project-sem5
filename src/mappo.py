"""
MAPPO — Multi-Agent PPO with centralized training, decentralized execution (CTDE)
==================================================================================
Reference: Yu et al., "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games", NeurIPS Deep RL Workshop 2022.

Design (matching proposal slide 6):
  * ONE shared actor MLP -> parameter sharing across rovers (decentralized:
    each rover acts only on its LOCAL observation).
  * ONE centralized critic MLP -> takes the GLOBAL state (all rover poses,
    payload pose/velocity, goal offset, rig stretches, LiDAR minima).
  * Beta policy on the bounded [-1,1] wheel actions, GAE(lambda) advantages,
    clipped surrogate objective, value clipping, orthogonal init.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta


def orthogonal_init(module: nn.Module, gain: float = np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)


class ActorCritic(nn.Module):
    """Shared actor (Beta policy) + centralized V(s) critic."""

    EPS = 1e-3  # keep Beta concentrations strictly positive away from 1

    def __init__(self, obs_dim: int, state_dim: int, act_dim: int,
                 hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden, act_dim)
        self.beta_head  = nn.Linear(hidden, act_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            orthogonal_init(m)
        orthogonal_init(self.alpha_head, gain=0.01)
        orthogonal_init(self.beta_head,  gain=0.01)
        orthogonal_init(self.critic, gain=1.0)

    def _dist(self, obs: torch.Tensor) -> Beta:
        h = self.trunk(obs)
        alpha = torch.nn.functional.softplus(self.alpha_head(h)) + 1.0 + self.EPS
        beta  = torch.nn.functional.softplus(self.beta_head(h))  + 1.0 + self.EPS
        return Beta(alpha, beta)

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """obs: (N, obs_dim). Returns (actions (N,2), logp (N,), entropy (N,))."""
        dist = self._dist(obs)
        if deterministic:
            a = dist.mean
            logp = dist.log_prob(a).sum(-1)
            ent = dist.entropy().sum(-1)
        else:
            a = dist.sample()
            logp = dist.log_prob(a).sum(-1)
            ent = dist.entropy().sum(-1)
        return a, logp, ent

    def evaluate_actions(self, obs: torch.Tensor, state: torch.Tensor,
                         actions: torch.Tensor):
        """
        obs: (M, obs_dim), state: (M_agents_expanded,) handled by caller.
        Caller passes obs/actions flattened over (batch*agents) and a matching
        per-sample value tensor index. Returns (logp (M,), entropy (M,), value (M,)).
        """
        dist = self._dist(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def values(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, state_dim) -> (B,)"""
        return self.critic(state).reshape(-1)


class MAPPO:
    """PPO update logic: GAE, minibatch epochs, clipped objectives."""

    def __init__(self, obs_dim: int, state_dim: int, act_dim: int,
                 n_agents: int = 3, lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 epochs: int = 10, minibatches: int = 4,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5, device: str = "cpu",
                 target_kl: float | None = 0.03):
        self.n_agents = n_agents
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_eps, self.epochs = clip_eps, epochs
        self.minibatches, self.entropy_coef = minibatches, entropy_coef
        self.value_coef, self.max_grad_norm = value_coef, max_grad_norm
        self.target_kl = target_kl
        self.device = torch.device(device)

        self.policy = ActorCritic(obs_dim, state_dim, act_dim).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------ #
    def compute_gae(self, rewards, values, dones, last_value):
        """Team-level GAE. rewards/values/dones are per env-step, shape (T,)."""
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float64)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_v * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
        return adv, adv + np.asarray(values, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def update(self, rollout: dict) -> dict:
        """
        rollout keys (torch tensors already on CPU):
          obs (T,N,obs)  state (T,state)  actions (T,N,2)  logp (T,N)
          rewards (T,)   dones (T,)       values (T,)
          last_state (state,)
        """
        device = self.device
        T = rollout["rewards"].shape[0]

        with torch.no_grad():
            last_value = self.policy.values(
                rollout["last_state"].unsqueeze(0).to(device)).item()

        adv, ret = self.compute_gae(rollout["rewards"],
                                    rollout["values"].numpy(),
                                    rollout["dones"].numpy(), last_value)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # flatten (T, N, ...) -> (T*N, ...)
        obs_b  = rollout["obs"].reshape(T * self.n_agents, -1).to(device)
        act_b  = rollout["actions"].reshape(T * self.n_agents, -1).to(device)
        logp_b = rollout["logp"].reshape(T * self.n_agents).to(device)

        # per flattened sample, which env-step t it belongs to
        step_idx = torch.arange(T).repeat_interleave(self.n_agents).to(device)
        ret_b = ret_t.to(device)[step_idx]
        adv_b = adv_t.to(device)[step_idx]

        B = obs_b.shape[0]
        mb_size = max(1, B // self.minibatches)
        idx_np = np.arange(B)

        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                 "clip_frac": 0.0, "approx_kl": 0.0, "n_updates": 0}
        early_stop = False

        for _ in range(self.epochs):
            if early_stop:
                break
            np.random.shuffle(idx_np)
            for start in range(0, B, mb_size):
                mb = idx_np[start:start + mb_size]
                if len(mb) == 0:
                    continue
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)

                logp, ent = self.policy.evaluate_actions(
                    obs_b[mb_t], None, act_b[mb_t])
                value = self.policy.values(
                    rollout["state"].to(device)[step_idx[mb_t]])

                # policy clipped surrogate
                ratio = torch.exp(logp - logp_b[mb_t])
                surr1 = ratio * adv_b[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv_b[mb_t]
                pi_loss = -torch.min(surr1, surr2).mean()

                # value clipped loss
                v_clipped = rollout["values"].to(device)[step_idx[mb_t]] + \
                    torch.clamp(value - rollout["values"].to(device)[step_idx[mb_t]],
                                -self.clip_eps, self.clip_eps)
                v_loss = 0.5 * torch.max(
                    (value - ret_b[mb_t]) ** 2,
                    (v_clipped - ret_b[mb_t]) ** 2).mean()

                loss = (pi_loss + self.value_coef * v_loss
                        - self.entropy_coef * ent.mean())

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["pi_loss"] += pi_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += ent.mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs()
                                           > self.clip_eps).float().mean().item()
                    log_ratio = logp - logp_b[mb_t]
                    stats["approx_kl"] += (logp_b[mb_t] - logp).mean().item() \
                        + 0.5 * (log_ratio ** 2).mean().item()
                    stats["n_updates"] += 1

            # early stop on KL divergence
            if self.target_kl is not None and stats["n_updates"] > 0:
                kl_now = stats["approx_kl"] / stats["n_updates"]
                if kl_now > 1.5 * self.target_kl:
                    early_stop = True

        n = max(stats["n_updates"], 1)
        return {k: v / n for k, v in stats.items() if k != "n_updates"} | \
               {"n_updates": stats["n_updates"], "early_stopped": early_stop}
