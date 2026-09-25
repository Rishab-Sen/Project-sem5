"""
QMIX — Monotonic Value Function Factorisation for Deep MARL
===========================================================
Reference: Rashid et al., "QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning", ICML 2018 (proposal reference [4]).

  * Per-agent Q networks: Q_i(tau_i, u_i) from LOCAL observations + last action.
  * Mixing network: Q_tot = f_mix(Q_1..Q_N, global_state) with NON-NEGATIVE
    weights (monotonicity) -> argmax_i Q_i == argmax Q_tot (CTDE guarantee).
  * Discretised actions: each wheel is binned into N_ACTIONS levels in [-1,1].
  * Double-Q evaluation, epsilon-greedy exploration, target networks with
    soft updates, n-step returns.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Per-agent Q network over local observation + previous action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor, last_action: torch.Tensor):
        """obs: (..., obs_dim), last_action: (..., n_actions) one-hot."""
        return self.net(torch.cat([obs, last_action], dim=-1))


class Mixer(nn.Module):
    """Monotonic mixing network conditioned on the global state."""

    def __init__(self, n_agents: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, n_agents * hidden))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden))
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 1))

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor):
        """
        agent_qs: (B, N), state: (B, state_dim) -> Q_tot: (B, 1)
        Absolute value enforces non-negative weights (monotonicity).
        """
        B = agent_qs.shape[0]
        w1 = self.hyper_w1(state).abs().reshape(B, self.n_agents, -1)
        b1 = self.hyper_b1(state).reshape(B, 1, -1)
        h = torch.relu(torch.bmm(agent_qs.reshape(B, 1, self.n_agents), w1) + b1)

        w2 = self.hyper_w2(state).abs().reshape(B, -1, 1)
        b2 = self.hyper_b2(state).reshape(B, 1, 1)
        return (torch.bmm(h, w2) + b2).reshape(B, 1)


def one_hot(indices: torch.Tensor, n: int) -> torch.Tensor:
    """(...,) int -> (..., n) float one-hot."""
    return F.one_hot(indices.long(), n).float()


class QMIX:
    def __init__(self, obs_dim: int, state_dim: int, n_actions: int,
                 n_agents: int = 3, lr: float = 1e-3, gamma: float = 0.99,
                 n_step: int = 3, buffer_size: int = 200_000,
                 batch_size: int = 64, target_update_tau: float = 0.005,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 100_000, device: str = "cpu"):
        self.n_agents, self.n_actions = n_agents, n_actions
        self.gamma, self.n_step = gamma, n_step
        self.batch_size = batch_size
        self.tau = target_update_tau
        self.device = torch.device(device)

        self.eps_start, self.eps_end = epsilon_start, epsilon_end
        self.eps_decay_steps = epsilon_decay_steps
        self.t_total = 0

        self.q = QNetwork(obs_dim, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mixer = Mixer(n_agents, state_dim).to(device)
        self.mixer_target = Mixer(n_agents, state_dim).to(device)
        self.mixer_target.load_state_dict(self.mixer.state_dict())

        self.params = list(self.q.parameters()) + list(self.mixer.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.last_loss = 0.0

        self.buffer = ReplayBuffer(buffer_size, n_agents, obs_dim, n_actions)
        # n-step accumulator per running episode
        self._ep_buffer: list[tuple] = []
        self._stored_upto = 0

    # ------------------------------------------------------------------ #
    def epsilon(self) -> float:
        frac = min(1.0, self.t_total / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, last_action_idx: np.ndarray,
            epsilon: float | None = None) -> np.ndarray:
        """Epsilon-greedy greedy-argmax over individual agent Qs (CTDE)."""
        eps = self.epsilon() if epsilon is None else epsilon
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_actions, size=self.n_agents)
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)  # (1,N,obs)
        last_t = one_hot(torch.as_tensor(last_action_idx), self.n_actions
                         ).to(self.device).unsqueeze(0)            # (1,N,A)
        q = self.q(obs_t, last_t).squeeze(0)                       # (N,A)
        return q.argmax(dim=-1).cpu().numpy()

    # ------------------------------------------------------------------ #
    def store_transition(self, obs, act_idx, reward, next_obs, done,
                         state, next_state, last_action_idx):
        """Accumulate with n-step returns; flush episode tail on done."""
        self._ep_buffer.append((obs, act_idx, reward, next_obs, done,
                                state, next_state, last_action_idx))
        if done:
            T = len(self._ep_buffer)
            for t in range(self._stored_upto, T):
                R, n_eff = 0.0, 0
                for k in range(t, min(t + self.n_step, T)):
                    R += (self.gamma ** (k - t)) * self._ep_buffer[k][2]
                    n_eff += 1
                o, a, _, _, _, s, _, la = self._ep_buffer[t]
                last_k = min(t + n_eff, T) - 1
                terminal = bool(self._ep_buffer[T - 1][4])
                self.buffer.add(o, a, R, self._ep_buffer[last_k][3],
                                terminal, s, self._ep_buffer[last_k][5], la)
            self._ep_buffer = []
            self._stored_upto = 0
        elif len(self._ep_buffer) >= self.n_step:
            t = self._stored_upto
            R = 0.0
            for k in range(t, t + self.n_step):
                R += (self.gamma ** (k - t)) * self._ep_buffer[k][2]
            o, a, _, _, _, s, _, la = self._ep_buffer[t]
            nk = self._ep_buffer[t + self.n_step - 1]
            self.buffer.add(o, a, R, nk[3], False, s, nk[5], la)
            self._stored_upto += 1

    def start_new_episode(self):
        """Discard partial episode data (call before reset of a fresh episode)."""
        self._ep_buffer = []
        self._stored_upto = 0

    # ------------------------------------------------------------------ #
    def update(self) -> dict | None:
        if self.buffer.size < max(self.batch_size * 10, 1000):
            return None

        batch = self.buffer.sample(self.batch_size)
        obs      = torch.as_tensor(batch["obs"], device=self.device)
        act      = torch.as_tensor(batch["act"], dtype=torch.long,
                                   device=self.device)
        reward   = torch.as_tensor(batch["reward"], dtype=torch.float32,
                                   device=self.device).reshape(-1, 1)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        done     = torch.as_tensor(batch["done"], dtype=torch.float32,
                                   device=self.device).reshape(-1, 1)
        state    = torch.as_tensor(batch["state"], device=self.device)
        next_st  = torch.as_tensor(batch["next_state"], device=self.device)
        last_act = torch.as_tensor(batch["last_act"], dtype=torch.long,
                                   device=self.device)

        B, N = obs.shape[0], self.n_agents
        oh_act  = one_hot(act, self.n_actions)
        oh_last = one_hot(last_act, self.n_actions)

        # Q_tot(s, a)
        q_vals = self.q(obs.reshape(B * N, -1), oh_last.reshape(B * N, -1))
        q_vals = q_vals.reshape(B, N, self.n_actions)
        picked = q_vals.gather(-1, act.unsqueeze(-1)).squeeze(-1)     # (B,N)
        q_tot = self.mixer(picked, state)                             # (B,1)

        # Double-Q target: online net selects, target net evaluates
        with torch.no_grad():
            q_next_online = self.q(next_obs.reshape(B * N, -1),
                                   oh_act.reshape(B * N, -1)
                                   ).reshape(B, N, self.n_actions)
            max_next = q_next_online.argmax(dim=-1)                   # (B,N)
            oh_max = one_hot(max_next, self.n_actions)
            q_next_tgt = self.q_target(next_obs.reshape(B * N, -1),
                                       oh_max.reshape(B * N, -1)
                                       ).reshape(B, N, self.n_actions)
            best_q = q_next_tgt.gather(-1, max_next.unsqueeze(-1)).squeeze(-1)
            q_tot_next = self.mixer_target(best_q, next_st)
            target = reward + (1 - done) * (self.gamma ** self.n_step) * q_tot_next

        loss = F.smooth_l1_loss(q_tot, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, 10.0)
        self.opt.step()

        # soft target updates
        for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.mixer.parameters(),
                         self.mixer_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.t_total += 1
        self.last_loss = loss.item()
        return {"q_loss": loss.item(),
                "q_tot_mean": q_tot.mean().item(),
                "epsilon": self.epsilon()}


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, n_actions: int):
        self.capacity = capacity
        self.n = n_agents
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.ptr = 0
        self.size = 0
        self.obs      = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.act      = np.zeros((capacity, n_agents), np.int64)
        self.reward   = np.zeros((capacity,), np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.done     = np.zeros((capacity,), np.float32)
        self.state    = np.zeros((capacity, obs_dim), np.float32)  # resized lazily
        self.next_state = np.zeros((capacity, obs_dim), np.float32)
        self.last_act = np.zeros((capacity, n_agents), np.int64)
        self._state_dim = None

    def add(self, obs, act, reward, next_obs, done, state, next_state, last_act):
        if self._state_dim is None:
            sd = np.asarray(state).size
            self.state = np.zeros((self.capacity, sd), np.float32)
            self.next_state = np.zeros((self.capacity, sd), np.float32)
            self._state_dim = sd
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.state[i] = state
        self.next_state[i] = next_state
        self.last_act[i] = last_act
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch)
        return {"obs": self.obs[idx], "act": self.act[idx],
                "reward": self.reward[idx], "next_obs": self.next_obs[idx],
                "done": self.done[idx], "state": self.state[idx],
                "next_state": self.next_state[idx],
                "last_act": self.last_act[idx]}
