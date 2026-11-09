"""Independent DQN: one Q-network per traffic signal, in plain PyTorch.

SB3 cannot train separate per-agent policies inside one multi-agent env, so
each intersection gets its own online network, target network and replay
buffer. Each network is trained on the target

.. math::

    y = r + \\gamma \\, (1 - d) \\, \\max_{a'} Q_{\\bar\\theta}(s', a')

with the Huber loss between :math:`Q_\\theta(s, a)` and :math:`y`, where
:math:`\\bar\\theta` is the target network, copied from :math:`\\theta` every
``target_update`` environment steps.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.envs.make_env import _DELTA_TIME, _NUM_SECONDS, make_env
from src.evaluation.metrics import _flatten_info

REPO_ROOT = Path(__file__).resolve().parents[2]
CURVE_COLUMNS = ["episode", "mean_system_waiting_time"]
EXPLORE_FRACTION = 0.3


def model_path(scenario: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "models" / "dqn" / scenario / f"seed{seed}.pt"


def curve_path(scenario: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "raw" / "train_curves" / f"dqn_{scenario}_seed{seed}.csv"


class QNetwork(nn.Module):
    """MLP with two hidden layers of 64 units."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size numpy ring buffer of (s, a, r, s', done)."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(self, obs, action: int, reward: float, next_obs, done: float) -> None:
        i = self.index
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, done
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict[str, Any], rng: np.random.Generator):
        self.n_actions = n_actions
        self.gamma = float(cfg["gamma"])
        self.q = QNetwork(obs_dim, n_actions)
        self.target = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=float(cfg["lr"]))
        self.buffer = ReplayBuffer(int(cfg["buffer"]), obs_dim)
        self.rng = rng

    def act(self, obs, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs, dtype=torch.float32)).argmax())

    def learn(self, batch: dict[str, np.ndarray]) -> float:
        obs = torch.as_tensor(batch["obs"])
        actions = torch.as_tensor(batch["actions"])
        rewards = torch.as_tensor(batch["rewards"])
        next_obs = torch.as_tensor(batch["next_obs"])
        dones = torch.as_tensor(batch["dones"])
        with torch.no_grad():
            target = rewards + self.gamma * (1.0 - dones) * self.target(next_obs).max(dim=1).values
        q_taken = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.q.state_dict())


def epsilon_at(step: int, total_steps: int, final_eps: float, fraction: float = EXPLORE_FRACTION) -> float:
    """Linear decay from 1.0 to final_eps over the first `fraction` of steps."""
    decay_steps = max(1, int(fraction * total_steps))
    progress = min(1.0, step / decay_steps)
    return 1.0 + progress * (final_eps - 1.0)


def save_model(agents: dict[str, DQNAgent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            agent_id: {
                "state_dict": agent.q.state_dict(),
                "obs_dim": agent.buffer.obs.shape[1],
                "n_actions": agent.n_actions,
            }
            for agent_id, agent in agents.items()
        },
        path,
    )


def _reset(env, seed: int):
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def _all_done(flags) -> bool:
    return isinstance(flags, dict) and bool(flags) and all(flags.values())


def train(
    scenario: str,
    seed: int,
    episodes: int,
    config: dict[str, Any],
    demand: str = "default",
) -> tuple[Path, Path]:
    cfg = config["dqn"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    batch_size, target_update = int(cfg["batch"]), int(cfg["target_update"])
    total_steps = episodes * (_NUM_SECONDS // _DELTA_TIME)

    env = make_env(scenario, demand=demand, split="train", seed=seed)
    agents: dict[str, DQNAgent] = {}
    curve, step = [], 0
    try:
        for episode in range(1, episodes + 1):
            obs = _reset(env, seed)
            for agent_id in env.agents:
                if agent_id not in agents:
                    agents[agent_id] = DQNAgent(
                        env.observation_space(agent_id).shape[0], env.action_space(agent_id).n, cfg, rng
                    )
            waits = []
            while env.agents:
                eps = epsilon_at(step, total_steps, float(cfg["final_eps"]))
                actions = {a: agents[a].act(obs[a], eps) for a in env.agents}
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                for a, action in actions.items():
                    if a not in next_obs:
                        continue
                    # Hitting the time limit is truncation, not a true terminal state.
                    done = float(bool(terminated.get(a, False)))
                    agent = agents[a]
                    agent.buffer.push(obs[a], action, float(rewards[a]), next_obs[a], done)
                    if len(agent.buffer) >= batch_size:
                        agent.learn(agent.buffer.sample(batch_size, rng))
                step += 1
                if step % target_update == 0:
                    for agent in agents.values():
                        agent.sync_target()
                flat = _flatten_info(info)
                if "system_mean_waiting_time" in flat:
                    waits.append(float(flat["system_mean_waiting_time"]))
                obs = next_obs
                if _all_done(terminated) or _all_done(truncated):
                    break
            mean_wait = float(np.mean(waits)) if waits else float("nan")
            curve.append({"episode": episode, "mean_system_waiting_time": mean_wait})
            print(f"dqn {scenario} seed={seed} episode={episode}/{episodes} eps={eps:.3f} mean_system_waiting_time={mean_wait:.3f}")
    finally:
        env.close()

    saved_model = model_path(scenario, seed)
    save_model(agents, saved_model)
    saved_curve = curve_path(scenario, seed)
    saved_curve.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve, columns=CURVE_COLUMNS).to_csv(saved_curve, index=False)
    return saved_model, saved_curve


def policy_factory(path: str | Path):
    """Greedy policy_fn(obs_dict, env) from a saved model file."""
    data = torch.load(path, map_location="cpu")
    nets = {}
    for agent_id, entry in data.items():
        net = QNetwork(int(entry["obs_dim"]), int(entry["n_actions"]))
        net.load_state_dict(entry["state_dict"])
        net.eval()
        nets[agent_id] = net

    def policy_fn(obs_dict: dict, env) -> dict:
        actions = {}
        with torch.no_grad():
            for agent_id in env.agents:
                net = nets.get(agent_id)
                if net is None:
                    actions[agent_id] = 0
                else:
                    obs = torch.as_tensor(obs_dict[agent_id], dtype=torch.float32)
                    actions[agent_id] = int(net(obs).argmax())
        return actions

    return policy_fn
