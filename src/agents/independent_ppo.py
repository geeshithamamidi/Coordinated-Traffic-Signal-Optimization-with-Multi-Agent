"""Independent PPO: one actor-critic per traffic signal, no parameter sharing.

CleanRL-style PPO in plain PyTorch. All agents step together in one parallel
SUMO environment, but each agent keeps its own rollout buffer of its own
decisions and runs its own clipped-surrogate update when the buffer is full.

Why independent learners face non-stationarity
----------------------------------------------
Each agent treats the rest of the network as part of its environment. When a
neighboring intersection changes its policy, the arrival pattern on this
agent's incoming lanes changes too, so the transition and reward distribution
this agent learns from keeps moving during training. Its value estimates and
advantages are therefore computed against a target that drifts, which can slow
or destabilize learning. Parameter sharing and coordination (neighbor
observations or rewards) are the remedies studied elsewhere in this project;
this module is the uncoordinated reference point.

Advantages use GAE:

.. math::

    \\delta_t = r_t + \\gamma (1 - d_t) V(s_{t+1}) - V(s_t), \\qquad
    \\hat A_t = \\delta_t + \\gamma \\lambda (1 - d_t) \\hat A_{t+1}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from src.envs.make_env import make_env
from src.evaluation.metrics import _flatten_info

REPO_ROOT = Path(__file__).resolve().parents[2]
CURVE_COLUMNS = ["episode", "mean_system_waiting_time"]
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


def model_path(scenario: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "models" / "ippo" / scenario / f"seed{seed}.pt"


def curve_path(scenario: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "raw" / "train_curves" / f"ippo_{scenario}_seed{seed}.csv"


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out_dim))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.actor = _mlp(obs_dim, n_actions)
        self.critic = _mlp(obs_dim, 1)

    def dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(x))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)


def compute_gae(rewards, values, dones, last_value: float, gamma: float, lam: float):
    """Return (advantages, returns). dones[t] = 1 if the episode ended after step t."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values


class Rollout:
    def __init__(self, n_steps: int, obs_dim: int):
        self.n_steps = n_steps
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.logps = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.size = 0

    def add(self, obs, action, logp, reward, done, value) -> None:
        i = self.size
        self.obs[i], self.actions[i], self.logps[i] = obs, action, logp
        self.rewards[i], self.dones[i], self.values[i] = reward, done, value
        self.size += 1

    @property
    def full(self) -> bool:
        return self.size >= self.n_steps


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict[str, Any], rng: np.random.Generator):
        self.n_actions = int(n_actions)
        self.obs_dim = int(obs_dim)
        self.cfg = cfg
        self.net = ActorCritic(self.obs_dim, self.n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg["lr"]), eps=1e-5)
        self.buffer = Rollout(int(cfg["n_steps"]), self.obs_dim)
        self.rng = rng

    def act(self, obs) -> tuple[int, float, float]:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            dist = self.net.dist(x)
            action = dist.sample()
            return int(action), float(dist.log_prob(action)), float(self.net.value(x))

    def update(self, last_obs) -> None:
        cfg, buf = self.cfg, self.buffer
        n = buf.size
        with torch.no_grad():
            last_value = float(self.net.value(torch.as_tensor(last_obs, dtype=torch.float32)))
        adv, ret = compute_gae(
            buf.rewards[:n], buf.values[:n], buf.dones[:n], last_value, float(cfg["gamma"]), float(cfg["gae_lambda"])
        )
        obs = torch.as_tensor(buf.obs[:n])
        actions = torch.as_tensor(buf.actions[:n])
        old_logp = torch.as_tensor(buf.logps[:n])
        adv_t, ret_t = torch.as_tensor(adv), torch.as_tensor(ret)
        clip, batch = float(cfg["clip"]), int(cfg["batch"])
        index = np.arange(n)
        for _ in range(int(cfg["n_epochs"])):
            self.rng.shuffle(index)
            for start in range(0, n, batch):
                mb = torch.as_tensor(index[start : start + batch])
                dist = self.net.dist(obs[mb])
                ratio = torch.exp(dist.log_prob(actions[mb]) - old_logp[mb])
                mb_adv = adv_t[mb]
                if mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                v_loss = 0.5 * ((self.net.value(obs[mb]) - ret_t[mb]) ** 2).mean()
                loss = pg_loss + VALUE_COEF * v_loss - ENTROPY_COEF * dist.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
        buf.size = 0


def save_model(agents: dict[str, PPOAgent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            agent_id: {"state_dict": agent.net.state_dict(), "obs_dim": agent.obs_dim, "n_actions": agent.n_actions}
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
    cfg = config["ppo"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(scenario, demand=demand, split="train", seed=seed)
    agents: dict[str, PPOAgent] = {}
    curve = []
    try:
        for episode in range(1, episodes + 1):
            obs = _reset(env, seed)
            for agent_id in env.agents:
                if agent_id not in agents:
                    agents[agent_id] = PPOAgent(
                        env.observation_space(agent_id).shape[0], env.action_space(agent_id).n, cfg, rng
                    )
            waits = []
            while env.agents:
                decisions = {a: agents[a].act(obs[a]) for a in env.agents}
                actions = {a: d[0] for a, d in decisions.items()}
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                ended = not env.agents or _all_done(terminated) or _all_done(truncated)
                for a, (action, logp, value) in decisions.items():
                    agent = agents[a]
                    done = 1.0 if ended or terminated.get(a, False) else 0.0
                    agent.buffer.add(obs[a], action, logp, float(rewards.get(a, 0.0)), done, value)
                    if agent.buffer.full:
                        agent.update(next_obs.get(a, obs[a]))
                flat = _flatten_info(info)
                if "system_mean_waiting_time" in flat:
                    waits.append(float(flat["system_mean_waiting_time"]))
                obs = next_obs
                if ended:
                    break
            mean_wait = float(np.mean(waits)) if waits else float("nan")
            curve.append({"episode": episode, "mean_system_waiting_time": mean_wait})
            print(f"ippo {scenario} seed={seed} episode={episode}/{episodes} mean_system_waiting_time={mean_wait:.3f}")
    finally:
        env.close()

    saved_model = model_path(scenario, seed)
    save_model(agents, saved_model)
    saved_curve = curve_path(scenario, seed)
    saved_curve.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve, columns=CURVE_COLUMNS).to_csv(saved_curve, index=False)
    return saved_model, saved_curve


def policy_factory(path: str | Path):
    """Deterministic (argmax) policy_fn(obs_dict, env) from a saved model file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    nets = {}
    for agent_id, entry in data.items():
        net = ActorCritic(int(entry["obs_dim"]), int(entry["n_actions"]))
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
                    logits = net.actor(torch.as_tensor(obs_dict[agent_id], dtype=torch.float32))
                    actions[agent_id] = int(logits.argmax())
        return actions

    return policy_fn
