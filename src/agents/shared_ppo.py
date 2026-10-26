"""One shared PPO policy for every traffic signal, trained with Stable-Baselines3.

The vectorization follows sumo-rl's ``experiments/sb3_grid4x4.py``: pad heterogeneous
spaces, turn the PettingZoo parallel env into a vector env (one slot per agent),
then concatenate copies for SB3. ``make_env`` supplies the network, the 3600 s
horizon, and the reward, so this controller trains on the same episodes as the
other methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from supersuit.utils.action_transforms.homogenize_ops import (
    dehomogenize_actions,
    homogenize_observations,
)

from src.envs.make_env import make_env

# make_env fixes num_seconds=3600 and delta_time=5, so each episode is 720 decisions.
DECISION_STEPS = 720

REPO_ROOT = Path(__file__).resolve().parents[2]

CURVE_COLUMNS = ["episode", "mean_system_waiting_time"]


def model_path(scenario: str, seed: int) -> Path:
    """Zip path for a shared PPO run: ``results/models/shared_ppo/<scenario>/seed<k>.zip``."""
    return REPO_ROOT / "results" / "models" / "shared_ppo" / scenario / f"seed{seed}.zip"


def curve_path(scenario: str, seed: int) -> Path:
    """Per-episode training curve CSV for one scenario and seed."""
    return REPO_ROOT / "results" / "raw" / "train_curves" / f"shared_ppo_{scenario}_seed{seed}.csv"


def tensorboard_path(scenario: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "raw" / "train_curves" / "tensorboard" / "shared_ppo" / scenario / f"seed{seed}"


def training_timesteps(episodes: int, n_agents: int) -> int:
    """SB3 timesteps for ``episodes`` of traffic. Each decision steps every agent."""
    if episodes < 1:
        raise ValueError(f"episodes must be positive, got {episodes}.")
    if n_agents < 1:
        raise ValueError(f"n_agents must be positive, got {n_agents}.")
    return episodes * DECISION_STEPS * n_agents


def make_vec_env(
    scenario: str,
    seed: int,
    n_envs: int = 1,
    n_cpus: int = 1,
    reward_fn: str = "diff-waiting-time",
    demand: str = "default",
):
    """Build the SB3 vector env used for shared PPO.

    Padding is a no-op when every signal has the same observation and action
    space. RESCO cities such as Ingolstadt do not, and the shared MLP needs
    one space.
    """
    env = make_env(
        scenario,
        demand=demand,
        seed=seed,
        reward_fn=reward_fn,
        additional_sumo_cmd="--no-step-log --no-warnings",
    )
    n_agents = len(env.possible_agents)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)  # needed for heterogeneous RESCO cities
    # sumo-rl 1.4.5 stores render_mode on SumoEnvironment, not on SumoEnvironmentPZ.
    # SuperSuit reads par_env.unwrapped.render_mode while building the vector env.
    base = env.unwrapped
    if not hasattr(base, "render_mode"):
        base.render_mode = getattr(getattr(base, "env", None), "render_mode", None)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=n_cpus, base_class="stable_baselines3")
    # SuperSuit forwards seed() into the gymnasium vec env, which has no seed().
    # SB3 only stores the seeds here; SUMO is already seeded by make_env(sumo_seed=seed).
    env.seed = lambda seed=None: VecEnv.seed(env, seed)
    env = VecMonitor(env)
    return env, n_agents


def build_model(env, config: dict[str, Any], seed: int, tensorboard_log: str, verbose: int = 1) -> PPO:
    """PPO with the ``ppo`` block of the merged config."""
    ppo = config["ppo"]
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo["lr"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip"],
        n_steps=int(ppo["n_steps"]),
        n_epochs=int(ppo["n_epochs"]),
        batch_size=int(ppo["batch"]),
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )


class MeanSystemWaitingCallback(BaseCallback):
    """Record the mean of ``system_mean_waiting_time`` over each finished episode.

    The vector env has one slot per agent (and per ``n_envs`` copy). Those slots
    share a simulation, so each episode is read from the first agent of a copy.
    Training stops once ``episodes`` traffic episodes have been recorded, which
    keeps the curve aligned with the 720-decision horizon.
    """

    def __init__(self, n_agents: int, episodes: int):
        super().__init__()
        self.n_agents = n_agents
        self.episodes = episodes
        self.rows: list[dict[str, float]] = []
        self._buffers: list[list[float]] = []

    def _on_training_start(self) -> None:
        if self.training_env.num_envs % self.n_agents != 0:
            raise RuntimeError(
                f"Vector env size {self.training_env.num_envs} is not a multiple of {self.n_agents} agents."
            )
        n_copies = self.training_env.num_envs // self.n_agents
        self._buffers = [[] for _ in range(n_copies)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for copy_idx, buffer in enumerate(self._buffers):
            index = copy_idx * self.n_agents
            buffer.append(_system_mean_waiting_time(infos[index]))
            if dones[index]:
                if len(buffer) != DECISION_STEPS:
                    raise RuntimeError(
                        f"Episode ended after {len(buffer)} decisions, expected {DECISION_STEPS}."
                    )
                self.rows.append(
                    {
                        "episode": len(self.rows) + 1,
                        "mean_system_waiting_time": float(np.mean(buffer)),
                    }
                )
                self._buffers[copy_idx] = []
        return len(self.rows) < self.episodes

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=CURVE_COLUMNS).to_csv(path, index=False)


def train(
    scenario: str,
    seed: int,
    episodes: int,
    config: dict[str, Any],
    n_envs: int = 1,
    n_cpus: int = 1,
    verbose: int = 1,
    demand: str = "default",
) -> tuple[Path, Path]:
    """Train shared PPO and write the model zip and the waiting-time curve.

    ``total_timesteps`` is ``episodes * 720 * n_agents``. The callback also
    stops on the episode boundary so a rollout does not run past that budget.
    """
    saved_model = model_path(scenario, seed)
    saved_curve = curve_path(scenario, seed)
    log_dir = tensorboard_path(scenario, seed)
    log_dir.mkdir(parents=True, exist_ok=True)

    env, n_agents = make_vec_env(
        scenario,
        seed=seed,
        n_envs=n_envs,
        n_cpus=n_cpus,
        reward_fn=config["env"]["reward_fn"],
        demand=demand,
    )
    callback = MeanSystemWaitingCallback(n_agents=n_agents, episodes=episodes)
    total_timesteps = training_timesteps(episodes, n_agents)
    print(
        f"shared PPO: scenario={scenario} seed={seed} episodes={episodes} "
        f"agents={n_agents} total_timesteps={total_timesteps}"
    )
    try:
        model = build_model(env, config, seed=seed, tensorboard_log=str(log_dir), verbose=verbose)
        model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name="shared_ppo")
        saved_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(saved_model)
    finally:
        callback.write(saved_curve)
        env.close()
    return saved_model, saved_curve


def policy_factory(model_path: str | Path):
    """Load a shared PPO zip and return ``policy_fn(obs_dict, env) -> action_dict``.

    Actions are deterministic. Observations are padded to the space the policy
    trained on, and actions are mapped back onto each signal's real discrete
    space, matching SuperSuit's pad wrappers. ``evaluate`` can call it as
    ``lambda train_seed: policy_factory(model_path(scenario, train_seed))``.
    """
    model = PPO.load(model_path)

    def policy_fn(obs_dict: dict, env) -> dict:
        actions = {}
        for agent in env.agents:
            observation = np.asarray(obs_dict[agent], dtype=model.observation_space.dtype)
            padded = homogenize_observations(model.observation_space, observation)
            action, _state = model.predict(padded, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            action = int(dehomogenize_actions(env.action_space(agent), action))
            actions[agent] = action
        return actions

    return policy_fn


def _system_mean_waiting_time(info: dict) -> float:
    if "system_mean_waiting_time" in info:
        return float(info["system_mean_waiting_time"])
    for value in info.values():
        if isinstance(value, dict) and "system_mean_waiting_time" in value:
            return float(value["system_mean_waiting_time"])
    raise KeyError("system_mean_waiting_time is missing from the vector-env info dict.")
