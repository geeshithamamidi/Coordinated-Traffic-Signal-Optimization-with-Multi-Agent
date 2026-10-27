from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.make_env import make_env  # noqa: E402
from src.evaluation.metrics import _flatten_info, _sumo_environment  # noqa: E402
from src.utils.config import load_config  # noqa: E402

N_BINS = 10


def model_path(scenario: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "models" / "ql" / scenario / f"seed{seed}.pkl"


def curve_path(scenario: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "raw" / "train_curves" / f"ql_{scenario}_seed{seed}.csv"


def encode_obs(obs, n_phases: int, n_bins: int = N_BINS) -> tuple[int, ...]:
    values = np.asarray(obs, dtype=float)
    phase = [int(round(v)) for v in values[:n_phases]]
    min_green = [int(round(values[n_phases]))] if len(values) > n_phases else []
    rest = np.clip(values[n_phases + 1:], 0.0, 1.0)
    bins = np.minimum((rest * n_bins).astype(int), n_bins - 1).tolist()
    return tuple(phase + min_green + bins)


def _phase_counts(env) -> dict[str, int]:
    sumo_env = _sumo_environment(env)
    return {ts_id: int(ts.num_green_phases) for ts_id, ts in sumo_env.traffic_signals.items()}


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
    from sumo_rl.agents import QLAgent
    from sumo_rl.exploration import EpsilonGreedy

    np.random.seed(seed)
    ql = config["ql"]
    env = make_env(scenario, demand=demand, split="train", seed=seed)
    agents: dict[str, QLAgent] = {}
    curve = []
    try:
        for episode in range(1, episodes + 1):
            obs = _reset(env, seed)
            phases = _phase_counts(env)
            for agent_id in env.agents:
                state = encode_obs(obs[agent_id], phases[agent_id])
                if agent_id not in agents:
                    agents[agent_id] = QLAgent(
                        starting_state=state,
                        state_space=env.observation_space(agent_id),
                        action_space=env.action_space(agent_id),
                        alpha=ql["alpha"],
                        gamma=ql["gamma"],
                        exploration_strategy=EpsilonGreedy(
                            initial_epsilon=ql["epsilon_start"],
                            min_epsilon=ql["epsilon_min"],
                            decay=ql["epsilon_decay"],
                        ),
                    )
                else:
                    agents[agent_id].state = state
                    agents[agent_id].q_table.setdefault(state, [0] * env.action_space(agent_id).n)
                    agents[agent_id].acc_reward = 0
            waits = []
            while env.agents:
                actions = {agent_id: agents[agent_id].act() for agent_id in env.agents}
                obs, rewards, terminated, truncated, info = env.step(actions)
                for agent_id, agent in agents.items():
                    if agent_id in obs:
                        agent.learn(encode_obs(obs[agent_id], phases[agent_id]), rewards[agent_id])
                flat = _flatten_info(info)
                if "system_mean_waiting_time" in flat:
                    waits.append(float(flat["system_mean_waiting_time"]))
                if _all_done(terminated) or _all_done(truncated):
                    break
            mean_wait = float(np.mean(waits)) if waits else float("nan")
            curve.append({"episode": episode, "mean_waiting_time": mean_wait})
            print(f"ql {scenario} seed={seed} episode={episode}/{episodes} mean_waiting_time={mean_wait:.3f}")
    finally:
        env.close()

    saved_model = model_path(scenario, seed)
    saved_model.parent.mkdir(parents=True, exist_ok=True)
    with saved_model.open("wb") as handle:
        pickle.dump({agent_id: dict(agent.q_table) for agent_id, agent in agents.items()}, handle)
    saved_curve = curve_path(scenario, seed)
    saved_curve.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(saved_curve, index=False)
    return saved_model, saved_curve


def policy_factory(path: str | Path):
    with Path(path).open("rb") as handle:
        q_tables: dict[str, dict] = pickle.load(handle)

    def policy_fn(obs_dict: dict, env) -> dict:
        sumo_env = _sumo_environment(env)
        actions = {}
        for agent_id in env.agents:
            ts = sumo_env.traffic_signals[agent_id]
            state = encode_obs(obs_dict[agent_id], int(ts.num_green_phases))
            values = q_tables.get(agent_id, {}).get(state)
            if values is None or max(values) == min(values):
                actions[agent_id] = (int(ts.green_phase) + 1) % int(ts.num_green_phases)
            else:
                actions[agent_id] = int(np.argmax(values))
        return actions

    return policy_fn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent tabular Q-learning agents.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--demand", default="default")
    parser.add_argument("--config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    saved_model, saved_curve = train(args.scenario, args.seed, args.episodes, config, args.demand)
    print(f"model: {saved_model}")
    print(f"curve: {saved_curve}")


if __name__ == "__main__":
    main()
