from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.make_env import make_env
from src.evaluation.metrics import _sumo_environment, run_episode

FIXED_TIME_ENV_KWARGS: dict[str, Any] = {"fixed_ts": True}
GREEN_CANDIDATES = (10, 20, 30)


class FixedTimeController:
    env_kwargs = FIXED_TIME_ENV_KWARGS

    def __call__(self, obs_dict: dict, env) -> dict:
        return {agent: 0 for agent in env.agents}


class FixedCycleController:
    env_kwargs: dict[str, Any] = {}

    def __init__(self, green_s: int = 20):
        if green_s <= 0:
            raise ValueError(f"green_s must be positive, got {green_s}.")
        self.green_s = green_s

    def __call__(self, obs_dict: dict, env) -> dict:
        sumo_env = _sumo_environment(env)
        elapsed = float(sumo_env.sim_step) - float(getattr(sumo_env, "begin_time", 0))
        slot = int(elapsed // self.green_s)
        actions = {}
        for agent in env.agents:
            n_phases = int(env.action_space(agent).n)
            actions[agent] = slot % n_phases
        return actions


def fixed_time_factory(train_seed: int) -> FixedTimeController:
    return FixedTimeController()


def fixed_cycle_factory(green_s: int):
    def factory(train_seed: int) -> FixedCycleController:
        return FixedCycleController(green_s)

    return factory


def tune_fixed_cycle(
    scenario: str,
    demand: str,
    validation_seeds: list[int],
    candidates: tuple[int, ...] = GREEN_CANDIDATES,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for green_s in candidates:
        waits = []
        for seed in validation_seeds:
            env = make_env(scenario, demand=demand, split="train", seed=seed)
            result = run_episode(env, FixedCycleController(green_s), seed)
            waits.append(result["avg_waiting_time"])
        scores[green_s] = float(np.mean(waits))
        print(f"fixed_cycle green_s={green_s} validation avg_waiting_time={scores[green_s]:.3f}")
    best = min(scores, key=scores.get)
    return best, scores
