#!/usr/bin/env python3
"""Train one method, then score it on the config test seeds.

    python scripts/run_experiment.py --method shared_ppo --scenario single --demand default --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import evaluate  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_JOBS = REPO_ROOT / "hpc" / "experiments.yaml"

DEFAULT_EPISODES = {
    "single": 100,
    "grid4x4": 300,
    "arterial4x4": 200,
    "cologne8": 200,
    "ingolstadt7": 200,
}

Trainer = Callable[[argparse.Namespace, dict[str, Any], int], Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one controller and evaluate it on test seeds.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--demand", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes", type=int, help="Override the training budget for this scenario")
    parser.add_argument("--config", help="YAML merged over configs/base.yaml")
    parser.add_argument("--jobs-config", type=Path, default=DEFAULT_JOBS, help="Sweep YAML with episode budgets")
    return parser.parse_args(argv)


def episode_budget(scenario: str, jobs_config: Path | None, override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError(f"episodes must be positive, got {override}.")
        return override
    if jobs_config is not None and jobs_config.is_file():
        spec = yaml.safe_load(jobs_config.read_text()) or {}
        episodes = spec.get("episodes") if isinstance(spec, dict) else None
        if isinstance(episodes, dict) and scenario in episodes:
            budget = int(episodes[scenario])
            if budget < 1:
                raise ValueError(f"episodes for {scenario} must be positive, got {budget}.")
            return budget
    if scenario not in DEFAULT_EPISODES:
        known = ", ".join(DEFAULT_EPISODES)
        raise ValueError(f"No episode budget for scenario {scenario!r}. Known: {known}.")
    return DEFAULT_EPISODES[scenario]


def resolve_trainer(method: str) -> Trainer:
    """Pick the trainer for a jobs.txt method name."""
    if method == "shared_ppo":
        return _train_shared_ppo
    if method == "ql":
        return _train_ql
    if method == "dqn":
        return _train_dqn
    if method.startswith("coordinated_"):
        alpha = _coordinated_alpha(method)
        return lambda args, config, episodes, alpha=alpha: _train_coordinated(args, config, episodes, alpha)
    raise ValueError(
        f"Unknown method {method!r}. Expected shared_ppo, ql, dqn, or coordinated_<alpha>."
    )


def run(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    episodes = episode_budget(args.scenario, args.jobs_config, args.episodes)
    test_seeds = [int(seed) for seed in config["seeds"]["test"]]
    print(
        f"run_experiment method={args.method} scenario={args.scenario} demand={args.demand} "
        f"seed={args.seed} episodes={episodes} test_seeds={test_seeds}"
    )
    trainer = resolve_trainer(args.method)
    trainer(args, config, episodes)
    return evaluate(
        args.method,
        args.scenario,
        args.demand,
        _policy_factory(args.method, args.scenario),
        args.seed,
        test_seeds,
    )


def _train_shared_ppo(args: argparse.Namespace, config: dict[str, Any], episodes: int) -> None:
    from src.agents.shared_ppo import train

    model_file, curve_file = train(
        args.scenario,
        args.seed,
        episodes,
        config,
        demand=args.demand,
    )
    print(f"model: {model_file}")
    print(f"curve: {curve_file}")


def _train_ql(args: argparse.Namespace, config: dict[str, Any], episodes: int) -> None:
    from src.baselines.independent_ql import train

    model_file, curve_file = train(args.scenario, args.seed, episodes, config, demand=args.demand)
    print(f"model: {model_file}")
    print(f"curve: {curve_file}")


def _train_dqn(args: argparse.Namespace, config: dict[str, Any], episodes: int) -> None:
    raise NotImplementedError(
        "TODO: DQN trainer. Train with configs/base.yaml dqn hyperparameters "
        f"(lr={config['dqn']['lr']}, gamma={config['dqn']['gamma']}, "
        f"buffer={config['dqn']['buffer']}) for {episodes} episodes, "
        "save a model, and add a policy_factory branch in scripts/run_experiment.py."
    )


def _train_coordinated(args: argparse.Namespace, config: dict[str, Any], episodes: int, alpha: float) -> None:
    known = config["coordinated"]["neighbor_alpha"]
    raise NotImplementedError(
        "TODO: coordinated trainer. "
        f"neighbor_alpha={alpha} (sweep {known}) for {episodes} episodes of {args.scenario}. "
        "Save a model and add a policy_factory branch in scripts/run_experiment.py."
    )


def _coordinated_alpha(method: str) -> float:
    suffix = method.removeprefix("coordinated_")
    try:
        return float(suffix)
    except ValueError as exc:
        raise ValueError(f"Method {method!r} must look like coordinated_0.25.") from exc


def _policy_factory(method: str, scenario: str):
    """Return evaluate's policy_factory(train_seed) -> policy_fn."""
    if method == "shared_ppo":
        from src.agents.shared_ppo import model_path, policy_factory

        def factory(train_seed: int):
            return policy_factory(model_path(scenario, train_seed))

        return factory
    if method == "ql":
        from src.baselines.independent_ql import model_path as ql_model_path
        from src.baselines.independent_ql import policy_factory as ql_policy_factory

        def ql_factory(train_seed: int):
            return ql_policy_factory(ql_model_path(scenario, train_seed))

        return ql_factory
    raise NotImplementedError(
        f"TODO: policy_factory for {method} on {scenario}. "
        "The trainer must save a model this function can load with deterministic actions."
    )


def main(argv: list[str] | None = None) -> None:
    frame = run(parse_args(argv))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
