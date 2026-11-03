"""Shared PPO with neighbor observations and a spatial reward.

The SuperSuit stack is the one in ``src.agents.shared_ppo``. This module only
changes the PettingZoo env that enters that stack:

``SpatialRewardWrapper(NeighborObsWrapper(make_env(...), neighbor_map), neighbor_map, alpha)``

``--neighbor-obs`` and ``--alpha`` cover the four ablations:

* shared PPO: observations off, alpha 0
* obs only: observations on, alpha 0
* reward only: observations off, alpha > 0
* coordinated: observations on, the alpha chosen on validation seeds

Alpha is scored on ``configs/base.yaml`` ``seeds.validation`` and written to
``configs/frozen.yaml``. Test seeds are rejected before any rollout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from src.agents.shared_ppo import policy_factory as shared_policy_factory
from src.agents.shared_ppo import train as shared_train
from src.envs.coordination import NeighborObsWrapper, SpatialRewardWrapper
from src.envs.make_env import make_env
from src.envs.neighbors import build_neighbor_map
from src.evaluation.metrics import _sumo_environment, run_episode

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_PATH = REPO_ROOT / "configs" / "frozen.yaml"

ALLOWED_ALPHAS = (0.0, 0.25, 0.5, 0.75)

# Method name -> how run_experiment turns it into wrapper flags.
# ``alpha`` and ``reward_alpha`` are read from configs/frozen.yaml.
ABLATIONS = {
    "coordinated_ppo": {"neighbor_obs": True, "alpha_key": "alpha"},
    "obs_only": {"neighbor_obs": True, "alpha_key": None},
    "reward_only": {"neighbor_obs": False, "alpha_key": "reward_alpha"},
}

ScoreFn = Callable[[float, list[int]], float]


def canonical_alpha(alpha: float) -> float:
    """Return ``alpha`` when it is one of 0, 0.25, 0.5, or 0.75."""
    value = float(alpha)
    for allowed in ALLOWED_ALPHAS:
        if abs(value - allowed) <= 1e-8:
            return allowed
    allowed = ", ".join(f"{item:g}" for item in ALLOWED_ALPHAS)
    raise ValueError(f"alpha must be one of {allowed}, got {alpha}.")


def alpha_tag(alpha: float) -> str:
    """Stable path fragment: ``0``, ``0.25``, ``0.5``, ``0.75``."""
    return f"{canonical_alpha(alpha):g}"


def parse_neighbor_obs(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "on":
        return True
    if text == "off":
        return False
    raise ValueError(f"neighbor-obs must be 'on' or 'off', got {value!r}.")


def model_path(scenario: str, seed: int, neighbor_obs: bool, alpha: float) -> Path:
    """Zip path that keeps each ablation from overwriting the others."""
    obs = "on" if neighbor_obs else "off"
    tag = alpha_tag(alpha)
    return (
        REPO_ROOT
        / "results"
        / "models"
        / "coordinated_ppo"
        / scenario
        / f"obs_{obs}_alpha{tag}"
        / f"seed{seed}.zip"
    )


def curve_path(scenario: str, seed: int, neighbor_obs: bool, alpha: float) -> Path:
    obs = "on" if neighbor_obs else "off"
    tag = alpha_tag(alpha)
    name = f"coordinated_ppo_{scenario}_obs_{obs}_alpha{tag}_seed{seed}.csv"
    return REPO_ROOT / "results" / "raw" / "train_curves" / name


def tensorboard_path(scenario: str, seed: int, neighbor_obs: bool, alpha: float) -> Path:
    obs = "on" if neighbor_obs else "off"
    tag = alpha_tag(alpha)
    return (
        REPO_ROOT
        / "results"
        / "raw"
        / "train_curves"
        / "tensorboard"
        / "coordinated_ppo"
        / scenario
        / f"obs_{obs}_alpha{tag}"
        / f"seed{seed}"
    )


def make_coordinated_env(
    scenario: str,
    seed: int,
    neighbor_obs: bool,
    alpha: float,
    reward_fn: str = "diff-waiting-time",
    demand: str = "default",
    split: str = "train",
):
    """Build the PettingZoo env that shared PPO's SuperSuit stack will wrap.

    Neighbor observations, when enabled, are inside the spatial reward so the
    policy sees the neighbor block and PPO learns from the mixed reward.
    ``alpha`` 0 leaves the local reward unchanged.
    """
    alpha = canonical_alpha(alpha)
    if split == "test":
        raise ValueError("Coordinated training and alpha selection do not use the test split.")
    env = make_env(
        scenario,
        demand=demand,
        split=split,
        seed=seed,
        reward_fn=reward_fn,
        additional_sumo_cmd="--no-step-log --no-warnings",
    )
    neighbor_map = _neighbor_map(env)
    if neighbor_obs:
        env = NeighborObsWrapper(env, neighbor_map)
    return SpatialRewardWrapper(env, neighbor_map, alpha)


def train(
    scenario: str,
    seed: int,
    episodes: int,
    config: dict[str, Any],
    neighbor_obs: bool,
    alpha: float,
    n_envs: int = 1,
    n_cpus: int = 1,
    verbose: int = 1,
    demand: str = "default",
    saved_model: Path | None = None,
    saved_curve: Path | None = None,
    log_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Train one ablation through the shared PPO pipeline."""
    neighbor_obs = parse_neighbor_obs(neighbor_obs)
    alpha = canonical_alpha(alpha)
    if saved_model is None:
        saved_model = model_path(scenario, seed, neighbor_obs, alpha)
    if saved_curve is None:
        saved_curve = curve_path(scenario, seed, neighbor_obs, alpha)
    if log_dir is None:
        log_dir = tensorboard_path(scenario, seed, neighbor_obs, alpha)
    env = make_coordinated_env(
        scenario,
        seed,
        neighbor_obs,
        alpha,
        reward_fn=config["env"]["reward_fn"],
        demand=demand,
        split="train",
    )
    obs = "on" if neighbor_obs else "off"
    return shared_train(
        scenario,
        seed,
        episodes,
        config,
        n_envs=n_envs,
        n_cpus=n_cpus,
        verbose=verbose,
        demand=demand,
        base_env=env,
        saved_model=saved_model,
        saved_curve=saved_curve,
        log_dir=log_dir,
        log_name="coordinated_ppo",
        label=f"coordinated PPO obs={obs} alpha={alpha:g}",
    )


def policy_factory(model_path: str | Path, neighbor_obs: bool):
    """Load a coordinated PPO zip and return ``policy_fn(obs_dict, env)``.

    ``evaluate`` builds a plain ``make_env``. When this model was trained with
    neighbor observations, the policy appends the same neighbor block before
    the shared PPO padding. The spatial reward is a training signal only.
    """
    inner = shared_policy_factory(model_path)
    if not parse_neighbor_obs(neighbor_obs):
        return inner

    cache: dict[int, NeighborObsWrapper] = {}

    def policy_fn(obs_dict: dict, env) -> dict:
        wrapper = cache.get(id(env))
        if wrapper is None:
            wrapper = NeighborObsWrapper(env, _neighbor_map(env))
            cache.clear()
            cache[id(env)] = wrapper
        return inner(wrapper._augment(obs_dict), env)

    return policy_fn


def validation_seeds(config: dict[str, Any], seeds: list[int] | None = None) -> list[int]:
    """Return validation route seeds, rejecting anything else.

    Test seeds raise before a caller can roll them out. Train seeds are not
    validation seeds either.
    """
    if "seeds" not in config or "validation" not in config["seeds"] or "test" not in config["seeds"]:
        raise ValueError("Config is missing seeds.validation or seeds.test.")
    allowed = [int(seed) for seed in config["seeds"]["validation"]]
    test = {int(seed) for seed in config["seeds"]["test"]}
    chosen = allowed if seeds is None else [int(seed) for seed in seeds]
    if not chosen:
        raise ValueError("Alpha selection needs at least one validation seed.")
    overlap = [seed for seed in chosen if seed in test]
    if overlap:
        raise ValueError(f"Alpha selection must not use test seeds, got {overlap}.")
    unknown = [seed for seed in chosen if seed not in set(allowed)]
    if unknown:
        raise ValueError(
            f"Alpha selection only accepts validation route seeds {allowed}, got {unknown}."
        )
    return chosen


def choose_alpha(
    scenario: str,
    train_seed: int,
    episodes: int,
    config: dict[str, Any],
    demand: str = "default",
    alphas: tuple[float, ...] = ALLOWED_ALPHAS,
    seeds: list[int] | None = None,
    score_fn: ScoreFn | None = None,
    frozen_path: Path = FROZEN_PATH,
    verbose: int = 0,
) -> float:
    """Pick the spatial-reward alpha on validation route seeds and save it.

    Each candidate is trained with neighbor observations on, then scored by
    mean ``avg_waiting_time`` on the validation seeds. The lowest wait wins.
    Ties keep the smaller alpha. ``reward_alpha`` is that winner when it is
    positive, otherwise the best alpha among the positive candidates, so the
    reward-only ablation stays at alpha > 0. ``score_fn`` replaces training
    in tests. Test seeds never reach it.
    """
    route_seeds = validation_seeds(config, seeds)
    _check_train_seed(config, train_seed)
    candidates = tuple(canonical_alpha(alpha) for alpha in alphas)
    if not candidates:
        raise ValueError("Alpha selection needs at least one candidate.")
    scores: dict[float, float] = {}
    for alpha in candidates:
        if score_fn is None:
            score = _score_alpha(
                scenario,
                train_seed,
                episodes,
                config,
                demand,
                alpha,
                route_seeds,
                verbose=verbose,
            )
        else:
            score = float(score_fn(alpha, list(route_seeds)))
        if not np.isfinite(score):
            raise RuntimeError(f"Validation score for alpha={alpha:g} is not finite: {score}.")
        scores[alpha] = score
        print(f"alpha={alpha:g} validation avg_waiting_time={score:.6f} seeds={route_seeds}")

    best = min(scores, key=lambda alpha: (scores[alpha], alpha))
    positive = {alpha: score for alpha, score in scores.items() if alpha > 0.0}
    if not positive:
        raise ValueError("Alpha selection needs a candidate greater than 0 for the reward-only ablation.")
    reward_alpha = best if best > 0.0 else min(positive, key=lambda alpha: (positive[alpha], alpha))
    payload = {
        "alpha": best,
        "reward_alpha": reward_alpha,
        "scenario": scenario,
        "demand": demand,
        "train_seed": int(train_seed),
        "episodes": int(episodes),
        "neighbor_obs": "on",
        "validation_seeds": route_seeds,
        "scores": [{"alpha": alpha, "avg_waiting_time": scores[alpha]} for alpha in candidates],
    }
    _write_frozen(payload, frozen_path)
    print(
        f"froze alpha={best:g} reward_alpha={reward_alpha:g} "
        f"from validation seeds {route_seeds} to {frozen_path}"
    )
    return best


def load_frozen(path: Path = FROZEN_PATH, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load ``configs/frozen.yaml`` and reject a file that lists test seeds."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Choose alpha on the validation seeds first, for example "
            "python scripts/train_coordinated_ppo.py --select-alpha --scenario grid4x4 --seed 0 --episodes 8"
        )
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen config must be a mapping: {path}")
    for key in ("alpha", "reward_alpha", "validation_seeds"):
        if key not in payload:
            raise ValueError(f"Frozen config is missing {key}: {path}")
    payload["alpha"] = canonical_alpha(payload["alpha"])
    payload["reward_alpha"] = canonical_alpha(payload["reward_alpha"])
    if payload["reward_alpha"] <= 0.0:
        raise ValueError(f"reward_alpha must be greater than 0, got {payload['reward_alpha']}.")
    seeds = [int(seed) for seed in payload["validation_seeds"]]
    if config is not None:
        validation_seeds(config, seeds)
    elif not seeds:
        raise ValueError(f"Frozen config has no validation seeds: {path}")
    payload["validation_seeds"] = seeds
    return payload


def ablation_settings(
    method: str,
    frozen_path: Path = FROZEN_PATH,
    config: dict[str, Any] | None = None,
) -> tuple[bool, float]:
    """Return ``(neighbor_obs, alpha)`` for a registered ablation name."""
    if method not in ABLATIONS:
        known = ", ".join(ABLATIONS)
        raise ValueError(f"Unknown coordinated ablation {method!r}. Expected one of: {known}.")
    spec = ABLATIONS[method]
    if spec["alpha_key"] is None:
        return spec["neighbor_obs"], 0.0
    frozen = load_frozen(frozen_path, config)
    return spec["neighbor_obs"], frozen[spec["alpha_key"]]


def _score_alpha(
    scenario: str,
    train_seed: int,
    episodes: int,
    config: dict[str, Any],
    demand: str,
    alpha: float,
    route_seeds: list[int],
    verbose: int,
) -> float:
    """Train obs-on PPO at ``alpha`` and return mean validation waiting time."""
    tag = alpha_tag(alpha)
    saved_model = (
        REPO_ROOT
        / "results"
        / "models"
        / "coordinated_ppo"
        / "_alpha_selection"
        / scenario
        / f"alpha{tag}_seed{train_seed}.zip"
    )
    saved_curve = (
        REPO_ROOT
        / "results"
        / "raw"
        / "train_curves"
        / f"alpha_selection_{scenario}_alpha{tag}_seed{train_seed}.csv"
    )
    log_dir = (
        REPO_ROOT
        / "results"
        / "raw"
        / "train_curves"
        / "tensorboard"
        / "coordinated_ppo"
        / "_alpha_selection"
        / scenario
        / f"alpha{tag}_seed{train_seed}"
    )
    train(
        scenario,
        train_seed,
        episodes,
        config,
        neighbor_obs=True,
        alpha=alpha,
        verbose=verbose,
        demand=demand,
        saved_model=saved_model,
        saved_curve=saved_curve,
        log_dir=log_dir,
    )
    policy_fn = policy_factory(saved_model, neighbor_obs=True)
    waits = []
    for route_seed in route_seeds:
        env = make_env(
            scenario,
            demand=demand,
            split="train",
            seed=route_seed,
            reward_fn=config["env"]["reward_fn"],
            additional_sumo_cmd="--no-step-log --no-warnings",
        )
        metrics = run_episode(env, policy_fn, route_seed)
        waits.append(metrics["avg_waiting_time"])
        print(
            f"alpha={alpha:g} validation_seed={route_seed} "
            f"avg_waiting_time={metrics['avg_waiting_time']:.6f}"
        )
    return float(np.mean(waits))


def _neighbor_map(env) -> dict[str, list[str]]:
    sumo_env = _sumo_environment(env)
    return build_neighbor_map(sumo_env._net, list(env.possible_agents))


def _check_train_seed(config: dict[str, Any], train_seed: int) -> None:
    train_seeds = {int(seed) for seed in config["seeds"]["train"]}
    if int(train_seed) not in train_seeds:
        raise ValueError(
            f"Alpha selection trains on one of {sorted(train_seeds)}, got {train_seed}. "
            "Validation and test route seeds stay out of training."
        )


def _write_frozen(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(
        "# Spatial-reward alpha for coordinated PPO.\n"
        "# Chosen on validation route seeds only. Test seeds were not used.\n"
        + body
    )
