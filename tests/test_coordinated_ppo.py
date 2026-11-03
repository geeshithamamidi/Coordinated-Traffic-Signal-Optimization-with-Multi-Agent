"""Coordinated PPO ablations and validation-only alpha selection."""

import shutil

import pandas as pd
import pytest

from src.agents.coordinated_ppo import (
    ablation_settings,
    canonical_alpha,
    choose_alpha,
    curve_path,
    load_frozen,
    make_coordinated_env,
    model_path,
    policy_factory,
    tensorboard_path,
    train,
    validation_seeds,
)
from src.envs.coordination import NeighborObsWrapper, SpatialRewardWrapper
from src.envs.make_env import make_env
from src.utils.config import load_config


def test_canonical_alpha_accepts_the_four_weights():
    assert canonical_alpha(0) == 0.0
    assert canonical_alpha("0.25") == 0.25
    assert canonical_alpha(0.5) == 0.5
    assert canonical_alpha(0.75) == 0.75
    with pytest.raises(ValueError, match="alpha must be one of"):
        canonical_alpha(0.1)


def test_validation_seeds_reject_test_and_train_seeds():
    config = load_config()
    assert validation_seeds(config) == [5, 6, 7, 8, 9]
    with pytest.raises(ValueError, match="must not use test seeds"):
        validation_seeds(config, [5, 10])
    with pytest.raises(ValueError, match="only accepts validation"):
        validation_seeds(config, [0, 5])


def test_choose_alpha_freezes_the_validation_winner(tmp_path):
    config = load_config()
    seen = []

    def score(alpha, seeds):
        seen.append((alpha, list(seeds)))
        return {0.0: 3.0, 0.25: 1.5, 0.5: 2.0, 0.75: 4.0}[alpha]

    frozen = tmp_path / "frozen.yaml"
    best = choose_alpha(
        "grid4x4",
        train_seed=0,
        episodes=4,
        config=config,
        score_fn=score,
        frozen_path=frozen,
    )
    assert best == 0.25
    assert [seeds for _alpha, seeds in seen] == [[5, 6, 7, 8, 9]] * 4
    payload = load_frozen(frozen, config)
    assert payload["alpha"] == 0.25
    assert payload["reward_alpha"] == 0.25
    assert payload["validation_seeds"] == [5, 6, 7, 8, 9]
    assert ablation_settings("coordinated_ppo", frozen, config) == (True, 0.25)
    assert ablation_settings("obs_only", frozen, config) == (True, 0.0)
    assert ablation_settings("reward_only", frozen, config) == (False, 0.25)


def test_reward_alpha_stays_positive_when_zero_wins(tmp_path):
    config = load_config()

    def score(alpha, seeds):
        assert seeds == [5, 6, 7, 8, 9]
        return {0.0: 0.1, 0.25: 2.0, 0.5: 1.0, 0.75: 3.0}[alpha]

    frozen = tmp_path / "frozen.yaml"
    best = choose_alpha(
        "grid4x4",
        train_seed=1,
        episodes=2,
        config=config,
        score_fn=score,
        frozen_path=frozen,
    )
    assert best == 0.0
    payload = load_frozen(frozen, config)
    assert payload["reward_alpha"] == 0.5
    assert ablation_settings("reward_only", frozen, config) == (False, 0.5)
    assert ablation_settings("coordinated_ppo", frozen, config) == (True, 0.0)


def test_choose_alpha_refuses_a_test_train_seed():
    config = load_config()
    with pytest.raises(ValueError, match="trains on one of"):
        choose_alpha(
            "grid4x4",
            train_seed=10,
            episodes=1,
            config=config,
            score_fn=lambda alpha, seeds: 1.0,
        )


def test_neighbor_obs_wrapper_is_inside_the_spatial_reward():
    env = make_coordinated_env("single", seed=0, neighbor_obs=True, alpha=0.0)
    plain = None
    try:
        assert isinstance(env, SpatialRewardWrapper)
        assert isinstance(env.env, NeighborObsWrapper)
        assert env.alpha == 0.0
        plain = make_env("single", seed=0, additional_sumo_cmd="--no-step-log --no-warnings")
        agent = env.possible_agents[0]
        assert env.observation_space(agent).shape[0] > plain.observation_space(agent).shape[0]
    finally:
        env.close()
        if plain is not None:
            plain.close()


def test_train_one_episode_and_policy_uses_neighbor_block():
    scenario = "single"
    seed = 0
    saved_model = model_path(scenario, seed, True, 0.0)
    config = load_config()
    env = None
    try:
        model_file, curve_file = train(
            scenario,
            seed,
            episodes=1,
            config=config,
            neighbor_obs=True,
            alpha=0.0,
            verbose=0,
        )
        frame = pd.read_csv(curve_file)
        assert list(frame.columns) == ["episode", "mean_system_waiting_time"]
        assert len(frame) == 1

        policy_fn = policy_factory(model_file, neighbor_obs=True)
        env = make_env(scenario, seed=seed, additional_sumo_cmd="--no-step-log --no-warnings")
        obs, _info = env.reset(seed=seed)
        for _ in range(3):
            actions = policy_fn(obs, env)
            assert set(actions) == set(env.agents)
            for agent, action in actions.items():
                assert env.action_space(agent).contains(action)
            obs, _reward, terminated, truncated, _info = env.step(actions)
            if all(terminated.values()) or all(truncated.values()):
                break
    finally:
        if env is not None:
            env.close()
        if saved_model.is_file():
            saved_model.unlink()
        curve = curve_path(scenario, seed, True, 0.0)
        if curve.is_file():
            curve.unlink()
        log_dir = tensorboard_path(scenario, seed, True, 0.0)
        if log_dir.exists():
            shutil.rmtree(log_dir)
