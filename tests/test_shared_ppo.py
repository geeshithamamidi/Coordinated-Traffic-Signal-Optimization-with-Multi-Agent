"""Shared PPO writes a model and a waiting-time curve, and acts through policy_factory."""

import shutil

import pandas as pd

from src.agents.shared_ppo import (
    DECISION_STEPS,
    curve_path,
    model_path,
    policy_factory,
    tensorboard_path,
    train,
    training_timesteps,
)
from src.envs.make_env import make_env
from src.utils.config import load_config


def test_timestep_budget_counts_every_agent():
    assert DECISION_STEPS == 720
    assert training_timesteps(100, 1) == 72_000
    assert training_timesteps(300, 16) == 300 * 720 * 16


def test_train_one_episode_and_policy_factory():
    scenario = "single"
    seed = 0
    saved_model = model_path(scenario, seed)
    saved_curve = curve_path(scenario, seed)
    for path in (saved_model, saved_curve):
        if path.exists():
            path.unlink()

    config = load_config()
    env = None
    try:
        model_file, curve_file = train(scenario, seed, episodes=1, config=config, verbose=0)

        assert model_file == saved_model
        assert model_file.is_file()
        frame = pd.read_csv(curve_file)
        assert list(frame.columns) == ["episode", "mean_system_waiting_time"]
        assert len(frame) == 1
        assert frame.loc[0, "episode"] == 1
        assert frame.loc[0, "mean_system_waiting_time"] >= 0

        policy_fn = policy_factory(model_file)
        env = make_env(scenario, seed=seed, additional_sumo_cmd="--no-step-log --no-warnings")
        obs, _info = env.reset(seed=seed)
        for _ in range(5):
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
        for path in (saved_model, saved_curve):
            if path.is_file():
                path.unlink()
        log_dir = tensorboard_path(scenario, seed)
        if log_dir.exists():
            shutil.rmtree(log_dir)
