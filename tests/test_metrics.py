"""Evaluation writes the same summary columns for every controller."""

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import SUMMARY_COLUMNS, evaluate

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _random_policy(train_seed):
    del train_seed

    def policy(obs, env):
        del obs
        return {agent: env.action_space(agent).sample() for agent in env.agents}

    return policy


def test_evaluate_random_policy_writes_summary_columns():
    method = "random_policy_test"
    csv_path = _REPO_ROOT / "results" / "raw" / "single" / "default" / f"{method}.csv"
    if csv_path.exists():
        csv_path.unlink()

    frame = evaluate(
        method_name=method,
        scenario="single",
        demand="default",
        policy_factory=_random_policy,
        train_seed=0,
        test_seeds=[10],
    )

    assert list(frame.columns) == SUMMARY_COLUMNS
    saved = pd.read_csv(csv_path)
    assert list(saved.columns) == SUMMARY_COLUMNS
    assert len(saved) == 1
    assert saved.loc[0, "method"] == method
    assert saved.loc[0, "scenario"] == "single"
    assert saved.loc[0, "demand"] == "default"
    assert saved.loc[0, "train_seed"] == 0
    assert saved.loc[0, "test_seed"] == 10
    assert saved.loc[0, "avg_waiting_time"] >= 0
    assert saved.loc[0, "throughput"] > 0
    assert saved.loc[0, "avg_trip_duration"] > 0
    assert saved.loc[0, "avg_time_loss"] >= 0
