import pytest

from src.baselines.fixed_time import FixedCycleController, FixedTimeController
from src.envs.make_env import make_env
from src.evaluation.metrics import run_episode


@pytest.mark.sumo
def test_fixed_time_runs_with_sumo_program():
    env = make_env("single", seed=0, fixed_ts=True)
    result = run_episode(env, FixedTimeController(), 0)
    assert result["throughput"] > 0


@pytest.mark.sumo
def test_fixed_cycle_cycles_through_phases():
    env = make_env("single", seed=0)
    result = run_episode(env, FixedCycleController(20), 0)
    assert result["avg_waiting_time"] >= 0


def test_fixed_cycle_rejects_bad_green():
    with pytest.raises(ValueError):
        FixedCycleController(0)
