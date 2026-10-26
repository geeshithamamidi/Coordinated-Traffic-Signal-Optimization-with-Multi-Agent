"""Job-list cross product and experiment dispatch."""

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.make_jobs import job_lines, write_jobs
from scripts.run_experiment import episode_budget, resolve_trainer


def test_job_lines_are_the_cross_product(tmp_path: Path):
    spec = {
        "methods": ["ql", "shared_ppo"],
        "scenarios": ["single", "grid4x4"],
        "demands": ["default"],
        "seeds": {"train": [0, 1]},
    }
    lines = job_lines(spec)
    assert lines == [
        "ql single default 0",
        "ql single default 1",
        "ql grid4x4 default 0",
        "ql grid4x4 default 1",
        "shared_ppo single default 0",
        "shared_ppo single default 1",
        "shared_ppo grid4x4 default 0",
        "shared_ppo grid4x4 default 1",
    ]
    output = tmp_path / "jobs.txt"
    write_jobs(spec, output)
    assert output.read_text().splitlines() == lines


def test_episode_budget_prefers_the_sweep_yaml(tmp_path: Path):
    jobs = tmp_path / "experiments.yaml"
    jobs.write_text("episodes:\n  single: 3\n")
    assert episode_budget("single", jobs, None) == 3
    assert episode_budget("single", jobs, 12) == 12
    assert episode_budget("grid4x4", Path("missing.yaml"), None) == 300


def test_resolve_trainer_names():
    assert resolve_trainer("shared_ppo").__name__ == "_train_shared_ppo"
    with pytest.raises(NotImplementedError, match="QL trainer"):
        resolve_trainer("ql")(None, {"ql": {"alpha": 0.1, "gamma": 0.99, "epsilon_start": 0.05}}, 1)
    with pytest.raises(NotImplementedError, match="neighbor_alpha=0.25"):
        resolve_trainer("coordinated_0.25")(
            Namespace(scenario="single"),
            {"coordinated": {"neighbor_alpha": [0.25]}},
            1,
        )
    with pytest.raises(ValueError, match="Unknown method"):
        resolve_trainer("fixed_time")
