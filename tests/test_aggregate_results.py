"""aggregate_results must include a row for every CSV in results/raw/<scenario>/default/."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import SUMMARY_COLUMNS  # noqa: E402
from src.evaluation.plot_style import METHOD_LABELS  # noqa: E402

METHODS = list(METHOD_LABELS)


def _load_script():
    spec = importlib.util.spec_from_file_location("aggregate_results", REPO_ROOT / "scripts" / "aggregate_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_csv(folder: Path, method: str, rng) -> None:
    rows = []
    for train_seed in (0, 1, 2):
        for test_seed in range(10, 15):
            rows.append(
                {
                    "method": method,
                    "scenario": "single",
                    "demand": "default",
                    "train_seed": train_seed,
                    "test_seed": test_seed,
                    "avg_waiting_time": rng.uniform(1, 10),
                    "avg_queue": rng.uniform(1, 10),
                    "avg_trip_duration": rng.uniform(30, 90),
                    "avg_time_loss": rng.uniform(1, 10),
                    "throughput": rng.uniform(1000, 2500),
                    "avg_speed": rng.uniform(2, 10),
                    "teleports": 0,
                }
            )
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(folder / f"{method}.csv", index=False)


def test_every_method_appears_with_readable_label(tmp_path, monkeypatch):
    module = _load_script()
    raw = tmp_path / "raw"
    folder = raw / "single" / "default"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for method in METHODS:
        _fake_csv(folder, method, rng)
    # A non-result CSV in the same folder must be ignored, not crash.
    pd.DataFrame({"cycle": [30, 40], "wait": [1.0, 2.0]}).to_csv(folder / "fixed_cycle_tuning.csv", index=False)

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)  # the final print uses relative_to(REPO_ROOT)
    monkeypatch.setattr(module, "RAW", raw)
    monkeypatch.setattr(module, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(module, "FIGURES", tmp_path / "figures")
    module.main(["--scenario", "single"])

    text = (tmp_path / "tables" / "main_single.tex").read_text()
    for method, label in METHOD_LABELS.items():
        assert label in text, f"{method} ({label}) missing from table"
    assert "fixed\\_time" not in text and "ql &" not in text
    assert (tmp_path / "figures" / "bars_single.png").is_file()
