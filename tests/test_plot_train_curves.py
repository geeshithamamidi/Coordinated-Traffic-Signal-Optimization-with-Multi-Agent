"""Training-curve figure from fake CSVs. No SUMO."""

import numpy as np
import pandas as pd
import pytest

from scripts.plot_train_curves import (
    load_method,
    main,
    max_pressure_mean,
    parse_args,
    smooth_series,
    summarize_curves,
)
from src.evaluation.plot_style import color_for
from src.evaluation.stats import ci95


def _curve(folder, method, scenario, seed, values):
    pd.DataFrame(
        {"episode": list(range(1, len(values) + 1)), "mean_system_waiting_time": values}
    ).to_csv(folder / f"{method}_{scenario}_seed{seed}.csv", index=False)


def test_default_out_path_uses_the_scenario():
    args = parse_args(["--scenario", "grid4x4", "--method", "ql"])
    assert args.out.name == "curves_grid4x4.png"
    assert args.smooth == 5


def test_smooth_window_is_centered():
    smoothed = smooth_series(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    np.testing.assert_allclose(smoothed, [1.5, 2.0, 3.0, 4.0, 4.5])
    np.testing.assert_allclose(smooth_series(np.array([1.0, 2.0]), 1), [1.0, 2.0])


def test_mean_and_interval_across_seeds(tmp_path):
    _curve(tmp_path, "ql", "single", 0, [1.0, 2.0, 3.0])
    _curve(tmp_path, "ql", "single", 1, [3.0, 4.0, 5.0])
    loaded = load_method(tmp_path, "ql", "single", window=1)
    summary = summarize_curves(loaded)
    np.testing.assert_allclose(summary["mean"], [2.0, 3.0, 4.0])
    assert summary["half"].iloc[0] == pytest.approx(ci95([1.0, 3.0]))
    assert int(loaded["seed"].nunique()) == 2


def test_missing_curves_are_printed_and_do_not_fail(tmp_path, capsys):
    out = tmp_path / "curves_single.png"
    main(
        [
            "--scenario",
            "single",
            "--method",
            "ql",
            "shared_ppo",
            "--curves-dir",
            str(tmp_path / "curves"),
            "--raw",
            str(tmp_path / "raw"),
            "--out",
            str(out),
            "--smooth",
            "1",
        ]
    )
    text = capsys.readouterr().out
    assert "missing ql curves" in text
    assert "missing shared_ppo curves" in text
    assert "missing" in text and "max_pressure.csv" in text
    assert "no training curves found" in text
    assert not out.exists()


def test_figure_uses_shared_colors_and_the_max_pressure_mean(tmp_path, capsys):
    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves, "ql", "single", 0, [1.0, 2.0, 3.0])
    _curve(curves, "ql", "single", 1, [3.0, 4.0, 5.0])
    _curve(curves, "shared_ppo", "single", 0, [8.0, 7.0, 6.0])
    raw = tmp_path / "raw" / "single" / "default"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "method": ["max_pressure", "max_pressure"],
            "train_seed": [0, 0],
            "avg_waiting_time": [1.0, 3.0],
        }
    ).to_csv(raw / "max_pressure.csv", index=False)
    assert max_pressure_mean(raw / "max_pressure.csv") == pytest.approx(2.0)
    out = tmp_path / "curves_single.png"
    main(
        [
            "--scenario",
            "single",
            "--method",
            "ql",
            "shared_ppo",
            "--curves-dir",
            str(curves),
            "--raw",
            str(tmp_path / "raw"),
            "--out",
            str(out),
            "--smooth",
            "1",
        ]
    )
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "max_pressure test mean=2.000000" in capsys.readouterr().out
    assert color_for("ql") != color_for("shared_ppo")
    assert color_for("max_pressure") == "#E69F00"
