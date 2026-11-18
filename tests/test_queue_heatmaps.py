"""Queue heatmap placement and cache. The SUMO episode is not run."""

import numpy as np
import pandas as pd
import pytest

from scripts.queue_heatmaps import (
    cache_path,
    grid_from_means,
    intersection_order,
    load_or_run,
    plot_spatial,
    plot_surge,
    signal_coordinates,
    stopped_means,
)
from scripts.run_experiment import _policy_factory


def test_policy_factories_cover_the_heatmap_methods():
    assert _policy_factory("fixed_time", "grid4x4")(0).env_kwargs["fixed_ts"] is True
    assert callable(_policy_factory("max_pressure", "grid4x4")(0))


def test_cache_skips_simulation(tmp_path):
    path = cache_path(tmp_path, "default", "fixed_time", 0, 10)
    pd.DataFrame({"time_s": [0, 5], "A0_stopped": [1.0, 3.0], "agents_total_stopped": [9.0, 9.0]}).to_csv(
        path, index=False
    )

    def runner():
        raise AssertionError("cached episode was simulated again")

    loaded = load_or_run(path, runner)
    assert stopped_means(loaded) == {"A0": 2.0}


def test_grid_uses_sumo_coordinates():
    ids = [f"{column}{row}" for column in "ABCD" for row in range(4)]
    coords = signal_coordinates(ids)
    means = {ts_id: 0.0 for ts_id in ids}
    means["A3"] = 4.0
    means["D0"] = 1.0
    grid, labels = grid_from_means(means, coords)
    assert grid.shape == (4, 4)
    assert labels[0][0] == "A3"
    assert labels[0][-1] == "D3"
    assert labels[-1][0] == "A0"
    assert labels[-1][-1] == "D0"
    assert grid[0, 0] == pytest.approx(4.0)
    assert grid[-1, -1] == pytest.approx(1.0)
    assert intersection_order(coords)[:4] == ["A3", "B3", "C3", "D3"]


def test_plots_share_one_color_scale(tmp_path):
    coords = {
        "A0": (0.0, 0.0),
        "B0": (10.0, 0.0),
        "A1": (0.0, 10.0),
        "B1": (10.0, 10.0),
    }
    low = pd.DataFrame({"time_s": [0, 5], "A1_stopped": [0, 0], "B1_stopped": [0, 0], "A0_stopped": [0, 0], "B0_stopped": [0, 0]})
    high = low.copy()
    high["B0_stopped"] = [10.0, 10.0]
    spatial = tmp_path / "heatmap_grid4x4.png"
    plot_spatial([("fixed_time", low), ("max_pressure", high)], coords, spatial)
    assert spatial.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    surge = tmp_path / "heatmap_surge_grid4x4.png"
    plot_surge(
        [("max_pressure", low), ("coordinated_ppo", high)],
        intersection_order(coords),
        surge,
    )
    assert surge.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.isfinite(high["B0_stopped"]).all()
