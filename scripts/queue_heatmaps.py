#!/usr/bin/env python3
"""Queue heatmaps for grid4x4 from one test episode per controller.

    python scripts/queue_heatmaps.py

Runs test seed 10 for fixed time, max pressure, shared PPO, and coordinated PPO
(train seed 0) on the default demand, plus max pressure and coordinated PPO on
the surge test route. Per-step tables are cached under results/raw/heatmaps/,
and a later run only redraws the figures.

The four default-demand panels share one color scale. Junctions are placed
from their SUMO coordinates, so the letter axis runs west to east and the
number axis runs north to south.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_experiment import _policy_factory  # noqa: E402
from src.envs.make_env import _DELTA_TIME, _SCENARIOS, _ensure_sumo_home, _sumo_rl_api, make_env  # noqa: E402
from src.envs.neighbors import _controlled_nodes  # noqa: E402
from src.evaluation.metrics import run_episode  # noqa: E402
from src.utils.config import load_config  # noqa: E402

SCENARIO = "grid4x4"
TEST_SEED = 10
TRAIN_SEED = 0
METHODS = ("fixed_time", "max_pressure", "shared_ppo", "coordinated_ppo")
SURGE_METHODS = ("max_pressure", "coordinated_ppo")
LABELS = {
    "fixed_time": "Fixed time",
    "max_pressure": "Max-Pressure",
    "shared_ppo": "Shared PPO",
    "coordinated_ppo": "Coordinated PPO",
}
SKIP_STOPPED = {"system_total_stopped", "agents_total_stopped"}
CACHE_DIR = REPO_ROOT / "results" / "raw" / "heatmaps"
FIGURES = REPO_ROOT / "results" / "figures"


def cache_path(directory: Path, demand: str, method: str, train_seed: int, test_seed: int) -> Path:
    return directory / f"{SCENARIO}_{demand}_{method}_train{train_seed}_test{test_seed}.csv"


def load_or_run(path: Path, runner) -> pd.DataFrame:
    """Return the cached per-step table, or run ``runner`` and store it."""
    if path.is_file():
        print(f"using cache {_display(path)}")
        return pd.read_csv(path)
    steps = runner()
    if "time_s" not in steps.columns:
        steps = steps.copy()
        steps.insert(0, "time_s", np.arange(len(steps)) * _DELTA_TIME)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps.to_csv(path, index=False)
    print(f"wrote {_display(path)}")
    return steps


def simulate(method: str, demand: str, config: dict) -> pd.DataFrame:
    """One test episode. ``split='test'`` selects the surge test route."""
    policy = _policy_factory(method, SCENARIO, config)(TRAIN_SEED)
    env = make_env(
        SCENARIO,
        demand=demand,
        split="test",
        seed=TEST_SEED,
        additional_sumo_cmd="--no-step-log --no-warnings",
        **dict(getattr(policy, "env_kwargs", {}) or {}),
    )
    steps = run_episode(env, policy, TEST_SEED)["steps"]
    steps = steps.copy()
    steps.insert(0, "time_s", np.arange(len(steps)) * _DELTA_TIME)
    return steps


def stopped_means(steps: pd.DataFrame) -> dict[str, float]:
    """Time average of each ``<intersection>_stopped`` column."""
    means = {}
    for column in steps.columns:
        if column in SKIP_STOPPED or not str(column).endswith("_stopped"):
            continue
        means[column.removesuffix("_stopped")] = float(steps[column].mean())
    if not means:
        raise ValueError("Per-step table has no <intersection>_stopped columns.")
    return means


def signal_coordinates(ts_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Mean junction coordinate of each signal, read from the grid4x4 net."""
    _ensure_sumo_home()
    import sumolib

    sumo_rl, _, _ = _sumo_rl_api()
    net_file = Path(sumo_rl.__file__).resolve().parent / _SCENARIOS[SCENARIO]["net"]
    net = sumolib.net.readNet(str(net_file))
    coords = {}
    for ts_id in ts_ids:
        nodes = _controlled_nodes(net, ts_id)
        xs = [node.getCoord()[0] for node in nodes]
        ys = [node.getCoord()[1] for node in nodes]
        coords[ts_id] = (float(np.mean(xs)), float(np.mean(ys)))
    return coords


def axis_bins(values: list[float]) -> list[float]:
    """Cluster coordinates that belong to the same row or column."""
    ordered = sorted(float(value) for value in values)
    if len(ordered) <= 1:
        return ordered
    gaps = np.diff(ordered)
    positive = gaps[gaps > 1e-6]
    tolerance = float(positive.min()) / 2.0 if positive.size else 1.0
    bins = [ordered[0]]
    for value in ordered[1:]:
        if value - bins[-1] > tolerance:
            bins.append(value)
    return bins


def _nearest(bins: list[float], value: float) -> int:
    return int(np.argmin([abs(value - center) for center in bins]))


def place(coords: dict[str, tuple[float, float]]) -> tuple[dict[str, tuple[int, int]], list[float], list[float]]:
    """Map each signal to a heatmap cell. Row 0 is the northernmost junction."""
    xs = axis_bins([point[0] for point in coords.values()])
    ys = list(reversed(axis_bins([point[1] for point in coords.values()])))
    cells: dict[str, tuple[int, int]] = {}
    for ts_id, (x_coord, y_coord) in coords.items():
        cell = (_nearest(ys, y_coord), _nearest(xs, x_coord))
        if cell in cells.values():
            other = next(name for name, placed in cells.items() if placed == cell)
            raise ValueError(f"{ts_id} and {other} share a grid cell at {cell}.")
        cells[ts_id] = cell
    return cells, xs, ys


def grid_from_means(
    means: dict[str, float], coords: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, list[list[str]]]:
    cells, xs, ys = place(coords)
    grid = np.full((len(ys), len(xs)), np.nan)
    labels = [["" for _ in xs] for _ in ys]
    missing = [ts_id for ts_id in means if ts_id not in cells]
    if missing:
        raise ValueError(f"No junction coordinate for {', '.join(missing)}.")
    for ts_id, value in means.items():
        row, column = cells[ts_id]
        grid[row, column] = value
        labels[row][column] = ts_id
    return grid, labels


def intersection_order(coords: dict[str, tuple[float, float]]) -> list[str]:
    """North-to-south, then west-to-east."""
    cells, _, _ = place(coords)
    return [ts_id for ts_id, _cell in sorted(cells.items(), key=lambda item: item[1])]


def plot_spatial(frames: list[tuple[str, pd.DataFrame]], coords: dict[str, tuple[float, float]], path: Path) -> None:
    panels = [(method, *grid_from_means(stopped_means(steps), coords)) for method, steps in frames]
    grids = [grid for _method, grid, _labels in panels]
    vmin, vmax = _shared_limits(grids)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6), squeeze=False)
    image = None
    for ax, (method, grid, labels) in zip(axes[0], panels):
        image = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
        _annotate(ax, grid, labels, vmin, vmax)
        ax.set_title(LABELS.get(method, method))
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
    colorbar.set_label("Mean stopped vehicles")
    fig.suptitle(f"{SCENARIO}, default demand, test seed {TEST_SEED}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_surge(frames: list[tuple[str, pd.DataFrame]], order: list[str], path: Path) -> None:
    panels = []
    for method, steps in frames:
        missing = [ts_id for ts_id in order if f"{ts_id}_stopped" not in steps.columns]
        if missing:
            raise ValueError(f"{method} is missing stopped columns for {', '.join(missing)}.")
        values = np.column_stack([steps[f"{ts_id}_stopped"].to_numpy(dtype=float) for ts_id in order])
        time_s = steps["time_s"].to_numpy(dtype=float) if "time_s" in steps.columns else np.arange(len(steps)) * _DELTA_TIME
        panels.append((method, values, time_s))
    vmin, vmax = _shared_limits([values for _method, values, _time in panels])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), squeeze=False)
    image = None
    for ax, (method, values, time_s) in zip(axes[0], panels):
        image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto", origin="upper", interpolation="nearest")
        ax.set_title(LABELS.get(method, method))
        ax.set_xlabel("Intersection")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90)
        _time_ticks(ax, time_s)
        ax.set_ylabel("Time (s)")
    colorbar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
    colorbar.set_label("Stopped vehicles")
    fig.suptitle(f"{SCENARIO}, surge test route, test seed {TEST_SEED}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config()
    default_frames = [
        (method, load_or_run(cache_path(CACHE_DIR, "default", method, TRAIN_SEED, TEST_SEED), lambda method=method: simulate(method, "default", config)))
        for method in METHODS
    ]
    coords = signal_coordinates(list(stopped_means(default_frames[0][1])))
    spatial = FIGURES / "heatmap_grid4x4.png"
    plot_spatial(default_frames, coords, spatial)
    print(f"wrote {_display(spatial)}")

    surge_frames = [
        (method, load_or_run(cache_path(CACHE_DIR, "surge", method, TRAIN_SEED, TEST_SEED), lambda method=method: simulate(method, "surge", config)))
        for method in SURGE_METHODS
    ]
    surge = FIGURES / "heatmap_surge_grid4x4.png"
    plot_surge(surge_frames, intersection_order(coords), surge)
    print(f"wrote {_display(surge)}")


def _annotate(ax, grid: np.ndarray, labels: list[list[str]], vmin: float, vmax: float) -> None:
    midpoint = (vmin + vmax) / 2.0
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            label = labels[row][column]
            if not label:
                continue
            value = grid[row, column]
            color = "white" if np.isfinite(value) and value > midpoint else "black"
            ax.text(column, row, label, ha="center", va="center", color=color, fontsize=8)


def _time_ticks(ax, time_s: np.ndarray) -> None:
    if time_s.size == 0:
        return
    targets = np.arange(0, time_s[-1] + _DELTA_TIME, 600)
    indexes = [int(np.argmin(np.abs(time_s - target))) for target in targets]
    indexes = list(dict.fromkeys(indexes))
    ax.set_yticks(indexes)
    ax.set_yticklabels([str(int(time_s[index])) for index in indexes])


def _shared_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()])
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        high = low + 1.0
    return low, high


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
