#!/usr/bin/env python3
"""Mean training curves with a 95% t-interval across seeds.

    python scripts/plot_train_curves.py --scenario grid4x4 --method ql dqn ippo shared_ppo

Reads results/raw/train_curves/<method>_<scenario>_seed*.csv
(columns episode, mean_system_waiting_time). A method with no files is printed
and skipped. Max-Pressure's test mean, when
results/raw/<scenario>/default/max_pressure.csv exists, is a dashed line.
"""

from __future__ import annotations

import argparse
import re
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

from src.evaluation.plot_style import color_for  # noqa: E402
from src.evaluation.stats import ci95  # noqa: E402

RAW = REPO_ROOT / "results" / "raw"
FIGURES = REPO_ROOT / "results" / "figures"
CURVE_COLUMNS = ("episode", "mean_system_waiting_time")
SEED_NAME = re.compile(r"_seed(?P<seed>\d+)\.csv$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mean training curves across seeds.")
    parser.add_argument("--method", nargs="+", required=True, help="One or more method names.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="Figure path. Defaults to results/figures/curves_<scenario>.png.",
    )
    parser.add_argument("--smooth", type=int, default=5, help="Rolling window applied per seed. Default 5.")
    parser.add_argument("--curves-dir", type=Path, default=RAW / "train_curves")
    parser.add_argument("--raw", type=Path, default=RAW, help="Root of evaluate() CSVs.")
    args = parser.parse_args(argv)
    if args.smooth < 1:
        raise ValueError(f"--smooth must be at least 1, got {args.smooth}.")
    if args.out is None:
        args.out = FIGURES / f"curves_{args.scenario}.png"
    return args


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean. A window of 1 leaves the series unchanged."""
    if window <= 1 or values.size == 0:
        return np.asarray(values, dtype=float)
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def load_method(curves_dir: Path, method: str, scenario: str, window: int) -> pd.DataFrame | None:
    """One row per episode and seed, after the per-seed rolling mean."""
    pattern = f"{method}_{scenario}_seed*.csv"
    paths = [path for path in sorted(curves_dir.glob(pattern)) if SEED_NAME.search(path.name)]
    if not paths:
        print(f"missing {method} curves matching {pattern} in {_display(curves_dir)}")
        return None
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = [column for column in CURVE_COLUMNS if column not in frame.columns]
        if missing:
            print(f"{_display(path)} is missing columns: {', '.join(missing)}")
            continue
        ordered = frame.sort_values("episode")
        frames.append(
            pd.DataFrame(
                {
                    "episode": ordered["episode"].to_numpy(),
                    "waiting": smooth_series(ordered["mean_system_waiting_time"].to_numpy(dtype=float), window),
                    "seed": int(SEED_NAME.search(path.name).group("seed")),
                }
            )
        )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def summarize_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% t-interval half-width of the smoothed curves at each episode."""
    rows = []
    for episode, group in frame.groupby("episode"):
        values = group["waiting"].to_numpy(dtype=float)
        rows.append(
            {
                "episode": episode,
                "mean": float(np.mean(values)),
                "half": ci95(values),
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows).sort_values("episode")


def max_pressure_mean(path: Path) -> float | None:
    """Mean test ``avg_waiting_time`` for Max-Pressure. None when the file is absent."""
    if not path.is_file():
        print(f"missing {_display(path)}")
        return None
    frame = pd.read_csv(path)
    if "avg_waiting_time" not in frame.columns or frame.empty:
        print(f"{_display(path)} has no avg_waiting_time rows")
        return None
    if "train_seed" in frame.columns and frame["train_seed"].nunique() > 1:
        values = frame.groupby("train_seed")["avg_waiting_time"].mean()
    else:
        values = frame["avg_waiting_time"]
    return float(values.mean())


def save_figure(
    summaries: list[tuple[str, pd.DataFrame, int]],
    scenario: str,
    reference: float | None,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for index, (method, summary, n_seeds) in enumerate(summaries):
        color = color_for(method, index)
        episode = summary["episode"].to_numpy()
        mean = summary["mean"].to_numpy()
        half = summary["half"].to_numpy(dtype=float)
        ax.plot(episode, mean, color=color, label=f"{method} (n={n_seeds})")
        ax.fill_between(
            episode,
            mean - half,
            mean + half,
            where=np.isfinite(half),
            color=color,
            alpha=0.2,
        )
    if reference is not None:
        ax.axhline(
            reference,
            color=color_for("max_pressure"),
            linestyle="--",
            label="Max-Pressure",
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean waiting time (s)")
    ax.set_title(f"{scenario} training curves, mean ± 95% CI")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = []
    for method in args.method:
        loaded = load_method(args.curves_dir, method, args.scenario, args.smooth)
        if loaded is None:
            continue
        summaries.append((method, summarize_curves(loaded), int(loaded["seed"].nunique())))
    reference_path = args.raw / args.scenario / "default" / "max_pressure.csv"
    reference = max_pressure_mean(reference_path)
    if reference is not None:
        print(f"max_pressure test mean={reference:.6f}")
    if not summaries:
        print("no training curves found, so no figure was written.")
        return
    save_figure(summaries, args.scenario, reference, args.out)
    print(f"wrote {_display(args.out)}")


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
