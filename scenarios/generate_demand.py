#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.make_env import _NUM_SECONDS, _SCENARIOS, _sumo_rl_api, make_env  # noqa: E402

LEVELS = {"light": 0.5, "medium": 1.0, "heavy": 1.5}
# surge: medium demand plus an extra stream at twice the medium rate during
# [SURGE_WINDOW) seconds of the episode. unbalanced: medium demand with
# WEST_SHARE of trips starting on west-side edges and the rest elsewhere.
EXTRA_LEVELS = ("surge", "unbalanced")
ALL_LEVELS = [*LEVELS, *EXTRA_LEVELS]
SURGE_WINDOW = (1200, 2100)
WEST_SHARE = 0.7
EXTRA_SEED_SHIFT = 500
SPLIT_SEED_OFFSET = {"train": 0, "test": 1000}
SCENARIO_DIR = REPO_ROOT / "scenarios"


def upstream_files(scenario: str) -> tuple[Path, Path, int]:
    sumo_rl, _, _ = _sumo_rl_api()
    spec = _SCENARIOS[scenario]
    root = Path(sumo_rl.__file__).resolve().parent
    return root / spec["net"], root / spec["route"], int(spec["begin_time"])


def _flow_count(flow: ET.Element, start: float, end: float) -> float:
    begin = max(float(flow.get("begin", 0)), start)
    finish = min(float(flow.get("end", end)), end)
    duration = max(0.0, finish - begin)
    if duration == 0:
        return 0.0
    if flow.get("number") is not None:
        total = max(1e-9, float(flow.get("end", end)) - float(flow.get("begin", 0)))
        return float(flow.get("number")) * duration / total
    if flow.get("vehsPerHour") is not None:
        return float(flow.get("vehsPerHour")) * duration / 3600.0
    if flow.get("period") is not None:
        return duration / float(flow.get("period"))
    if flow.get("probability") is not None:
        return float(flow.get("probability")) * duration
    return 0.0


def count_vehicles(route_file: Path, begin_time: int, horizon: int = _NUM_SECONDS) -> float:
    start, end = float(begin_time), float(begin_time + horizon)
    total = 0.0
    for element in ET.parse(route_file).getroot():
        tag = element.tag.split("}")[-1]
        if tag in ("vehicle", "trip"):
            depart = element.get("depart", "0")
            try:
                value = float(depart)
            except ValueError:
                value = start
            if start <= value < end:
                total += 1
        elif tag == "flow":
            total += _flow_count(element, start, end)
    return total


def random_trips(
    net_file: Path,
    out_file: Path,
    begin: int,
    end: int,
    period: float,
    seed: int,
    weights_prefix: Path | None = None,
) -> None:
    script = Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        trips = Path(tmp) / "trips.xml"
        cmd = [
            sys.executable,
            str(script),
            "-n", str(net_file),
            "-o", str(trips),
            "-r", str(out_file),
            "-b", str(begin),
            "-e", str(end),
            "-p", f"{period:.4f}",
            "--seed", str(seed),
            "--fringe-factor", "10",
            "--validate",
            "--prefix", f"s{seed}_",
        ]
        if weights_prefix is not None:
            cmd += ["--weights-prefix", str(weights_prefix)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    for extra in out_file.parent.glob(out_file.stem.replace(".rou", "") + "*.alt.xml"):
        extra.unlink()


def merge_route_files(parts: list[Path], out_file: Path) -> None:
    """Merge randomTrips route files into one, vehicles sorted by depart time."""
    vtypes: dict[str, ET.Element] = {}
    others, vehicles = [], []
    for part in parts:
        for element in ET.parse(part).getroot():
            tag = element.tag.split("}")[-1]
            if tag == "vType":
                vtypes.setdefault(element.get("id"), element)
            elif tag in ("vehicle", "trip", "flow"):
                vehicles.append(element)
            else:
                others.append(element)
    vehicles.sort(key=lambda e: float(e.get("depart", e.get("begin", 0))))
    root = ET.Element("routes")
    root.extend([*vtypes.values(), *others, *vehicles])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_file, encoding="UTF-8", xml_declaration=True)


def write_origin_weights(net_file: Path, prefix: Path, west: bool) -> int:
    """randomTrips source weights: 1 for west-side (or non-west) passenger edges, 0 otherwise.

    An edge is west-side when its start node lies in the western quarter of the
    network's x-range. Returns the number of edges with weight 1.
    """
    import sumolib

    net = sumolib.net.readNet(str(net_file))
    xs = [node.getCoord()[0] for node in net.getNodes()]
    cut = min(xs) + 0.25 * (max(xs) - min(xs))
    lines = ["<edgedata>", '  <interval begin="0" end="1000000">']
    selected = 0
    for edge in net.getEdges():
        if not edge.allows("passenger"):
            continue
        is_west = edge.getFromNode().getCoord()[0] <= cut
        value = 1.0 if is_west == west else 0.0
        selected += int(value > 0)
        lines.append(f'    <edge id="{edge.getID()}" value="{value}"/>')
    lines += ["  </interval>", "</edgedata>"]
    Path(f"{prefix}.src.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if selected == 0:
        raise RuntimeError(f"No {'west' if west else 'non-west'} origin edges found in {net_file}")
    return selected


def build_level(net_file: Path, out_file: Path, level: str, begin_time: int, default_count: float, seed: int) -> float:
    """Write one route file for `level` and return its target vehicle count."""
    end_time = begin_time + _NUM_SECONDS
    if level in LEVELS:
        target = default_count * LEVELS[level]
        random_trips(net_file, out_file, begin_time, end_time, _NUM_SECONDS / target, seed)
        return target

    medium_period = _NUM_SECONDS / default_count
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base, extra = tmp / "base.rou.xml", tmp / "extra.rou.xml"
        if level == "surge":
            start, stop = SURGE_WINDOW
            random_trips(net_file, base, begin_time, end_time, medium_period, seed)
            random_trips(net_file, extra, begin_time + start, begin_time + stop, medium_period / 2, seed + EXTRA_SEED_SHIFT)
            target = default_count + 2 * default_count * (stop - start) / _NUM_SECONDS
        elif level == "unbalanced":
            write_origin_weights(net_file, tmp / "west", west=True)
            write_origin_weights(net_file, tmp / "rest", west=False)
            random_trips(net_file, base, begin_time, end_time, medium_period / WEST_SHARE, seed, tmp / "west")
            random_trips(
                net_file, extra, begin_time, end_time, medium_period / (1 - WEST_SHARE), seed + EXTRA_SEED_SHIFT, tmp / "rest"
            )
            target = default_count
        else:
            raise ValueError(f"Unknown demand level {level!r}")
        merge_route_files([base, extra], out_file)
    return target


def generate(scenario: str, levels: list[str]) -> list[dict]:
    net_file, route_file, begin_time = upstream_files(scenario)
    default_count = count_vehicles(route_file, begin_time)
    print(f"{scenario}: default route file has {default_count:.0f} vehicles in {_NUM_SECONDS} s")
    rows = []
    for level in levels:
        index = ALL_LEVELS.index(level)  # seeds stay fixed whatever subset is generated
        for split, offset in SPLIT_SEED_OFFSET.items():
            seed = offset + index
            out_file = SCENARIO_DIR / scenario / f"{level}_{split}.rou.xml"
            target = build_level(net_file, out_file, level, begin_time, default_count, seed)
            actual = count_vehicles(out_file, begin_time)
            rows.append(
                {"scenario": scenario, "level": level, "split": split, "target": target, "actual": actual, "seed": seed}
            )
            print(f"  {out_file.name}: target {target:.0f}, generated {actual:.0f}, seed {seed}")
    return rows


def verify(scenario: str, levels: list[str], steps: int = 50) -> None:
    for level in levels:
        for split in SPLIT_SEED_OFFSET:
            env = make_env(scenario, demand=level, split=split, seed=0)
            env.reset(seed=0)
            for _ in range(steps):
                if not env.agents:
                    break
                env.step({agent: env.action_space(agent).sample() for agent in env.agents})
            env.close()
            print(f"  verified {scenario} {level}_{split} for {steps} steps")


def plot_profiles(scenarios: list[str], levels: tuple[str, ...] = ("medium", "heavy", "surge")) -> Path:
    """Departures per minute of the train route files, one panel per scenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = REPO_ROOT / "results" / "figures" / "demand_profiles.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 3.4), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        begin_time = upstream_files(scenario)[2]
        for level in levels:
            path = SCENARIO_DIR / scenario / f"{level}_train.rou.xml"
            if not path.is_file():
                continue
            departs = [
                float(e.get("depart")) - begin_time
                for e in ET.parse(path).getroot()
                if e.tag.split("}")[-1] in ("vehicle", "trip") and e.get("depart") is not None
            ]
            minutes = np.array([d // 60 for d in departs if 0 <= d < _NUM_SECONDS], dtype=int)
            ax.plot(np.bincount(minutes, minlength=_NUM_SECONDS // 60), label=level)
        ax.axvspan(SURGE_WINDOW[0] / 60, SURGE_WINDOW[1] / 60, color="grey", alpha=0.15, label="surge window")
        ax.set_title(scenario)
        ax.set_xlabel("Minute of episode")
        ax.set_ylabel("Departures per minute")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return out


def write_readme(rows: list[dict]) -> None:
    lines = [
        "# Scenarios",
        "",
        "Generated by `python scenarios/generate_demand.py --scenarios single grid4x4`.",
        "Counts are vehicles departing in the 3600 s episode (vehicles/hour).",
        "",
        f"- `surge`: medium demand plus an extra stream at twice the medium rate between "
        f"{SURGE_WINDOW[0]} s and {SURGE_WINDOW[1]} s of the episode.",
        f"- `unbalanced`: medium demand with about {WEST_SHARE:.0%} of trips starting on west-side "
        "edges (start node in the western quarter of the network) and the rest elsewhere.",
        "",
        "| scenario | demand | split | target veh/h | generated veh/h | seed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['level']} | {row['split']} | {row['target']:.0f} | {row['actual']:.0f} | {row['seed']} |"
        )
    (SCENARIO_DIR / "README.md").write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate light/medium/heavy/surge/unbalanced route files with randomTrips.")
    parser.add_argument("--scenarios", nargs="+", default=["single", "grid4x4"])
    parser.add_argument("--levels", nargs="+", default=list(ALL_LEVELS), choices=list(ALL_LEVELS))
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _sumo_rl_api()
    rows = []
    for scenario in args.scenarios:
        rows.extend(generate(scenario, args.levels))
        if not args.skip_verify:
            verify(scenario, args.levels)
    write_readme(rows)
    plot_profiles(args.scenarios)


if __name__ == "__main__":
    main()
