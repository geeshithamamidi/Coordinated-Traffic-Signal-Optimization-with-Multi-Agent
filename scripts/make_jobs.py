#!/usr/bin/env python3
"""Write hpc/jobs.txt as methods x scenarios x demands x seeds.

The YAML lists those four fields. ``seeds`` may be a list, or a mapping with a
``train`` list (the shape used in configs/base.yaml).

    python scripts/make_jobs.py
    python scripts/make_jobs.py --config hpc/experiments.yaml --output hpc/jobs.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "hpc" / "experiments.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "hpc" / "jobs.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a sweep YAML into hpc/jobs.txt.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Sweep YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Job list to write")
    return parser.parse_args(argv)


def load_sweep(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError(f"Sweep file must be a mapping: {path}")
    return spec


def sweep_seeds(spec: dict[str, Any]) -> list[int]:
    seeds = spec.get("seeds")
    if isinstance(seeds, dict):
        seeds = seeds.get("train")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("YAML seeds must be a non-empty list, or a mapping with a train list.")
    return [int(seed) for seed in seeds]


def sweep_values(spec: dict[str, Any], key: str) -> list[str]:
    values = spec.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"YAML {key} must be a non-empty list.")
    rendered = [str(value) for value in values]
    for value in rendered:
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"{key} entries cannot contain whitespace: {value!r}")
    return rendered


def job_lines(spec: dict[str, Any]) -> list[str]:
    """Cross product in method, scenario, demand, seed order."""
    methods = sweep_values(spec, "methods")
    scenarios = sweep_values(spec, "scenarios")
    demands = sweep_values(spec, "demands")
    seeds = sweep_seeds(spec)
    return [
        f"{method} {scenario} {demand} {seed}"
        for method in methods
        for scenario in scenarios
        for demand in demands
        for seed in seeds
    ]


def write_jobs(spec: dict[str, Any], output: Path) -> list[str]:
    lines = job_lines(spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return lines


def main(argv: list[str] | None = None) -> list[str]:
    args = parse_args(argv)
    lines = write_jobs(load_sweep(args.config), args.output)
    print(f"wrote {len(lines)} jobs to {args.output}")
    return lines


if __name__ == "__main__":
    main()
