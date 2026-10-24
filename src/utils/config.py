"""Load configs/base.yaml and deep-merge a CLI override file."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` applied. Nested mappings merge; other values replace."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(override_path: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/base.yaml`` and deep-merge ``override_path`` when given."""
    with BASE_CONFIG_PATH.open() as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, dict):
        raise ValueError(f"Base config must be a mapping: {BASE_CONFIG_PATH}")
    if override_path is None:
        return base

    path = Path(override_path)
    with path.open() as handle:
        override = yaml.safe_load(handle) or {}
    if not isinstance(override, dict):
        raise ValueError(f"Override file must be a mapping: {path}")
    return deep_merge(base, override)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load configs/base.yaml and merge an override YAML file.")
    parser.add_argument("--config", help="YAML file merged over configs/base.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)
    print(yaml.safe_dump(config, sort_keys=False), end="")
    return config


if __name__ == "__main__":
    main()
