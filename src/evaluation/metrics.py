"""Shared episode metrics for rule-based and learned controllers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.envs.make_env import make_env

SUMMARY_COLUMNS = [
    "method",
    "scenario",
    "demand",
    "train_seed",
    "test_seed",
    "avg_waiting_time",
    "avg_queue",
    "avg_trip_duration",
    "avg_time_loss",
    "throughput",
    "avg_speed",
    "teleports",
]

_STEP_COLUMNS = [
    "system_mean_waiting_time",
    "system_total_stopped",
    "system_total_arrived",
    "system_mean_speed",
    "system_total_teleported",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

PolicyFn = Callable[[dict, Any], dict]
PolicyFactory = Callable[[int], PolicyFn]


def run_episode(env, policy_fn: PolicyFn, seed: int) -> dict[str, Any]:
    """Roll out ``policy_fn`` for one full episode and summarize the info dict.

    ``policy_fn(obs_dict, env)`` returns an action dict. No learning happens
    here. The environment is closed so SUMO flushes ``--tripinfo-output``.
    """
    sumo_env = _sumo_environment(env)
    _attach_vehicle_counters(sumo_env)
    tripinfo_path = _tripinfo_path(sumo_env)

    reset_out = env.reset(seed=seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    rows: list[dict[str, Any]] = []
    try:
        while env.agents:
            actions = policy_fn(obs, env)
            obs, _reward, terminated, truncated, info = env.step(actions)
            rows.append(_step_record(info, sumo_env))
            if _episode_done(terminated, truncated):
                break
    finally:
        env.close()

    steps = _steps_frame(rows)
    avg_trip_duration, avg_time_loss = _trip_means(tripinfo_path)
    final = steps.iloc[-1] if not steps.empty else {}
    return {
        "avg_waiting_time": _column_mean(steps, "system_mean_waiting_time"),
        "avg_queue": _column_mean(steps, "system_total_stopped"),
        "throughput": _final_value(final, "system_total_arrived"),
        "avg_speed": _column_mean(steps, "system_mean_speed"),
        "teleports": _final_value(final, "system_total_teleported"),
        "avg_trip_duration": avg_trip_duration,
        "avg_time_loss": avg_time_loss,
        "steps": steps,
    }


def evaluate(
    method_name: str,
    scenario: str,
    demand: str,
    policy_factory: PolicyFactory,
    train_seed: int,
    test_seeds: list[int],
    env_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate one trained controller on each test seed and append a summary CSV.

    ``policy_factory(train_seed)`` returns ``policy_fn(obs_dict, env) -> action_dict``.
    Each episode uses ``make_env(..., split="test")`` and the fixed 3600 s horizon.
    Rows are appended to ``results/raw/<scenario>/<demand>/<method>.csv``.
    """
    policy_fn = policy_factory(train_seed)
    rows: list[dict[str, Any]] = []
    csv_path = _summary_path(scenario, demand, method_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for test_seed in test_seeds:
        tripinfo_path = csv_path.parent / f"{method_name}_seed{test_seed}_tripinfo.xml"
        env = make_env(
            scenario=scenario,
            demand=demand,
            split="test",
            seed=test_seed,
            parallel=True,
            use_gui=False,
            additional_sumo_cmd=f"--tripinfo-output {tripinfo_path} --no-step-log",
            **(env_kwargs or {}),
        )
        try:
            metrics = run_episode(env, policy_fn, test_seed)
        finally:
            env.close()
        rows.append(
            {
                "method": method_name,
                "scenario": scenario,
                "demand": demand,
                "train_seed": train_seed,
                "test_seed": test_seed,
                "avg_waiting_time": metrics["avg_waiting_time"],
                "avg_queue": metrics["avg_queue"],
                "avg_trip_duration": metrics["avg_trip_duration"],
                "avg_time_loss": metrics["avg_time_loss"],
                "throughput": metrics["throughput"],
                "avg_speed": metrics["avg_speed"],
                "teleports": metrics["teleports"],
            }
        )

    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    write_header = not csv_path.exists()
    frame.to_csv(csv_path, mode="a", header=write_header, index=False)
    return frame


def _summary_path(scenario: str, demand: str, method_name: str) -> Path:
    return _REPO_ROOT / "results" / "raw" / scenario / demand / f"{method_name}.csv"


def _sumo_environment(env):
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_sumo_step") and hasattr(current, "additional_sumo_cmd"):
            return current
        current = getattr(current, "aec_env", None) or getattr(current, "env", None)
    raise RuntimeError("Could not find the SUMO environment under the PettingZoo wrappers.")


def _attach_vehicle_counters(sumo_env) -> None:
    """Count arrivals and teleports when this sumo-rl build omits them from info.

    sumo-rl 1.4.5 reports queue, waiting time, and speed, but not
    ``system_total_arrived`` or ``system_total_teleported``. Later releases
    accumulate those inside ``_sumo_step``; leave that path alone.
    """
    if hasattr(sumo_env, "num_arrived_vehicles"):
        return
    sumo_env.num_arrived_vehicles = 0
    sumo_env.num_teleported_vehicles = 0
    original_step = sumo_env._sumo_step

    def _sumo_step():
        original_step()
        sumo_env.num_arrived_vehicles += sumo_env.sumo.simulation.getArrivedNumber()
        sumo_env.num_teleported_vehicles += sumo_env.sumo.simulation.getEndingTeleportNumber()

    sumo_env._sumo_step = _sumo_step


def _tripinfo_path(sumo_env) -> Path | None:
    parts = str(sumo_env.additional_sumo_cmd or "").split()
    if "--tripinfo-output" not in parts:
        return None
    index = parts.index("--tripinfo-output")
    if index + 1 >= len(parts):
        raise ValueError("additional_sumo_cmd is missing the tripinfo output path.")
    return Path(parts[index + 1])


def _flatten_info(info: dict) -> dict[str, Any]:
    if not info:
        return {}
    if all(isinstance(value, dict) for value in info.values()):
        flat: dict[str, Any] = {}
        for agent_info in info.values():
            flat.update(agent_info)
        return flat
    return dict(info)


def _step_record(info: dict, sumo_env) -> dict[str, Any]:
    flat = _flatten_info(info)
    flat.setdefault("system_total_arrived", float(sumo_env.num_arrived_vehicles))
    flat.setdefault("system_total_teleported", float(sumo_env.num_teleported_vehicles))
    stopped = {
        key: value
        for key, value in flat.items()
        if key.endswith("_stopped") and not key.startswith("system_")
    }
    record = {column: flat.get(column, np.nan) for column in _STEP_COLUMNS}
    record.update(stopped)
    return record


def _episode_done(terminated, truncated) -> bool:
    if isinstance(truncated, dict) and truncated and all(truncated.values()):
        return True
    if isinstance(terminated, dict) and terminated and all(terminated.values()):
        return True
    return False


def _steps_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    stopped_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key.endswith("_stopped") and not key.startswith("system_")
        }
    )
    columns = _STEP_COLUMNS + stopped_columns
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _column_mean(steps: pd.DataFrame, column: str) -> float:
    if steps.empty or column not in steps.columns:
        return float("nan")
    return float(np.mean(steps[column].to_numpy(dtype=float)))


def _final_value(final, column: str) -> float:
    if isinstance(final, pd.Series) and column in final.index and pd.notna(final[column]):
        return float(final[column])
    return float("nan")


def _trip_means(path: Path | None) -> tuple[float, float]:
    if path is None or not path.is_file():
        return float("nan"), float("nan")
    durations: list[float] = []
    time_losses: list[float] = []
    for trip in ET.parse(path).getroot().iter("tripinfo"):
        if "duration" in trip.attrib:
            durations.append(float(trip.attrib["duration"]))
        if "timeLoss" in trip.attrib:
            time_losses.append(float(trip.attrib["timeLoss"]))
    duration = float(np.mean(durations)) if durations else float("nan")
    time_loss = float(np.mean(time_losses)) if time_losses else float("nan")
    return duration, time_loss
