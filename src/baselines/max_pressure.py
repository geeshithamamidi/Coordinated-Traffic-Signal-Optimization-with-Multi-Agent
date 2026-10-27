from __future__ import annotations

from src.evaluation.metrics import _sumo_environment


def served_lane_pairs(ts) -> list[set[tuple[str, str]]]:
    links = ts.sumo.trafficlight.getControlledLinks(ts.id)
    per_phase = []
    for phase in ts.green_phases:
        pairs: set[tuple[str, str]] = set()
        for index, char in enumerate(phase.state):
            if char not in ("G", "g") or index >= len(links):
                continue
            for link in links[index]:
                if link:
                    pairs.add((link[0], link[1]))
        per_phase.append(pairs)
    return per_phase


def phase_pressures(ts, pairs_per_phase: list[set[tuple[str, str]]] | None = None) -> list[float]:
    if pairs_per_phase is None:
        pairs_per_phase = served_lane_pairs(ts)
    lane = ts.sumo.lane
    pressures = []
    for pairs in pairs_per_phase:
        total = 0.0
        for in_lane, out_lane in pairs:
            total += lane.getLastStepHaltingNumber(in_lane) - lane.getLastStepVehicleNumber(out_lane)
        pressures.append(float(total))
    return pressures


def select_phase(ts, pairs_per_phase: list[set[tuple[str, str]]] | None = None) -> int:
    current = int(ts.green_phase)
    if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time:
        return current
    pressures = phase_pressures(ts, pairs_per_phase)
    best = max(pressures)
    if pressures[current] >= best:
        return current
    return int(pressures.index(best))


class MaxPressureController:
    def __init__(self):
        self._pairs: dict[str, list[set[tuple[str, str]]]] = {}

    def __call__(self, obs_dict: dict, env) -> dict:
        sumo_env = _sumo_environment(env)
        actions = {}
        for agent in env.agents:
            ts = sumo_env.traffic_signals[agent]
            if agent not in self._pairs:
                self._pairs[agent] = served_lane_pairs(ts)
            actions[agent] = select_phase(ts, self._pairs[agent])
        return actions


def max_pressure_factory(train_seed: int) -> MaxPressureController:
    return MaxPressureController()
