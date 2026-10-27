from types import SimpleNamespace

import pytest

from src.baselines.max_pressure import phase_pressures, select_phase, served_lane_pairs


class FakeLane:
    def __init__(self, halting, vehicles):
        self.halting = halting
        self.vehicles = vehicles

    def getLastStepHaltingNumber(self, lane):
        return self.halting.get(lane, 0)

    def getLastStepVehicleNumber(self, lane):
        return self.vehicles.get(lane, 0)


class FakeTrafficLight:
    def __init__(self, links):
        self.links = links

    def getControlledLinks(self, ts_id):
        return self.links


def make_ts(halting, vehicles, green_phase=0, elapsed=100):
    links = [
        [("n_in", "s_out", "v0")],
        [("n_in", "s_out", "v1")],
        [("w_in", "e_out", "v2")],
        [("e_in", "w_out", "v3")],
    ]
    sumo = SimpleNamespace(trafficlight=FakeTrafficLight(links), lane=FakeLane(halting, vehicles))
    phases = [SimpleNamespace(state="GGrr"), SimpleNamespace(state="rrGG")]
    return SimpleNamespace(
        id="t",
        sumo=sumo,
        green_phases=phases,
        green_phase=green_phase,
        time_since_last_phase_change=elapsed,
        min_green=5,
        yellow_time=2,
    )


def test_duplicate_lane_pair_counted_once():
    ts = make_ts({"n_in": 6}, {"s_out": 1})
    pairs = served_lane_pairs(ts)
    assert pairs[0] == {("n_in", "s_out")}
    assert phase_pressures(ts) == [5.0, 0.0]


def test_pressure_math():
    ts = make_ts({"n_in": 2, "w_in": 7, "e_in": 3}, {"s_out": 1, "e_out": 2, "w_out": 0})
    assert phase_pressures(ts) == pytest.approx([1.0, 8.0])
    assert select_phase(ts) == 1


def test_tie_keeps_current_phase():
    ts = make_ts({"n_in": 4, "w_in": 4}, {}, green_phase=1)
    assert select_phase(ts) == 1


def test_min_green_blocks_switch():
    ts = make_ts({"w_in": 9}, {}, green_phase=0, elapsed=3)
    assert select_phase(ts) == 0
