"""Neighbor degrees on grid4x4, and TLS ids that name a different junction."""

from pathlib import Path

import numpy as np
import sumo_rl

from src.envs.neighbors import build_neighbor_map, neighbor_distance_matrix

_RESCO = Path(sumo_rl.__file__).resolve().parent / "nets" / "RESCO"
_GRID_IDS = [f"{column}{row}" for column in "ABCD" for row in range(4)]


def _net(*parts: str) -> str:
    return str(_RESCO.joinpath(*parts))


def test_grid4x4_neighbor_degrees_and_hops():
    neighbor_map = build_neighbor_map(_net("grid4x4", "grid4x4.net.xml"), _GRID_IDS)
    assert list(neighbor_map) == _GRID_IDS
    for signal, neighbors in neighbor_map.items():
        column, row = signal[0], int(signal[1])
        boundaries = (column in "AD") + (row in (0, 3))
        assert len(neighbors) == 4 - boundaries
        for other in neighbors:
            assert signal in neighbor_map[other]

    assert neighbor_map["A0"] == ["A1", "B0"]
    assert neighbor_map["B1"] == ["A1", "B0", "B2", "C1"]

    distance = neighbor_distance_matrix(neighbor_map)
    assert list(distance.index) == _GRID_IDS
    assert distance.loc["A0", "A0"] == 0
    assert distance.loc["A0", "B0"] == 1
    assert distance.loc["A0", "B1"] == 2
    assert distance.loc["A0", "D3"] == 6
    assert np.array_equal(distance.to_numpy(), distance.to_numpy().T)


def test_ingolstadt_tls_uses_controlled_junction():
    net_file = _net("ingolstadt7", "ingolstadt7.net.xml")
    import sumolib

    net = sumolib.net.readNet(net_file)
    ts_ids = [tls.getID() for tls in net.getTrafficLights()]
    assert "gneJ207" in ts_ids
    assert not net.hasNode("gneJ207")

    neighbor_map = build_neighbor_map(net_file, ts_ids)
    assert "gneJ143" in neighbor_map["gneJ207"]
    assert "gneJ207" in neighbor_map["gneJ143"]
