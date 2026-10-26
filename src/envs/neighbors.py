"""Traffic-light neighbor graphs read from a SUMO network."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import sumolib


def build_neighbor_map(net_file: str, ts_ids: list[str]) -> dict[str, list[str]]:
    """Return each signal's upstream and downstream traffic-light neighbors.

    An incoming edge's from-node is upstream, and an outgoing edge's to-node is
    downstream. The node counts only when it is one of ``ts_ids``. The result
    is symmetric: a one-way link is stored in both directions. Neighbor ids are
    sorted. When a TLS id is not itself a node, as in some RESCO city nets, the
    signal is placed on the junctions controlled by that TLS.
    """
    net = sumolib.net.readNet(net_file)
    controlled = {ts_id: _controlled_nodes(net, ts_id) for ts_id in ts_ids}
    node_to_signal = {
        node.getID(): ts_id
        for ts_id, nodes in controlled.items()
        for node in nodes
    }

    neighbors = {ts_id: set() for ts_id in ts_ids}
    for ts_id, nodes in controlled.items():
        for node in nodes:
            for edge in node.getIncoming():
                _link(neighbors, node_to_signal, ts_id, edge.getFromNode())
            for edge in node.getOutgoing():
                _link(neighbors, node_to_signal, ts_id, edge.getToNode())

    for ts_id, linked in list(neighbors.items()):
        for other in linked:
            neighbors[other].add(ts_id)

    return {ts_id: sorted(neighbors[ts_id]) for ts_id in ts_ids}


def neighbor_distance_matrix(neighbor_map: dict[str, list[str]]) -> pd.DataFrame:
    """Hop distance between every pair of signals, from breadth-first search.

    The diagonal is 0. A pair in different components is ``inf``. Rows and
    columns follow the insertion order of ``neighbor_map``.
    """
    ids = list(neighbor_map)
    distance = {start: {goal: np.inf for goal in ids} for start in ids}
    for start in ids:
        distance[start][start] = 0
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in neighbor_map[current]:
                if distance[start][nxt] == np.inf:
                    distance[start][nxt] = distance[start][current] + 1
                    queue.append(nxt)
    return pd.DataFrame(distance, index=ids, columns=ids).T.astype(float)


def _controlled_nodes(net, ts_id: str) -> list:
    if net.hasNode(ts_id):
        return [net.getNode(ts_id)]
    junctions = {}
    for in_lane, _out_lane, _link in net.getTLS(ts_id).getConnections():
        junction = in_lane.getEdge().getToNode()
        junctions[junction.getID()] = junction
    if not junctions:
        raise ValueError(f"Traffic light {ts_id} controls no junctions in the network.")
    return list(junctions.values())


def _link(neighbors: dict[str, set[str]], node_to_signal: dict[str, str], ts_id: str, node) -> None:
    other = node_to_signal.get(node.getID())
    if other is not None and other != ts_id:
        neighbors[ts_id].add(other)
