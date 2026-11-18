"""Shared method colors for result figures.

Okabe–Ito colors, so the same method keeps the same color in every plot.
"""

from __future__ import annotations

METHOD_COLORS = {
    "fixed_time": "#000000",
    "fixed_cycle": "#56B4E9",
    "max_pressure": "#E69F00",
    "ql": "#0072B2",
    "dqn": "#009E73",
    "ippo": "#CC79A7",
    "shared_ppo": "#D55E00",
    "obs_only": "#F0E442",
    "reward_only": "#332288",
    "coordinated_ppo": "#AA4499",
}

# Used only for a method that is not in METHOD_COLORS.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def color_for(method: str, index: int = 0) -> str:
    """Color for ``method``, or the palette entry at ``index`` when it is new."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return PALETTE[index % len(PALETTE)]


# Readable names for tables and figures. Keys are the method ids used in the CSVs.
METHOD_LABELS = {
    "fixed_time": "Fixed-time",
    "fixed_cycle": "Tuned fixed-cycle",
    "max_pressure": "Max-Pressure",
    "ql": "Independent Q-learning",
    "dqn": "Independent DQN",
    "ippo": "Independent PPO",
    "shared_ppo": "Shared PPO",
    "coordinated_ppo": "Coordinated PPO",
    "obs_only": "Obs-only",
    "reward_only": "Reward-only",
}


def label_for(method: str) -> str:
    """Readable label for ``method``; unknown ids are returned unchanged."""
    return METHOD_LABELS.get(method, method)
