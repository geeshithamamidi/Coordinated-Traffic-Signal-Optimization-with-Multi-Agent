# Coordinated Traffic Signal Optimization with Multi-Agent

## Team

- geeshithamamidi
- rudolfbertin
- laharipydikondala-byte

## Summary

This course project studies coordinated traffic-signal control with multi-agent reinforcement learning. Intersections are simulated in Eclipse SUMO and exposed as a PettingZoo environment through sumo-rl. We train policies with Stable-Baselines3, compare them with classical signal baselines, and collect delay, queue, and throughput results for the report.

## Install

```bash
conda create -n traffic python=3.11
conda activate traffic
pip install -r requirements.txt
conda env config vars set SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"
```

Reactivate the environment so `SUMO_HOME` is exported:

```bash
conda deactivate
conda activate traffic
```

## How to run

```bash
conda activate traffic
pip install -r requirements.txt
python scripts/check_install.py
pytest -q
```

`python scripts/check_install.py` should print SUMO 1.27.1, the sumo-rl version, and a final info dict that includes `system_mean_waiting_time`, then `INSTALL OK`. TraCI may also print `Step #` and `Retrying in 1 seconds` while SUMO starts. Those lines are normal.

## Troubleshooting

SUMO 1.28 needs macOS 15, so we pin 1.27.1. `SUMO_HOME` is set via `conda env config vars set SUMO_HOME=...`. Reactivate `traffic` after setting it so the variable is exported.

