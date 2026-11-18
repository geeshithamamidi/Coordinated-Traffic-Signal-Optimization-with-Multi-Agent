"""Ablation table from evaluation CSVs. No SUMO."""

import pandas as pd

from scripts.ablation_table import load_frames, main, waiting_pvalues
from src.evaluation.stats import compare, holm_correction


def _write_csv(folder, method: str, wait: float) -> None:
    rows = []
    for seed in range(5):
        for test_seed in (10, 11):
            rows.append(
                {
                    "method": method,
                    "scenario": "grid4x4",
                    "demand": "default",
                    "train_seed": seed,
                    "test_seed": test_seed,
                    "avg_waiting_time": wait + seed,
                    "avg_queue": wait / 2 + seed,
                    "avg_trip_duration": 100 + wait + seed,
                    "avg_time_loss": 1.0,
                    "throughput": 1400 - wait - seed,
                    "avg_speed": 5.0,
                    "teleports": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(folder / f"{method}.csv", index=False)


def _frozen(folder) -> str:
    path = folder / "frozen.yaml"
    path.write_text(
        "alpha: 0.5\nreward_alpha: 0.25\nvalidation_seeds: [5, 6, 7, 8, 9]\n",
        encoding="utf-8",
    )
    return str(path)


def test_missing_csvs_are_printed_and_do_not_fail(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    table = tmp_path / "ablation_grid4x4.tex"
    figure = tmp_path / "ablation_grid4x4.png"
    main(["--raw", str(raw), "--table", str(table), "--figure", str(figure)])
    out = capsys.readouterr().out
    for name in ("shared_ppo", "obs_only", "reward_only", "coordinated_ppo"):
        assert f"missing {raw / f'{name}.csv'}" in out
    assert "no ablation CSVs found" in out
    assert not table.exists()
    assert not figure.exists()


def test_table_reports_ablation_settings_and_holm_pvalues(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_csv(raw, "shared_ppo", 10.0)
    _write_csv(raw, "obs_only", 9.0)
    _write_csv(raw, "reward_only", 8.0)
    _write_csv(raw, "coordinated_ppo", 6.0)
    table = tmp_path / "ablation_grid4x4.tex"
    figure = tmp_path / "ablation_grid4x4.png"
    main(
        [
            "--raw",
            str(raw),
            "--table",
            str(table),
            "--figure",
            str(figure),
            "--frozen",
            _frozen(tmp_path),
        ]
    )
    text = table.read_text()
    assert "Neighbor obs" in text and "p vs shared PPO" in text
    assert "Shared PPO & off & 0 &" in text
    assert "Obs only & on & 0 &" in text
    assert "Reward only & off & 0.25 &" in text
    assert "Coordinated PPO & on & 0.5 &" in text
    assert text.split("Shared PPO")[1].split("\\\\")[0].rstrip().endswith("---")
    assert "$\\pm$" in text
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    df = load_frames(raw)
    order = ["obs_only", "reward_only", "coordinated_ppo"]
    raw_p = [compare(df, method, "shared_ppo", "avg_waiting_time", "grid4x4")["p_value"] for method in order]
    adjusted = holm_correction(raw_p)
    assert waiting_pvalues(df) == dict(zip(order, adjusted))
    for p in adjusted:
        shown = "$<0.001$" if p < 0.001 else f"{p:.3f}"
        assert shown in text
    capsys.readouterr()
