from __future__ import annotations

import csv

from bayesian_ensemble_bon.v3_evidence import V3Config, run_v3_evidence


def test_v3_evidence_writes_stress_and_learned_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "results"
    figure_dir = tmp_path / "figures"
    config = V3Config(
        seeds=(0,),
        n_values=(4, 8),
        problems_per_condition=2,
        calibration_candidates=64,
        train_candidates=90,
        horizon=8,
        ensemble_size=6,
    )
    outputs = run_v3_evidence(config, output_dir, figure_dir)
    assert outputs["stress_summary"].exists()
    assert outputs["learned_summary"].exists()
    assert outputs["paired_effects"].exists()
    assert outputs["generated_results"].exists()
    with outputs["paired_effects"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["comparison"] for row in rows}
