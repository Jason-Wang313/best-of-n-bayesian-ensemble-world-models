from __future__ import annotations

import csv

from bayesian_ensemble_bon.experiment import ExperimentConfig, run_experiment


def test_smoke_experiment_writes_summary(tmp_path) -> None:
    config = ExperimentConfig(
        preset="test",
        seeds=(0,),
        n_values=(1, 4),
        problems_per_seed=3,
        calibration_candidates=64,
        horizon=8,
        ensemble_size=6,
    )
    outputs = run_experiment(config, tmp_path)
    assert outputs["summary"].exists()
    assert outputs["generated_results"].exists()
    with outputs["summary"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    methods = {row["method"] for row in rows}
    assert "posterior_sample_bon" in methods
    assert "calibrated_pessimistic_bon" in methods
