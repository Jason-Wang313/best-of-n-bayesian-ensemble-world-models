from __future__ import annotations

import csv

from bayesian_ensemble_bon.mujoco_benchmark import MuJoCoBenchmarkConfig, run_mujoco_benchmark


def test_v4_mujoco_benchmark_writes_protocol_and_summaries(tmp_path) -> None:
    config = MuJoCoBenchmarkConfig(
        seeds=(0,),
        env_names=("InvertedPendulum-v5",),
        n_values=(6, 8),
        train_transitions=40,
        calibration_problems=1,
        calibration_candidates=8,
        test_problems=1,
        ensemble_size=3,
        cem_iterations=1,
        cem_population=8,
        cem_elites=3,
    )
    outputs = run_mujoco_benchmark(config, tmp_path / "results", tmp_path / "figures")
    assert outputs["mujoco_summary"].exists()
    assert outputs["mujoco_paired_effects"].exists()
    assert outputs["protocol_freeze"].exists()
    assert outputs["generated_results"].exists()
    with outputs["mujoco_summary"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    methods = {row["method"] for row in rows}
    assert "posterior_sample_bon" in methods
    assert "calibrated_pessimistic_bon" in methods
    assert "cem_mean_mpc" in methods
