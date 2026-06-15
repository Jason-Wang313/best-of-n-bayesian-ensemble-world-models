from __future__ import annotations

import csv

from bayesian_ensemble_bon.gym_benchmark import GymBenchmarkConfig, run_gym_benchmark


def test_v4_gym_benchmark_writes_protocol_and_summaries(tmp_path) -> None:
    config = GymBenchmarkConfig(
        seeds=(0,),
        env_names=("Pendulum-v1",),
        n_values=(6, 8),
        train_transitions=48,
        calibration_problems=1,
        calibration_candidates=8,
        test_problems=1,
        ensemble_size=3,
        cem_iterations=1,
        cem_population=10,
        cem_elites=3,
    )
    outputs = run_gym_benchmark(config, tmp_path / "results", tmp_path / "figures")
    assert outputs["gym_summary"].exists()
    assert outputs["gym_paired_effects"].exists()
    assert outputs["protocol_freeze"].exists()
    assert outputs["generated_results"].exists()
    with outputs["gym_summary"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    methods = {row["method"] for row in rows}
    assert "posterior_sample_bon" in methods
    assert "calibrated_pessimistic_bon" in methods
    assert "scripted_teacher" in methods
