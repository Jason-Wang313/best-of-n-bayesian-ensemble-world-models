from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bayesian_ensemble_bon.metrics import evaluate_selection, stderr
from bayesian_ensemble_bon.selectors import METHODS, fit_calibration_beta, select_candidate
from bayesian_ensemble_bon.world import (
    WorldConfig,
    make_posterior,
    rollout_ensemble,
    rollout_true,
    sample_action_sequences,
)


@dataclass(frozen=True)
class ExperimentConfig:
    preset: str = "smoke"
    seeds: tuple[int, ...] = (0, 1, 2)
    n_values: tuple[int, ...] = (1, 4, 16, 64)
    problems_per_seed: int = 35
    calibration_candidates: int = 512
    horizon: int = 18
    ensemble_size: int = 28
    posterior_spread: float = 1.0
    blindspot: float = 1.0
    action_scale: float = 1.0

    @classmethod
    def for_preset(cls, preset: str) -> "ExperimentConfig":
        if preset == "smoke":
            return cls()
        if preset == "full":
            return cls(
                preset="full",
                seeds=(0, 1, 2, 3, 4),
                n_values=(1, 2, 4, 8, 16, 32, 64, 128),
                problems_per_seed=45,
                calibration_candidates=1536,
                ensemble_size=32,
            )
        raise ValueError(f"unknown preset: {preset}")

    @property
    def max_n(self) -> int:
        return max(self.n_values)

    def world_config(self) -> WorldConfig:
        return WorldConfig(
            horizon=self.horizon,
            ensemble_size=self.ensemble_size,
            posterior_spread=self.posterior_spread,
            blindspot=self.blindspot,
            action_scale=self.action_scale,
        )


def _write_rows(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, str], list[dict[str, float | int | str | bool]]] = {}
    for row in rows:
        groups.setdefault((int(row["n_candidates"]), str(row["method"])), []).append(row)

    metrics = [
        key
        for key in rows[0].keys()
        if key
        not in {
            "preset",
            "seed",
            "problem",
            "n_candidates",
            "method",
            "selected_index",
            "uses_truth",
        }
    ]
    summary: list[dict[str, float | int | str]] = []
    for (n_candidates, method), group in sorted(groups.items()):
        out: dict[str, float | int | str] = {
            "n_candidates": n_candidates,
            "method": method,
            "replicates": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_stderr"] = stderr(values)
        summary.append(out)
    return summary


def _write_generated_results(path: Path, summary: list[dict[str, float | int | str]], config: ExperimentConfig) -> None:
    by_key = {(int(row["n_candidates"]), str(row["method"])): row for row in summary}
    high_n = max(config.n_values)
    low_n = min(n for n in config.n_values if n > 1) if any(n > 1 for n in config.n_values) else min(config.n_values)

    def value(n: int, method: str, metric: str) -> float:
        return float(by_key[(n, method)][f"{metric}_mean"])

    macros = {
        "PaperSeeds": len(config.seeds),
        "PaperProblems": len(config.seeds) * config.problems_per_seed,
        "PaperHighN": high_n,
        "PaperLowN": low_n,
        "MeanHighTrue": value(high_n, "mean_bon", "selected_true_return"),
        "SampleHighTrue": value(high_n, "posterior_sample_bon", "selected_true_return"),
        "ThompsonHighTrue": value(high_n, "thompson_bon", "selected_true_return"),
        "UcbHighTrue": value(high_n, "ucb_bon", "selected_true_return"),
        "PessHighTrue": value(high_n, "calibrated_pessimistic_bon", "selected_true_return"),
        "OracleHighTrue": value(high_n, "oracle", "selected_true_return"),
        "SampleHighRegret": value(high_n, "posterior_sample_bon", "regret_to_candidate_oracle"),
        "PessHighRegret": value(high_n, "calibrated_pessimistic_bon", "regret_to_candidate_oracle"),
        "SampleHighOptimism": value(high_n, "posterior_sample_bon", "optimism_gap"),
        "PessHighOptimism": value(high_n, "calibrated_pessimistic_bon", "optimism_gap"),
        "SampleHighOOD": value(high_n, "posterior_sample_bon", "selected_ood_mass"),
        "PessHighOOD": value(high_n, "calibrated_pessimistic_bon", "selected_ood_mass"),
        "SampleHighTail": value(high_n, "posterior_sample_bon", "hazard_tail_selected"),
        "PessHighTail": value(high_n, "calibrated_pessimistic_bon", "hazard_tail_selected"),
        "SampleLowTrue": value(low_n, "posterior_sample_bon", "selected_true_return"),
    }
    lines = []
    for name, raw in macros.items():
        if isinstance(raw, int):
            rendered = str(raw)
        else:
            rendered = f"{raw:.3f}"
        lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: ExperimentConfig, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    world_config = config.world_config()
    rows: list[dict[str, float | int | str | bool]] = []
    calibration_rows: list[dict[str, float | int | str]] = []

    for seed in config.seeds:
        posterior = make_posterior(seed=10_000 + seed, config=world_config)
        calibration_rng = np.random.default_rng(20_000 + seed)
        calibration_actions = sample_action_sequences(
            calibration_rng,
            config.calibration_candidates,
            config.horizon,
            config.action_scale,
        )
        calibration_true = rollout_true(calibration_actions)
        calibration_predictions = rollout_ensemble(calibration_actions, posterior)
        beta = fit_calibration_beta(calibration_predictions, calibration_true.returns)
        calibration_rows.append(
            {
                "preset": config.preset,
                "seed": seed,
                "calibration_beta": beta,
                "calibration_mean_abs_error": float(
                    np.mean(np.abs(calibration_predictions.mean(axis=1) - calibration_true.returns))
                ),
                "calibration_mean_std": float(np.mean(calibration_predictions.std(axis=1))),
            }
        )

        for problem in range(config.problems_per_seed):
            problem_seed = 30_000 + 1009 * seed + problem
            rng = np.random.default_rng(problem_seed)
            actions = sample_action_sequences(rng, config.max_n, config.horizon, config.action_scale)
            true_rollout = rollout_true(actions)
            predictions = rollout_ensemble(actions, posterior)
            sampled_member_ids = rng.integers(0, config.ensemble_size, size=config.max_n)

            for n_candidates in config.n_values:
                true_prefix = type(true_rollout)(
                    returns=true_rollout.returns[:n_candidates],
                    max_position=true_rollout.max_position[:n_candidates],
                    ood_mass=true_rollout.ood_mass[:n_candidates],
                    hazard_mass=true_rollout.hazard_mass[:n_candidates],
                    action_energy=true_rollout.action_energy[:n_candidates],
                )
                prediction_prefix = predictions[:n_candidates]
                for method in METHODS:
                    method_rng = np.random.default_rng(problem_seed + 17 * n_candidates + 101 * METHODS.index(method))
                    result = select_candidate(
                        prediction_prefix,
                        true_prefix.returns,
                        method=method,
                        rng=method_rng,
                        beta=beta,
                        sampled_member_ids=sampled_member_ids[:n_candidates],
                    )
                    metrics = evaluate_selection(true_prefix, prediction_prefix, result)
                    rows.append(
                        {
                            "preset": config.preset,
                            "seed": seed,
                            "problem": problem,
                            "n_candidates": n_candidates,
                            "method": method,
                            "selected_index": result.index,
                            "uses_truth": result.uses_truth,
                            "calibration_beta": beta,
                            **metrics.as_dict(),
                        }
                    )

    summary = _aggregate(rows)
    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    calibration_path = output_dir / "calibration.csv"
    generated_results_path = output_dir / "generated_results.tex"
    manifest_path = output_dir / "manifest.json"
    _write_rows(results_path, rows)
    _write_rows(summary_path, summary)
    _write_rows(calibration_path, calibration_rows)
    _write_generated_results(generated_results_path, summary, config)
    manifest_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "methods": METHODS,
                "rows": len(rows),
                "summary_rows": len(summary),
                "files": {
                    "results": str(results_path),
                    "summary": str(summary_path),
                    "calibration": str(calibration_path),
                    "generated_results": str(generated_results_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "results": results_path,
        "summary": summary_path,
        "calibration": calibration_path,
        "generated_results": generated_results_path,
        "manifest": manifest_path,
    }
