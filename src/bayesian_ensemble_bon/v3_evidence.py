from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bayesian_ensemble_bon.experiment import ExperimentConfig, _write_rows
from bayesian_ensemble_bon.learned import fit_bootstrap_return_ensemble
from bayesian_ensemble_bon.metrics import evaluate_selection, stderr
from bayesian_ensemble_bon.selectors import METHODS, fit_calibration_beta, select_candidate
from bayesian_ensemble_bon.world import (
    TrueRollout,
    WorldConfig,
    make_posterior,
    rollout_ensemble,
    rollout_true,
    sample_action_sequences,
)


@dataclass(frozen=True)
class V3Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    n_values: tuple[int, ...] = (16, 64, 128)
    problems_per_condition: int = 24
    calibration_candidates: int = 768
    train_candidates: int = 1400
    horizon: int = 18
    ensemble_size: int = 24

    @property
    def max_n(self) -> int:
        return max(self.n_values)


STRESS_CONDITIONS = [
    {"condition": "nominal", "posterior_spread": 1.0, "blindspot": 1.0, "action_scale": 1.0},
    {"condition": "low_blindspot", "posterior_spread": 1.0, "blindspot": 0.55, "action_scale": 1.0},
    {"condition": "high_blindspot", "posterior_spread": 1.0, "blindspot": 1.55, "action_scale": 1.0},
    {"condition": "low_spread", "posterior_spread": 0.65, "blindspot": 1.0, "action_scale": 1.0},
    {"condition": "high_spread", "posterior_spread": 1.45, "blindspot": 1.0, "action_scale": 1.0},
    {"condition": "gentle_actions", "posterior_spread": 1.0, "blindspot": 1.0, "action_scale": 0.75},
    {"condition": "wide_actions", "posterior_spread": 1.0, "blindspot": 1.0, "action_scale": 1.25},
    {"condition": "hard_shift", "posterior_spread": 1.35, "blindspot": 1.45, "action_scale": 1.25},
    {"condition": "calibrated_shift", "posterior_spread": 0.85, "blindspot": 1.35, "action_scale": 1.15},
]


def _slice_rollout(rollout: TrueRollout, n: int) -> TrueRollout:
    return TrueRollout(
        returns=rollout.returns[:n],
        max_position=rollout.max_position[:n],
        ood_mass=rollout.ood_mass[:n],
        hazard_mass=rollout.hazard_mass[:n],
        action_energy=rollout.action_energy[:n],
    )


def _evaluate_rows(
    *,
    suite: str,
    condition: str,
    seed: int,
    problem: int,
    n_values: tuple[int, ...],
    true_rollout: TrueRollout,
    predictions: np.ndarray,
    beta: float,
    rng_seed: int,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    sampled_member_rng = np.random.default_rng(rng_seed + 83)
    sampled_member_ids = sampled_member_rng.integers(0, predictions.shape[1], size=max(n_values))
    for n_candidates in n_values:
        true_prefix = _slice_rollout(true_rollout, n_candidates)
        prediction_prefix = predictions[:n_candidates]
        for method in METHODS:
            method_rng = np.random.default_rng(rng_seed + 17 * n_candidates + 101 * METHODS.index(method))
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
                    "suite": suite,
                    "condition": condition,
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
    return rows


def _aggregate(rows: list[dict[str, float | int | str | bool]], keys: tuple[str, ...]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[object, ...], list[dict[str, float | int | str | bool]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    skip = set(keys) | {"seed", "problem", "selected_index", "uses_truth"}
    metrics = [key for key in rows[0].keys() if key not in skip and isinstance(rows[0][key], (float, int))]
    summary: list[dict[str, float | int | str]] = []
    for group_key, group in sorted(groups.items()):
        out: dict[str, float | int | str] = {key: group_key[idx] for idx, key in enumerate(keys)}
        out["replicates"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_stderr"] = stderr(values)
        summary.append(out)
    return summary


def _paired_effects(rows: list[dict[str, float | int | str | bool]], suite: str, condition: str, high_n: int) -> list[dict[str, float | int | str]]:
    by_problem: dict[tuple[int, int], dict[str, dict[str, float | int | str | bool]]] = {}
    for row in rows:
        if row["suite"] == suite and row["condition"] == condition and int(row["n_candidates"]) == high_n:
            by_problem.setdefault((int(row["seed"]), int(row["problem"])), {})[str(row["method"])] = row
    comparisons = [
        ("sample_minus_mean", "posterior_sample_bon", "mean_bon"),
        ("sample_minus_calibrated_lcb", "posterior_sample_bon", "calibrated_pessimistic_bon"),
        ("ucb_minus_mean", "ucb_bon", "mean_bon"),
        ("calibrated_lcb_minus_mean", "calibrated_pessimistic_bon", "mean_bon"),
    ]
    effects: list[dict[str, float | int | str]] = []
    for name, lhs, rhs in comparisons:
        diffs = []
        regret_diffs = []
        tail_diffs = []
        for methods in by_problem.values():
            if lhs in methods and rhs in methods:
                diffs.append(float(methods[lhs]["selected_true_return"]) - float(methods[rhs]["selected_true_return"]))
                regret_diffs.append(
                    float(methods[lhs]["regret_to_candidate_oracle"]) - float(methods[rhs]["regret_to_candidate_oracle"])
                )
                tail_diffs.append(float(methods[lhs]["hazard_tail_selected"]) - float(methods[rhs]["hazard_tail_selected"]))
        arr = np.asarray(diffs, dtype=np.float64)
        reg = np.asarray(regret_diffs, dtype=np.float64)
        tail = np.asarray(tail_diffs, dtype=np.float64)
        se = stderr(arr.tolist())
        effects.append(
            {
                "suite": suite,
                "condition": condition,
                "n_candidates": high_n,
                "comparison": name,
                "paired_replicates": int(arr.size),
                "true_return_gap_mean": float(arr.mean()),
                "true_return_gap_stderr": se,
                "true_return_gap_ci95_low": float(arr.mean() - 1.96 * se),
                "true_return_gap_ci95_high": float(arr.mean() + 1.96 * se),
                "regret_gap_mean": float(reg.mean()),
                "hazard_tail_gap_mean": float(tail.mean()),
                "lhs_worse_rate": float(np.mean(arr < 0.0)),
            }
        )
    return effects


def _coverage_rows(rows: list[dict[str, float | int | str | bool]], high_n: int) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for suite in sorted({str(row["suite"]) for row in rows}):
        for condition in sorted({str(row["condition"]) for row in rows if row["suite"] == suite}):
            group = [
                row
                for row in rows
                if row["suite"] == suite
                and row["condition"] == condition
                and int(row["n_candidates"]) == high_n
                and str(row["method"]) == "calibrated_pessimistic_bon"
            ]
            if not group:
                continue
            out.append(
                {
                    "suite": suite,
                    "condition": condition,
                    "n_candidates": high_n,
                    "replicates": len(group),
                    "optimistic_miss_rate": float(np.mean([float(row["optimism_gap"]) > 0.0 for row in group])),
                    "mean_beta": float(np.mean([float(row["calibration_beta"]) for row in group])),
                    "mean_selected_std": float(np.mean([float(row["selected_pred_std"]) for row in group])),
                    "mean_selected_ood": float(np.mean([float(row["selected_ood_mass"]) for row in group])),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    _write_rows(path, rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _value(summary: list[dict[str, str]], *, suite: str, condition: str, n: int, method: str, metric: str) -> float:
    for row in summary:
        if row["suite"] == suite and row["condition"] == condition and int(row["n_candidates"]) == n and row["method"] == method:
            return float(row[f"{metric}_mean"])
    raise KeyError((suite, condition, n, method, metric))


def _write_generated_results(path: Path, config: V3Config, stress_summary: Path, learned_summary: Path, paired: Path) -> None:
    stress = _read_csv(stress_summary)
    learned = _read_csv(learned_summary)
    effects = _read_csv(paired)
    high_n = config.max_n
    sample_vs_lcb = next(
        row
        for row in effects
        if row["suite"] == "posterior_stress"
        and row["condition"] == "nominal"
        and row["comparison"] == "sample_minus_calibrated_lcb"
    )
    learned_sample_vs_lcb = next(
        row
        for row in effects
        if row["suite"] == "learned_bootstrap"
        and row["condition"] == "eval_shift"
        and row["comparison"] == "sample_minus_calibrated_lcb"
    )
    conditions = {row["condition"] for row in stress if row["suite"] == "posterior_stress"}
    stress_lcb_beats_sample = []
    for condition in conditions:
        sample_regret = _value(
            stress, suite="posterior_stress", condition=condition, n=high_n, method="posterior_sample_bon", metric="regret_to_candidate_oracle"
        )
        lcb_regret = _value(
            stress,
            suite="posterior_stress",
            condition=condition,
            n=high_n,
            method="calibrated_pessimistic_bon",
            metric="regret_to_candidate_oracle",
        )
        stress_lcb_beats_sample.append(lcb_regret < sample_regret)

    macros: dict[str, float | int] = {
        "VThreeStressConditions": len(conditions),
        "VThreeStressReplicates": len(config.seeds) * config.problems_per_condition,
        "VThreeStressSampleRegret": _value(
            stress,
            suite="posterior_stress",
            condition="nominal",
            n=high_n,
            method="posterior_sample_bon",
            metric="regret_to_candidate_oracle",
        ),
        "VThreeStressLCBRegret": _value(
            stress,
            suite="posterior_stress",
            condition="nominal",
            n=high_n,
            method="calibrated_pessimistic_bon",
            metric="regret_to_candidate_oracle",
        ),
        "VThreeStressLCBWinConditions": int(sum(stress_lcb_beats_sample)),
        "VThreeLearnedProblems": len(config.seeds) * config.problems_per_condition,
        "VThreeLearnedSampleRegret": _value(
            learned,
            suite="learned_bootstrap",
            condition="eval_shift",
            n=high_n,
            method="posterior_sample_bon",
            metric="regret_to_candidate_oracle",
        ),
        "VThreeLearnedMeanRegret": _value(
            learned, suite="learned_bootstrap", condition="eval_shift", n=high_n, method="mean_bon", metric="regret_to_candidate_oracle"
        ),
        "VThreeLearnedLCBRegret": _value(
            learned,
            suite="learned_bootstrap",
            condition="eval_shift",
            n=high_n,
            method="calibrated_pessimistic_bon",
            metric="regret_to_candidate_oracle",
        ),
        "VThreeNominalSampleLCBGap": float(sample_vs_lcb["true_return_gap_mean"]),
        "VThreeNominalSampleLCBCILow": float(sample_vs_lcb["true_return_gap_ci95_low"]),
        "VThreeNominalSampleLCBCIHigh": float(sample_vs_lcb["true_return_gap_ci95_high"]),
        "VThreeLearnedSampleLCBGap": float(learned_sample_vs_lcb["true_return_gap_mean"]),
        "VThreeLearnedSampleLCBCILow": float(learned_sample_vs_lcb["true_return_gap_ci95_low"]),
        "VThreeLearnedSampleLCBCIHigh": float(learned_sample_vs_lcb["true_return_gap_ci95_high"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value if isinstance(value, int) else f'{value:.3f}'}}}"
            for name, value in macros.items()
        )
        + "\n",
        encoding="utf-8",
    )


def _plot_learned(learned_summary: Path, figure_dir: Path) -> Path:
    rows = _read_csv(learned_summary)
    methods = [
        ("oracle", "candidate oracle"),
        ("mean_bon", "posterior mean"),
        ("posterior_sample_bon", "sampled posterior"),
        ("ucb_bon", "mean + std"),
        ("calibrated_pessimistic_bon", "calibrated LCB"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    for method, label in methods:
        points = [
            (int(row["n_candidates"]), float(row["selected_true_return_mean"]))
            for row in rows
            if row["suite"] == "learned_bootstrap" and row["condition"] == "eval_shift" and row["method"] == method
        ]
        if points:
            points = sorted(points)
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("candidate budget N")
    ax.set_ylabel("selected true return")
    ax.set_title("Learned bootstrap ensemble under evaluation shift")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "v3_learned_bootstrap_return.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_stress(stress_summary: Path, figure_dir: Path, high_n: int) -> Path:
    rows = _read_csv(stress_summary)
    conditions = sorted({row["condition"] for row in rows if row["suite"] == "posterior_stress"})
    sample = []
    lcb = []
    for condition in conditions:
        sample.append(
            _value(
                rows,
                suite="posterior_stress",
                condition=condition,
                n=high_n,
                method="posterior_sample_bon",
                metric="regret_to_candidate_oracle",
            )
        )
        lcb.append(
            _value(
                rows,
                suite="posterior_stress",
                condition=condition,
                n=high_n,
                method="calibrated_pessimistic_bon",
                metric="regret_to_candidate_oracle",
            )
        )
    x = np.arange(len(conditions))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - width / 2, sample, width=width, label="sampled posterior")
    ax.bar(x + width / 2, lcb, width=width, label="calibrated LCB")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=7)
    ax.set_ylabel("regret at N=128")
    ax.set_title("Posterior stress grid: selected regret")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "v3_stress_regret_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_paired(paired_csv: Path, figure_dir: Path) -> Path:
    rows = [
        row
        for row in _read_csv(paired_csv)
        if row["comparison"] == "sample_minus_calibrated_lcb"
        and row["condition"] in {"nominal", "eval_shift"}
    ]
    labels = [f"{row['suite']}\n{row['condition']}" for row in rows]
    means = [float(row["true_return_gap_mean"]) for row in rows]
    lows = [float(row["true_return_gap_ci95_low"]) for row in rows]
    highs = [float(row["true_return_gap_ci95_high"]) for row in rows]
    lower = [mean - low for mean, low in zip(means, lows)]
    upper = [high - mean for mean, high in zip(means, highs)]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(x, means, color="#b91c1c")
    ax.errorbar(x, means, yerr=[lower, upper], fmt="none", color="#111827", capsize=4)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("sampled posterior minus LCB true return")
    ax.set_title("Paired high-N effect with 95% CI")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "v3_paired_effects.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_claim_audit(path: Path, config: V3Config, stress_summary: Path, learned_summary: Path, paired_csv: Path) -> None:
    stress = _read_csv(stress_summary)
    learned = _read_csv(learned_summary)
    paired = _read_csv(paired_csv)
    high_n = config.max_n
    stress_sample_regret = _value(
        stress,
        suite="posterior_stress",
        condition="nominal",
        n=high_n,
        method="posterior_sample_bon",
        metric="regret_to_candidate_oracle",
    )
    stress_lcb_regret = _value(
        stress,
        suite="posterior_stress",
        condition="nominal",
        n=high_n,
        method="calibrated_pessimistic_bon",
        metric="regret_to_candidate_oracle",
    )
    learned_sample_regret = _value(
        learned,
        suite="learned_bootstrap",
        condition="eval_shift",
        n=high_n,
        method="posterior_sample_bon",
        metric="regret_to_candidate_oracle",
    )
    learned_lcb_regret = _value(
        learned,
        suite="learned_bootstrap",
        condition="eval_shift",
        n=high_n,
        method="calibrated_pessimistic_bon",
        metric="regret_to_candidate_oracle",
    )
    nominal_effect = next(
        row for row in paired if row["suite"] == "posterior_stress" and row["condition"] == "nominal" and row["comparison"] == "sample_minus_calibrated_lcb"
    )
    lines = [
        "# V3 Claim-Evidence Audit",
        "",
        "This audit is generated from `results/v3` and scopes the v3 claims to CPU-light controlled evidence.",
        "",
        "| Claim | Evidence | Boundary |",
        "|---|---|---|",
        f"| Posterior-sampled high-N selection remains fragile under posterior stress. | Nominal stress regret at N={high_n}: sampled posterior {stress_sample_regret:.3f} vs calibrated LCB {stress_lcb_regret:.3f}. | Controlled latent system; not a real robot benchmark. |",
        f"| The effect is not only hand-coded posterior members. | Learned bootstrap return ensemble regret at N={high_n}: sampled posterior {learned_sample_regret:.3f} vs calibrated LCB {learned_lcb_regret:.3f}. | Lightweight learned value ensemble, not a large neural world model. |",
        f"| The high-N harm is paired, not just aggregate plotting. | Nominal paired true-return gap sampled minus LCB: {float(nominal_effect['true_return_gap_mean']):.3f} with 95% CI [{float(nominal_effect['true_return_gap_ci95_low']):.3f}, {float(nominal_effect['true_return_gap_ci95_high']):.3f}]. | Paired over synthetic candidate problems. |",
        f"| Calibration evidence is explicitly artifact-backed. | `calibration_coverage.csv` reports optimistic-miss rates and beta values for each suite/condition. | Coverage is diagnostic, not a formal conformal guarantee. |",
        "",
        "The v3 evidence strengthens scope and rigor without raising the claim to hardware, MuJoCo, or large-scale video world-model validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v3_evidence(config: V3Config, output_dir: Path, figure_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stress_rows: list[dict[str, float | int | str | bool]] = []
    learned_rows: list[dict[str, float | int | str | bool]] = []

    for condition_id, condition in enumerate(STRESS_CONDITIONS):
        world_config = WorldConfig(
            horizon=config.horizon,
            ensemble_size=config.ensemble_size,
            posterior_spread=float(condition["posterior_spread"]),
            blindspot=float(condition["blindspot"]),
            action_scale=float(condition["action_scale"]),
        )
        for seed in config.seeds:
            posterior = make_posterior(seed=60_000 + condition_id * 100 + seed, config=world_config)
            calibration_rng = np.random.default_rng(61_000 + condition_id * 100 + seed)
            calibration_actions = sample_action_sequences(
                calibration_rng,
                config.calibration_candidates,
                config.horizon,
                world_config.action_scale,
            )
            calibration_true = rollout_true(calibration_actions)
            calibration_predictions = rollout_ensemble(calibration_actions, posterior)
            beta = fit_calibration_beta(calibration_predictions, calibration_true.returns)
            for problem in range(config.problems_per_condition):
                rng_seed = 62_000 + condition_id * 10_000 + 997 * seed + problem
                rng = np.random.default_rng(rng_seed)
                actions = sample_action_sequences(rng, config.max_n, config.horizon, world_config.action_scale)
                true_rollout = rollout_true(actions)
                predictions = rollout_ensemble(actions, posterior)
                stress_rows.extend(
                    _evaluate_rows(
                        suite="posterior_stress",
                        condition=str(condition["condition"]),
                        seed=seed,
                        problem=problem,
                        n_values=config.n_values,
                        true_rollout=true_rollout,
                        predictions=predictions,
                        beta=beta,
                        rng_seed=rng_seed,
                    )
                )

    for seed in config.seeds:
        train_rng = np.random.default_rng(80_000 + seed)
        train_actions = sample_action_sequences(train_rng, config.train_candidates, config.horizon, action_scale=0.72)
        train_true = rollout_true(train_actions)
        model = fit_bootstrap_return_ensemble(
            train_actions,
            train_true.returns,
            seed=81_000 + seed,
            ensemble_size=config.ensemble_size,
        )
        calibration_rng = np.random.default_rng(82_000 + seed)
        calibration_actions = sample_action_sequences(calibration_rng, config.calibration_candidates, config.horizon, action_scale=0.95)
        calibration_true = rollout_true(calibration_actions)
        calibration_predictions = model.predict(calibration_actions)
        beta = fit_calibration_beta(calibration_predictions, calibration_true.returns)
        for problem in range(config.problems_per_condition):
            rng_seed = 83_000 + 997 * seed + problem
            rng = np.random.default_rng(rng_seed)
            actions = sample_action_sequences(rng, config.max_n, config.horizon, action_scale=1.15)
            true_rollout = rollout_true(actions)
            predictions = model.predict(actions)
            learned_rows.extend(
                _evaluate_rows(
                    suite="learned_bootstrap",
                    condition="eval_shift",
                    seed=seed,
                    problem=problem,
                    n_values=config.n_values,
                    true_rollout=true_rollout,
                    predictions=predictions,
                    beta=beta,
                    rng_seed=rng_seed,
                )
            )

    stress_summary = _aggregate(stress_rows, ("suite", "condition", "n_candidates", "method"))
    learned_summary = _aggregate(learned_rows, ("suite", "condition", "n_candidates", "method"))
    all_rows = stress_rows + learned_rows
    paired_rows: list[dict[str, float | int | str]] = []
    paired_rows.extend(_paired_effects(all_rows, "posterior_stress", "nominal", config.max_n))
    paired_rows.extend(_paired_effects(all_rows, "learned_bootstrap", "eval_shift", config.max_n))
    coverage_rows = _coverage_rows(all_rows, config.max_n)

    paths = {
        "stress_results": output_dir / "stress_results.csv",
        "stress_summary": output_dir / "stress_summary.csv",
        "learned_results": output_dir / "learned_results.csv",
        "learned_summary": output_dir / "learned_summary.csv",
        "paired_effects": output_dir / "paired_effects.csv",
        "calibration_coverage": output_dir / "calibration_coverage.csv",
        "generated_results": output_dir / "generated_results.tex",
        "claim_audit": output_dir / "claim_evidence_audit.md",
        "manifest": output_dir / "manifest.json",
    }
    _write_csv(paths["stress_results"], stress_rows)
    _write_csv(paths["stress_summary"], stress_summary)
    _write_csv(paths["learned_results"], learned_rows)
    _write_csv(paths["learned_summary"], learned_summary)
    _write_csv(paths["paired_effects"], paired_rows)
    _write_csv(paths["calibration_coverage"], coverage_rows)
    _write_generated_results(paths["generated_results"], config, paths["stress_summary"], paths["learned_summary"], paths["paired_effects"])
    _write_claim_audit(paths["claim_audit"], config, paths["stress_summary"], paths["learned_summary"], paths["paired_effects"])

    figures = {
        "learned_figure": _plot_learned(paths["learned_summary"], figure_dir),
        "stress_figure": _plot_stress(paths["stress_summary"], figure_dir, config.max_n),
        "paired_figure": _plot_paired(paths["paired_effects"], figure_dir),
    }
    paths.update(figures)
    paths["manifest"].write_text(
        json.dumps(
            {
                "config": asdict(config),
                "stress_conditions": STRESS_CONDITIONS,
                "stress_rows": len(stress_rows),
                "learned_rows": len(learned_rows),
                "paired_rows": len(paired_rows),
                "coverage_rows": len(coverage_rows),
                "files": {name: str(path) for name, path in paths.items() if name != "manifest"},
                "compute_note": "CPU-light numpy sweeps; no model downloads; bounded candidate and ensemble sizes.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
