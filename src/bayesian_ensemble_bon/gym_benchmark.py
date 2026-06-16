from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bayesian_ensemble_bon.metrics import stderr
from bayesian_ensemble_bon.selectors import METHODS, fit_calibration_beta, select_candidate


@dataclass(frozen=True)
class GymEnvSpec:
    name: str
    state_low: tuple[float, ...]
    state_high: tuple[float, ...]
    train_state_low: tuple[float, ...]
    train_state_high: tuple[float, ...]
    test_state_low: tuple[float, ...]
    test_state_high: tuple[float, ...]
    action_low: float
    action_high: float
    train_action_scale: float
    candidate_action_scale: float
    horizon: int

    @property
    def state_dim(self) -> int:
        return len(self.state_low)


@dataclass(frozen=True)
class GymBenchmarkConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    env_names: tuple[str, ...] = ("Pendulum-v1", "MountainCarContinuous-v0")
    n_values: tuple[int, ...] = (8, 32, 96)
    train_transitions: int = 900
    calibration_problems: int = 5
    calibration_candidates: int = 48
    test_problems: int = 12
    ensemble_size: int = 12
    ridge: float = 1e-3
    randomized_prior_scale: float = 0.18
    cem_iterations: int = 3
    cem_population: int = 80
    cem_elites: int = 10

    @property
    def max_n(self) -> int:
        return max(self.n_values)


ENV_SPECS = {
    "Pendulum-v1": GymEnvSpec(
        name="Pendulum-v1",
        state_low=(-np.pi, -8.0),
        state_high=(np.pi, 8.0),
        train_state_low=(-1.2, -2.5),
        train_state_high=(1.2, 2.5),
        test_state_low=(-np.pi, -4.0),
        test_state_high=(np.pi, 4.0),
        action_low=-2.0,
        action_high=2.0,
        train_action_scale=0.45,
        candidate_action_scale=1.0,
        horizon=25,
    ),
    "MountainCarContinuous-v0": GymEnvSpec(
        name="MountainCarContinuous-v0",
        state_low=(-1.2, -0.07),
        state_high=(0.6, 0.07),
        train_state_low=(-0.65, -0.025),
        train_state_high=(-0.35, 0.025),
        test_state_low=(-0.62, -0.015),
        test_state_high=(-0.38, 0.015),
        action_low=-1.0,
        action_high=1.0,
        train_action_scale=0.55,
        candidate_action_scale=1.0,
        horizon=45,
    ),
}

STANDALONE_BASELINES = ["scripted_teacher", "cem_mean_mpc"]
GATED_METHOD = "validation_gated_bon"
GATE_CANDIDATES = (
    "mean_bon",
    "posterior_sample_bon",
    "pessimistic_bon",
    "quantile_bon",
    "calibrated_pessimistic_bon",
)
GATE_TIE_PREFERENCE = (
    "calibrated_pessimistic_bon",
    "quantile_bon",
    "pessimistic_bon",
    "posterior_sample_bon",
    "mean_bon",
)
GATE_NEAR_TIE_MARGIN = 0.05
BENCHMARK_METHODS = METHODS + [GATED_METHOD] + STANDALONE_BASELINES


@dataclass(frozen=True)
class BootstrapDynamicsEnsemble:
    env_name: str
    weights: np.ndarray
    prior_weights: np.ndarray
    ridge: float
    randomized_prior_scale: float

    @property
    def ensemble_size(self) -> int:
        return int(self.weights.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.weights.shape[2] - 1)


def _feature_vector(env_name: str, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    parts: list[np.ndarray] = [
        np.ones(1),
        state,
        action,
        state**2,
        action**2,
        state * action[0],
    ]
    if env_name == "Pendulum-v1":
        parts.extend([np.sin(state[:1]), np.cos(state[:1])])
    return np.concatenate(parts).astype(np.float64)


def _normalize_state(spec: GymEnvSpec, state: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=np.float64).copy()
    if spec.name == "Pendulum-v1":
        out[0] = ((out[0] + np.pi) % (2.0 * np.pi)) - np.pi
        out[1] = np.clip(out[1], spec.state_low[1], spec.state_high[1])
    else:
        out = np.clip(out, np.asarray(spec.state_low), np.asarray(spec.state_high))
    return out


def _set_state(env: gym.Env, spec: GymEnvSpec, state: np.ndarray) -> None:
    env.unwrapped.state = _normalize_state(spec, state).astype(np.float64)


def _sample_states(rng: np.random.Generator, low: tuple[float, ...], high: tuple[float, ...], count: int) -> np.ndarray:
    return rng.uniform(np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64), size=(count, len(low)))


def _step_from_state(env: gym.Env, spec: GymEnvSpec, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
    _set_state(env, spec, state)
    _, reward, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
    return np.asarray(env.unwrapped.state, dtype=np.float64).copy(), float(reward)


def _collect_transition_data(env: gym.Env, spec: GymEnvSpec, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    states = _sample_states(rng, spec.train_state_low, spec.train_state_high, count)
    actions = rng.uniform(
        spec.action_low * spec.train_action_scale,
        spec.action_high * spec.train_action_scale,
        size=(count, 1),
    )
    features = []
    targets = []
    for state, action in zip(states, actions):
        next_state, reward = _step_from_state(env, spec, state, action)
        features.append(_feature_vector(spec.name, state, action))
        delta = _normalize_state(spec, next_state - state)
        if spec.name == "Pendulum-v1":
            delta[0] = ((next_state[0] - state[0] + np.pi) % (2.0 * np.pi)) - np.pi
        targets.append(np.concatenate([delta, np.asarray([reward], dtype=np.float64)]))
    return np.vstack(features), np.vstack(targets)


def _fit_bootstrap_ensemble(
    env_name: str,
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    ensemble_size: int,
    ridge: float,
    randomized_prior_scale: float,
) -> BootstrapDynamicsEnsemble:
    rng = np.random.default_rng(seed)
    weights = []
    prior_weights = []
    reg = ridge * np.eye(features.shape[1], dtype=np.float64)
    for _ in range(ensemble_size):
        idx = rng.integers(0, features.shape[0], size=features.shape[0])
        x = features[idx]
        y = targets[idx]
        prior = rng.normal(
            0.0,
            randomized_prior_scale / np.sqrt(features.shape[1]),
            size=(features.shape[1], targets.shape[1]),
        )
        weights.append(np.linalg.solve(x.T @ x + reg, x.T @ (y - x @ prior)))
        prior_weights.append(prior)
    return BootstrapDynamicsEnsemble(
        env_name=env_name,
        weights=np.stack(weights),
        prior_weights=np.stack(prior_weights),
        ridge=ridge,
        randomized_prior_scale=randomized_prior_scale,
    )


def _predict_step(model: BootstrapDynamicsEnsemble, spec: GymEnvSpec, state: np.ndarray, action: np.ndarray, member: int) -> tuple[np.ndarray, float]:
    features = _feature_vector(spec.name, state, action)
    pred = features @ model.weights[member] + features @ model.prior_weights[member]
    next_state = _normalize_state(spec, state + pred[: spec.state_dim])
    return next_state, float(pred[-1])


def _predict_returns(model: BootstrapDynamicsEnsemble, spec: GymEnvSpec, start_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    out = np.zeros((actions.shape[0], model.ensemble_size), dtype=np.float64)
    for candidate_id in range(actions.shape[0]):
        for member in range(model.ensemble_size):
            state = np.asarray(start_state, dtype=np.float64).copy()
            total = 0.0
            for action in actions[candidate_id]:
                state, reward = _predict_step(model, spec, state, action, member)
                total += reward
            out[candidate_id, member] = total
    return out


def _true_return(env: gym.Env, spec: GymEnvSpec, start_state: np.ndarray, actions: np.ndarray) -> float:
    _set_state(env, spec, start_state)
    total = 0.0
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        total += float(reward)
        if terminated or truncated:
            break
    return total


def _true_returns(env: gym.Env, spec: GymEnvSpec, start_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.asarray([_true_return(env, spec, start_state, candidate) for candidate in actions], dtype=np.float64)


def _scripted_action(spec: GymEnvSpec, state: np.ndarray) -> float:
    if spec.name == "Pendulum-v1":
        theta, theta_dot = float(state[0]), float(state[1])
        return float(np.clip(-2.0 * np.sin(theta) - 0.35 * theta_dot, spec.action_low, spec.action_high))
    velocity = float(state[1])
    direction = 1.0 if velocity >= 0.0 else -1.0
    return float(np.clip(direction, spec.action_low, spec.action_high))


def _scripted_sequence(env: gym.Env, spec: GymEnvSpec, start_state: np.ndarray) -> np.ndarray:
    _set_state(env, spec, start_state)
    actions = []
    for _ in range(spec.horizon):
        state = np.asarray(env.unwrapped.state, dtype=np.float64).copy()
        action = np.asarray([_scripted_action(spec, state)], dtype=np.float64)
        actions.append(action)
        env.step(action.astype(np.float32))
    return np.asarray(actions, dtype=np.float64)


def _smooth_random_sequences(rng: np.random.Generator, spec: GymEnvSpec, count: int) -> np.ndarray:
    raw = rng.uniform(
        spec.action_low * spec.candidate_action_scale,
        spec.action_high * spec.candidate_action_scale,
        size=(count, spec.horizon, 1),
    )
    if spec.horizon > 2:
        raw[:, 1:-1, :] = 0.25 * raw[:, :-2, :] + 0.5 * raw[:, 1:-1, :] + 0.25 * raw[:, 2:, :]
    return np.clip(raw, spec.action_low, spec.action_high)


def _cem_mean_sequence(
    model: BootstrapDynamicsEnsemble,
    spec: GymEnvSpec,
    start_state: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    population: int,
    elites: int,
) -> np.ndarray:
    mean = np.zeros((spec.horizon, 1), dtype=np.float64)
    std = np.full((spec.horizon, 1), (spec.action_high - spec.action_low) * 0.45, dtype=np.float64)
    best_sequence = mean.copy()
    best_score = -np.inf
    for _ in range(iterations):
        samples = rng.normal(mean, std, size=(population, spec.horizon, 1))
        samples = np.clip(samples, spec.action_low, spec.action_high)
        scores = _predict_returns(model, spec, start_state, samples).mean(axis=1)
        elite_idx = np.argsort(scores)[-elites:]
        elites_arr = samples[elite_idx]
        mean = elites_arr.mean(axis=0)
        std = np.maximum(elites_arr.std(axis=0), 0.08 * (spec.action_high - spec.action_low))
        if float(scores[elite_idx[-1]]) > best_score:
            best_score = float(scores[elite_idx[-1]])
            best_sequence = samples[elite_idx[-1]]
    return np.clip(best_sequence, spec.action_low, spec.action_high)


def _candidate_pool(
    spec: GymEnvSpec,
    rng: np.random.Generator,
    max_n: int,
) -> np.ndarray:
    zeros = np.zeros((spec.horizon, 1), dtype=np.float64)
    positive = np.full((spec.horizon, 1), spec.action_high, dtype=np.float64)
    negative = np.full((spec.horizon, 1), spec.action_low, dtype=np.float64)
    alternating = np.asarray(
        [[spec.action_high if step % 2 == 0 else spec.action_low] for step in range(spec.horizon)],
        dtype=np.float64,
    )
    fixed = np.stack([zeros, positive, negative, alternating])
    if max_n < fixed.shape[0]:
        raise ValueError("max_n must be at least the number of generic control candidates")
    random_part = _smooth_random_sequences(rng, spec, max_n - fixed.shape[0])
    actions = np.concatenate([fixed, random_part], axis=0)
    order = rng.permutation(actions.shape[0])
    return actions[order]


def _standalone_baselines(
    env: gym.Env,
    spec: GymEnvSpec,
    model: BootstrapDynamicsEnsemble,
    start_state: np.ndarray,
    rng: np.random.Generator,
    config: GymBenchmarkConfig,
) -> dict[str, np.ndarray]:
    return {
        "scripted_teacher": _scripted_sequence(env, spec, start_state),
        "cem_mean_mpc": _cem_mean_sequence(
            model,
            spec,
            start_state,
            rng,
            iterations=config.cem_iterations,
            population=config.cem_population,
            elites=config.cem_elites,
        ),
    }


def _fit_calibration_gate(
    env: gym.Env,
    spec: GymEnvSpec,
    model: BootstrapDynamicsEnsemble,
    rng: np.random.Generator,
    config: GymBenchmarkConfig,
) -> tuple[float, str, dict[str, float]]:
    predictions = []
    true_values = []
    starts = _sample_states(rng, spec.train_state_low, spec.train_state_high, config.calibration_problems)
    for start in starts:
        actions = _smooth_random_sequences(rng, spec, config.calibration_candidates)
        predictions.append(_predict_returns(model, spec, start, actions))
        true_values.append(_true_returns(env, spec, start, actions))
    beta = fit_calibration_beta(np.vstack(predictions), np.concatenate(true_values), quantile=0.85)
    validation_scores: dict[str, list[float]] = {method: [] for method in GATE_CANDIDATES}
    for problem, (prediction_matrix, true_returns) in enumerate(zip(predictions, true_values)):
        sampled_member_ids = rng.integers(0, model.ensemble_size, size=prediction_matrix.shape[0])
        for method in GATE_CANDIDATES:
            method_rng = np.random.default_rng(470_000 + problem * 37 + len(method))
            selected = select_candidate(
                prediction_matrix,
                true_returns,
                method=method,
                rng=method_rng,
                beta=beta,
                sampled_member_ids=sampled_member_ids,
            )
            validation_scores[method].append(float(true_returns[selected.index]))
    mean_scores = {method: float(np.mean(values)) for method, values in validation_scores.items()}
    best_score = max(mean_scores.values())
    near_best = {method for method, score in mean_scores.items() if score >= best_score - GATE_NEAR_TIE_MARGIN * max(1.0, abs(best_score))}
    gate_method = next(method for method in GATE_TIE_PREFERENCE if method in near_best)
    return beta, gate_method, mean_scores


def _action_ood_mass(spec: GymEnvSpec, actions: np.ndarray) -> np.ndarray:
    threshold = max(abs(spec.action_low), abs(spec.action_high)) * spec.train_action_scale
    return np.mean(np.abs(actions[..., 0]) > threshold, axis=1)


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    skip = set(keys) | {"seed", "problem", "selected_index", "uses_truth"}
    metric_keys = [key for key in rows[0] if key not in skip and isinstance(rows[0][key], (int, float, np.floating))]
    out = []
    for group_key, group in sorted(groups.items()):
        row: dict[str, Any] = {key: group_key[idx] for idx, key in enumerate(keys)}
        row["replicates"] = len(group)
        for metric in metric_keys:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_stderr"] = stderr(values)
        out.append(row)
    return out


def _paired(rows: list[dict[str, Any]], high_n: int) -> list[dict[str, Any]]:
    out = []
    comparisons = [
        ("gate_minus_sample", GATED_METHOD, "posterior_sample_bon"),
        ("gate_minus_lcb", GATED_METHOD, "calibrated_pessimistic_bon"),
        ("gate_minus_mean", GATED_METHOD, "mean_bon"),
        ("sample_minus_lcb", "posterior_sample_bon", "calibrated_pessimistic_bon"),
        ("sample_minus_mean", "posterior_sample_bon", "mean_bon"),
        ("ucb_minus_lcb", "ucb_bon", "calibrated_pessimistic_bon"),
        ("gate_minus_teacher", GATED_METHOD, "scripted_teacher"),
        ("gate_minus_cem", GATED_METHOD, "cem_mean_mpc"),
        ("lcb_minus_teacher", "calibrated_pessimistic_bon", "scripted_teacher"),
        ("lcb_minus_cem", "calibrated_pessimistic_bon", "cem_mean_mpc"),
    ]
    for env_name in sorted({row["env_name"] for row in rows}):
        problem_rows: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        for row in rows:
            if row["env_name"] == env_name and int(row["n_candidates"]) == high_n:
                problem_rows.setdefault((int(row["seed"]), int(row["problem"])), {})[row["method"]] = row
        for comparison, lhs, rhs in comparisons:
            diffs = []
            regret_diffs = []
            for methods in problem_rows.values():
                if lhs in methods and rhs in methods:
                    diffs.append(float(methods[lhs]["selected_true_return"]) - float(methods[rhs]["selected_true_return"]))
                    regret_diffs.append(float(methods[lhs]["regret_to_candidate_oracle"]) - float(methods[rhs]["regret_to_candidate_oracle"]))
            arr = np.asarray(diffs, dtype=np.float64)
            reg = np.asarray(regret_diffs, dtype=np.float64)
            se = stderr(arr.tolist())
            out.append(
                {
                    "env_name": env_name,
                    "n_candidates": high_n,
                    "comparison": comparison,
                    "paired_replicates": int(arr.size),
                    "true_return_gap_mean": float(arr.mean()),
                    "true_return_gap_stderr": se,
                    "true_return_gap_ci95_low": float(arr.mean() - 1.96 * se),
                    "true_return_gap_ci95_high": float(arr.mean() + 1.96 * se),
                    "regret_gap_mean": float(reg.mean()),
                    "lhs_worse_rate": float(np.mean(arr < 0.0)),
                }
            )
    return out


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_value(rows: list[dict[str, str]], env_name: str, n: int, method: str, metric: str) -> float:
    for row in rows:
        if row["env_name"] == env_name and int(row["n_candidates"]) == n and row["method"] == method:
            return float(row[f"{metric}_mean"])
    raise KeyError((env_name, n, method, metric))


def _write_macros(path: Path, config: GymBenchmarkConfig, summary_path: Path, paired_path: Path) -> None:
    summary = _read_csv(summary_path)
    paired = _read_csv(paired_path)
    high_n = config.max_n
    envs = sorted({row["env_name"] for row in summary})

    def mean_metric(method: str, metric: str) -> float:
        return float(np.mean([_summary_value(summary, env_name, high_n, method, metric) for env_name in envs]))

    sample_lcb_rows = [row for row in paired if row["comparison"] == "sample_minus_lcb"]
    gate_sample_rows = [row for row in paired if row["comparison"] == "gate_minus_sample"]
    lcb_wins = sum(float(row["true_return_gap_mean"]) < 0.0 for row in sample_lcb_rows)
    gate_wins = sum(float(row["true_return_gap_mean"]) > 0.0 for row in gate_sample_rows)
    macros: dict[str, float | int | str] = {
        "VFourGymEnvironments": len(envs),
        "VFourGymProblems": len(config.seeds) * config.test_problems * len(envs),
        "VFourGymHighN": high_n,
        "VFourGymEnsembleSize": config.ensemble_size,
        "VFourGymGateReturn": mean_metric(GATED_METHOD, "selected_true_return"),
        "VFourGymSampleReturn": mean_metric("posterior_sample_bon", "selected_true_return"),
        "VFourGymMeanReturn": mean_metric("mean_bon", "selected_true_return"),
        "VFourGymLCBReturn": mean_metric("calibrated_pessimistic_bon", "selected_true_return"),
        "VFourGymTeacherReturn": mean_metric("scripted_teacher", "selected_true_return"),
        "VFourGymCEMReturn": mean_metric("cem_mean_mpc", "selected_true_return"),
        "VFourGymGateRegret": mean_metric(GATED_METHOD, "regret_to_candidate_oracle"),
        "VFourGymSampleRegret": mean_metric("posterior_sample_bon", "regret_to_candidate_oracle"),
        "VFourGymMeanRegret": mean_metric("mean_bon", "regret_to_candidate_oracle"),
        "VFourGymLCBRegret": mean_metric("calibrated_pessimistic_bon", "regret_to_candidate_oracle"),
        "VFourGymGateWinEnvs": int(gate_wins),
        "VFourGymLCBWinEnvs": int(lcb_wins),
        "VFourGymGateSampleGap": float(np.mean([float(row["true_return_gap_mean"]) for row in gate_sample_rows])),
        "VFourGymGateSampleCILow": float(np.mean([float(row["true_return_gap_ci95_low"]) for row in gate_sample_rows])),
        "VFourGymGateSampleCIHigh": float(np.mean([float(row["true_return_gap_ci95_high"]) for row in gate_sample_rows])),
        "VFourGymSampleLCBGap": float(np.mean([float(row["true_return_gap_mean"]) for row in sample_lcb_rows])),
        "VFourGymSampleLCBCILow": float(np.mean([float(row["true_return_gap_ci95_low"]) for row in sample_lcb_rows])),
        "VFourGymSampleLCBCIHigh": float(np.mean([float(row["true_return_gap_ci95_high"]) for row in sample_lcb_rows])),
    }
    lines = []
    for name, value in macros.items():
        if isinstance(value, int):
            rendered = str(value)
        elif isinstance(value, float):
            rendered = f"{value:.3f}"
        else:
            rendered = str(value)
        lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(summary_path: Path, figure_dir: Path, high_n: int) -> dict[str, Path]:
    rows = _read_csv(summary_path)
    methods = [
        ("scripted_teacher", "teacher"),
        ("cem_mean_mpc", "CEM"),
        ("mean_bon", "mean"),
        ("posterior_sample_bon", "sample"),
        (GATED_METHOD, "gate"),
        ("ucb_bon", "UCB"),
        ("calibrated_pessimistic_bon", "LCB"),
        ("oracle", "oracle"),
    ]
    envs = sorted({row["env_name"] for row in rows})
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for env_idx, env_name in enumerate(envs):
        values = [_summary_value(rows, env_name, high_n, method, "selected_true_return") for method, _ in methods]
        ax.bar(x + (env_idx - 0.5) * width, values, width=width, label=env_name.replace("Continuous", "Cont."))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in methods], rotation=20, ha="right")
    ax.set_ylabel("true environment return")
    ax.set_title(f"Gymnasium benchmark selected return at N={high_n}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    return_path = figure_dir / "v4_gym_selected_return.png"
    fig.savefig(return_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for env_idx, env_name in enumerate(envs):
        values = [_summary_value(rows, env_name, high_n, method, "regret_to_candidate_oracle") for method, _ in methods]
        ax.bar(x + (env_idx - 0.5) * width, values, width=width, label=env_name.replace("Continuous", "Cont."))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in methods], rotation=20, ha="right")
    ax.set_ylabel("regret to in-pool oracle")
    ax.set_title(f"Gymnasium benchmark regret at N={high_n}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    regret_path = figure_dir / "v4_gym_oracle_regret.png"
    fig.savefig(regret_path, dpi=180)
    plt.close(fig)
    return {"gym_return_figure": return_path, "gym_regret_figure": regret_path}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol(config: GymBenchmarkConfig, outputs: dict[str, Path]) -> dict[str, Any]:
    return {
        "status": "FROZEN",
        "benchmark": "Gymnasium classic-control real environment dynamics bridge",
        "config": asdict(config),
        "env_specs": {name: asdict(ENV_SPECS[name]) for name in config.env_names},
        "methods": BENCHMARK_METHODS,
        "validation_gate_candidates": GATE_CANDIDATES,
        "validation_gate_tie_preference": GATE_TIE_PREFERENCE,
        "validation_gate_near_tie_margin": GATE_NEAR_TIE_MARGIN,
        "candidate_pool": "shared shuffled random-plus-generic control action sequences",
        "standalone_baselines": STANDALONE_BASELINES,
        "claim_gates": {
            "real_benchmark_present": True,
            "standard_envs": list(config.env_names),
            "uses_true_environment_returns_for_final_measurement": True,
            "randomized_prior_bootstrap_ensemble": True,
            "selector_gate_fit_on_calibration_only": True,
            "standalone_baselines_not_inserted_into_selector_pool": True,
            "final_evidence_after_protocol_freeze": True,
            "not_claimed": [
                "MuJoCo SOTA",
                "D4RL SOTA",
                "real robot validation",
                "large-scale video world-model dominance",
            ],
        },
        "artifact_hashes": {name: _sha256(path) for name, path in outputs.items() if path.exists() and path.is_file()},
    }


def run_gym_benchmark(config: GymBenchmarkConfig, output_dir: Path, figure_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for env_id, env_name in enumerate(config.env_names):
        spec = ENV_SPECS[env_name]
        env = gym.make(env_name)
        try:
            for seed in config.seeds:
                rng = np.random.default_rng(500_000 + env_id * 20_000 + seed)
                env.reset(seed=700_000 + seed)
                features, targets = _collect_transition_data(env, spec, rng, config.train_transitions)
                model = _fit_bootstrap_ensemble(
                    env_name,
                    features,
                    targets,
                    seed=510_000 + env_id * 20_000 + seed,
                    ensemble_size=config.ensemble_size,
                    ridge=config.ridge,
                    randomized_prior_scale=config.randomized_prior_scale,
                )
                beta, gate_method, gate_scores = _fit_calibration_gate(env, spec, model, rng, config)
                calibration_rows.append(
                    {
                        "env_name": env_name,
                        "seed": seed,
                        "calibration_beta": beta,
                        "validation_gate_method": gate_method,
                        **{f"{method}_validation_return": score for method, score in gate_scores.items()},
                        "train_transitions": config.train_transitions,
                        "calibration_problems": config.calibration_problems,
                        "calibration_candidates": config.calibration_candidates,
                    }
                )
                starts = _sample_states(rng, spec.test_state_low, spec.test_state_high, config.test_problems)
                for problem, start_state in enumerate(starts):
                    actions = _candidate_pool(spec, rng, config.max_n)
                    standalone_actions = _standalone_baselines(env, spec, model, start_state, rng, config)
                    standalone_predictions = {
                        method: _predict_returns(model, spec, start_state, action_sequence[None, ...])[0]
                        for method, action_sequence in standalone_actions.items()
                    }
                    standalone_true_returns = {
                        method: _true_return(env, spec, start_state, action_sequence)
                        for method, action_sequence in standalone_actions.items()
                    }
                    standalone_ood = {
                        method: float(_action_ood_mass(spec, action_sequence[None, ...])[0])
                        for method, action_sequence in standalone_actions.items()
                    }
                    predictions = _predict_returns(model, spec, start_state, actions)
                    true_returns = _true_returns(env, spec, start_state, actions)
                    action_ood = _action_ood_mass(spec, actions)
                    sampled_member_ids = rng.integers(0, config.ensemble_size, size=config.max_n)
                    for n_candidates in config.n_values:
                        pred_prefix = predictions[:n_candidates]
                        true_prefix = true_returns[:n_candidates]
                        action_ood_prefix = action_ood[:n_candidates]
                        prefix_oracle_return = float(np.max(true_prefix))
                        full_pool_oracle_return = float(np.max(true_returns))
                        for method in BENCHMARK_METHODS:
                            method_rng = np.random.default_rng(520_000 + env_id * 20_000 + seed * 997 + problem * 31 + n_candidates + len(method))
                            if method in STANDALONE_BASELINES:
                                baseline_prediction = standalone_predictions[method]
                                pred_mean = float(baseline_prediction.mean())
                                pred_std = float(baseline_prediction.std())
                                selected_true = float(standalone_true_returns[method])
                                result_index = -1
                                uses_truth = method == "scripted_teacher"
                                selected_is_teacher = float(method == "scripted_teacher")
                                selected_is_cem = float(method == "cem_mean_mpc")
                                selected_ood = standalone_ood[method]
                                sample_over_mean = float(
                                    baseline_prediction[int(method_rng.integers(0, config.ensemble_size))] - pred_mean
                                )
                            else:
                                selector_method = gate_method if method == GATED_METHOD else method
                                selected = select_candidate(
                                    pred_prefix,
                                    true_prefix,
                                    method=selector_method,
                                    rng=method_rng,
                                    beta=beta,
                                    sampled_member_ids=sampled_member_ids[:n_candidates],
                                )
                                result_index = selected.index
                                uses_truth = selected.uses_truth
                                pred_mean = float(pred_prefix[result_index].mean())
                                pred_std = float(pred_prefix[result_index].std())
                                selected_true = float(true_prefix[result_index])
                                selected_is_teacher = 0.0
                                selected_is_cem = 0.0
                                selected_ood = float(action_ood_prefix[result_index])
                                sample_over_mean = float(pred_prefix[result_index, sampled_member_ids[result_index]] - pred_mean)
                            rows.append(
                                {
                                    "env_name": env_name,
                                    "seed": seed,
                                    "problem": problem,
                                    "n_candidates": n_candidates,
                                    "method": method,
                                    "selected_index": result_index,
                                    "uses_truth": uses_truth,
                                    "validation_gate_method": gate_method if method == GATED_METHOD else "",
                                    "selected_true_return": selected_true,
                                    "regret_to_candidate_oracle": float(prefix_oracle_return - selected_true),
                                    "candidate_oracle_return": prefix_oracle_return,
                                    "regret_to_full_pool_oracle": float(full_pool_oracle_return - selected_true),
                                    "full_pool_oracle_return": full_pool_oracle_return,
                                    "selected_pred_mean": pred_mean,
                                    "selected_pred_std": pred_std,
                                    "sample_over_mean": sample_over_mean,
                                    "selected_action_ood_mass": selected_ood,
                                    "selected_is_teacher": selected_is_teacher,
                                    "selected_is_cem": selected_is_cem,
                                    "calibration_beta": beta,
                                }
                            )
        finally:
            env.close()

    summary_rows = _aggregate(rows, ("env_name", "n_candidates", "method"))
    paired_rows = _paired(rows, config.max_n)
    paths = {
        "gym_results": output_dir / "gym_results.csv",
        "gym_summary": output_dir / "gym_summary.csv",
        "gym_paired_effects": output_dir / "gym_paired_effects.csv",
        "gym_calibration": output_dir / "gym_calibration.csv",
        "generated_results": output_dir / "generated_v4_results.tex",
    }
    _write_rows(paths["gym_results"], rows)
    _write_rows(paths["gym_summary"], summary_rows)
    _write_rows(paths["gym_paired_effects"], paired_rows)
    _write_rows(paths["gym_calibration"], calibration_rows)
    _write_macros(paths["generated_results"], config, paths["gym_summary"], paths["gym_paired_effects"])
    paths.update(_plot_summary(paths["gym_summary"], figure_dir, config.max_n))
    protocol_path = output_dir / "protocol_freeze.json"
    paths["protocol_freeze"] = protocol_path
    protocol_path.write_text(json.dumps(_protocol(config, paths), indent=2) + "\n", encoding="utf-8")
    return paths
