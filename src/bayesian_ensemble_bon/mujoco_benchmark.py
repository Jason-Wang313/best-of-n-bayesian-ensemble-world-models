from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from bayesian_ensemble_bon.gym_benchmark import _aggregate, _read_csv, _sha256, _summary_value, _write_rows
from bayesian_ensemble_bon.metrics import stderr
from bayesian_ensemble_bon.selectors import METHODS, fit_calibration_beta, select_candidate


MUTLI_SEED_NOTE = "Each problem resets the simulator from a stored qpos/qvel snapshot."
MUJOCO_BASELINES = ["cem_mean_mpc"]
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
MUJOCO_METHODS = METHODS + [GATED_METHOD] + MUJOCO_BASELINES
ROLLOUT_OBS_CLIP = 20.0


@dataclass(frozen=True)
class MuJoCoEnvSpec:
    name: str
    horizon: int
    train_action_scale: float = 0.45
    candidate_action_scale: float = 1.0


@dataclass(frozen=True)
class MuJoCoBenchmarkConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    env_names: tuple[str, ...] = ("InvertedPendulum-v5", "Reacher-v5")
    n_values: tuple[int, ...] = (8, 32, 96)
    train_transitions: int = 1200
    calibration_problems: int = 4
    calibration_candidates: int = 32
    test_problems: int = 6
    ensemble_size: int = 8
    ridge: float = 5e-3
    randomized_prior_scale: float = 0.10
    cem_iterations: int = 2
    cem_population: int = 48
    cem_elites: int = 8

    @property
    def max_n(self) -> int:
        return max(self.n_values)


MUJOCO_SPECS = {
    "InvertedPendulum-v5": MuJoCoEnvSpec(name="InvertedPendulum-v5", horizon=40),
    "Reacher-v5": MuJoCoEnvSpec(name="Reacher-v5", horizon=35),
    "Swimmer-v5": MuJoCoEnvSpec(name="Swimmer-v5", horizon=50),
}


@dataclass(frozen=True)
class StandardizedBootstrapEnsemble:
    env_name: str
    weights: np.ndarray
    prior_weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    ridge: float
    randomized_prior_scale: float
    obs_dim: int

    @property
    def ensemble_size(self) -> int:
        return int(self.weights.shape[0])


def _feature_vector(obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    obs = np.nan_to_num(np.asarray(obs, dtype=np.float64).reshape(-1), nan=0.0, posinf=ROLLOUT_OBS_CLIP, neginf=-ROLLOUT_OBS_CLIP)
    obs = np.clip(obs, -ROLLOUT_OBS_CLIP, ROLLOUT_OBS_CLIP)
    action = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(-1), nan=0.0, posinf=ROLLOUT_OBS_CLIP, neginf=-ROLLOUT_OBS_CLIP)
    action = np.clip(action, -ROLLOUT_OBS_CLIP, ROLLOUT_OBS_CLIP)
    return np.concatenate(
        [
            np.ones(1, dtype=np.float64),
            obs,
            action,
            obs**2,
            action**2,
            np.outer(obs, action).reshape(-1),
        ]
    )


def _feature_matrix(obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    obs = np.nan_to_num(np.asarray(obs, dtype=np.float64), nan=0.0, posinf=ROLLOUT_OBS_CLIP, neginf=-ROLLOUT_OBS_CLIP)
    obs = np.clip(obs, -ROLLOUT_OBS_CLIP, ROLLOUT_OBS_CLIP)
    action = np.nan_to_num(np.asarray(action, dtype=np.float64), nan=0.0, posinf=ROLLOUT_OBS_CLIP, neginf=-ROLLOUT_OBS_CLIP)
    action = np.clip(action, -ROLLOUT_OBS_CLIP, ROLLOUT_OBS_CLIP)
    interaction = (obs[:, :, None] * action[:, None, :]).reshape(obs.shape[0], -1)
    return np.concatenate(
        [
            np.ones((obs.shape[0], 1), dtype=np.float64),
            obs,
            action,
            obs**2,
            action**2,
            interaction,
        ],
        axis=1,
    )


def _snapshot(env: gym.Env) -> dict[str, np.ndarray | float]:
    unwrapped = env.unwrapped
    return {
        "qpos": np.asarray(unwrapped.data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(unwrapped.data.qvel, dtype=np.float64).copy(),
        "time": float(unwrapped.data.time),
    }


def _restore(env: gym.Env, snapshot: dict[str, np.ndarray | float]) -> np.ndarray:
    if hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = 0
    unwrapped = env.unwrapped
    unwrapped.set_state(np.asarray(snapshot["qpos"], dtype=np.float64), np.asarray(snapshot["qvel"], dtype=np.float64))
    unwrapped.data.time = float(snapshot["time"])
    return np.asarray(unwrapped._get_obs(), dtype=np.float64).copy()


def _collect_transition_data(
    env: gym.Env,
    rng: np.random.Generator,
    count: int,
    train_action_scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
    obs = np.asarray(obs, dtype=np.float64)
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    action_low = low * train_action_scale
    action_high = high * train_action_scale
    while len(features) < count:
        action = rng.uniform(action_low, action_high)
        next_obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        next_obs = np.asarray(next_obs, dtype=np.float64)
        features.append(_feature_vector(obs, action))
        targets.append(np.concatenate([next_obs - obs, np.asarray([reward], dtype=np.float64)]))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
            obs = np.asarray(obs, dtype=np.float64)
    return np.vstack(features), np.vstack(targets), int(obs.shape[0])


def _fit_bootstrap_ensemble(
    env_name: str,
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    ensemble_size: int,
    ridge: float,
    randomized_prior_scale: float,
    obs_dim: int,
) -> StandardizedBootstrapEnsemble:
    rng = np.random.default_rng(seed)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0) + 1e-6
    target_mean = targets.mean(axis=0)
    target_scale = targets.std(axis=0) + 1e-6
    x_all = (features - feature_mean) / feature_scale
    y_all = (targets - target_mean) / target_scale
    reg = ridge * np.eye(x_all.shape[1], dtype=np.float64)
    weights = []
    prior_weights = []
    for _ in range(ensemble_size):
        idx = rng.integers(0, x_all.shape[0], size=x_all.shape[0])
        x = x_all[idx]
        y = y_all[idx]
        prior = rng.normal(
            0.0,
            randomized_prior_scale / np.sqrt(x_all.shape[1]),
            size=(x_all.shape[1], y_all.shape[1]),
        )
        weights.append(np.linalg.solve(x.T @ x + reg, x.T @ (y - x @ prior)))
        prior_weights.append(prior)
    return StandardizedBootstrapEnsemble(
        env_name=env_name,
        weights=np.stack(weights),
        prior_weights=np.stack(prior_weights),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        ridge=ridge,
        randomized_prior_scale=randomized_prior_scale,
        obs_dim=obs_dim,
    )


def _predict_step(model: StandardizedBootstrapEnsemble, obs: np.ndarray, action: np.ndarray, member: int) -> tuple[np.ndarray, float]:
    features = (_feature_vector(obs, action) - model.feature_mean) / model.feature_scale
    pred = features @ model.weights[member] + features @ model.prior_weights[member]
    pred = pred * model.target_scale + model.target_mean
    next_obs = np.asarray(obs, dtype=np.float64) + pred[: model.obs_dim]
    next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=ROLLOUT_OBS_CLIP, neginf=-ROLLOUT_OBS_CLIP)
    next_obs = np.clip(next_obs, -ROLLOUT_OBS_CLIP, ROLLOUT_OBS_CLIP)
    reward = float(np.nan_to_num(pred[-1], nan=-1e6, posinf=1e6, neginf=-1e6))
    return next_obs, reward


def _predict_returns(model: StandardizedBootstrapEnsemble, start_obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    out = np.zeros((actions.shape[0], model.ensemble_size), dtype=np.float64)
    for member in range(model.ensemble_size):
        obs = np.repeat(np.asarray(start_obs, dtype=np.float64)[None, :], actions.shape[0], axis=0)
        total = np.zeros(actions.shape[0], dtype=np.float64)
        for step in range(actions.shape[1]):
            features = (_feature_matrix(obs, actions[:, step, :]) - model.feature_mean) / model.feature_scale
            pred = features @ model.weights[member] + features @ model.prior_weights[member]
            pred = pred * model.target_scale + model.target_mean
            next_obs = obs + pred[:, : model.obs_dim]
            next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=ROLLOUT_OBS_CLIP, neginf=-ROLLOUT_OBS_CLIP)
            obs = np.clip(next_obs, -ROLLOUT_OBS_CLIP, ROLLOUT_OBS_CLIP)
            rewards = np.nan_to_num(pred[:, -1], nan=-1e6, posinf=1e6, neginf=-1e6)
            total += rewards
        out[:, member] = total
    return out


def _true_return(env: gym.Env, snapshot: dict[str, np.ndarray | float], actions: np.ndarray) -> float:
    _restore(env, snapshot)
    total = 0.0
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        total += float(reward)
        if terminated or truncated:
            break
    return total


def _true_returns(env: gym.Env, snapshot: dict[str, np.ndarray | float], actions: np.ndarray) -> np.ndarray:
    return np.asarray([_true_return(env, snapshot, candidate) for candidate in actions], dtype=np.float64)


def _sample_snapshots(
    env: gym.Env,
    rng: np.random.Generator,
    count: int,
    seed_base: int,
    train_action_scale: float,
) -> list[tuple[dict[str, np.ndarray | float], np.ndarray]]:
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    action_low = low * train_action_scale
    action_high = high * train_action_scale
    out = []
    for item in range(count):
        obs, _ = env.reset(seed=seed_base + item)
        obs = np.asarray(obs, dtype=np.float64)
        warmup = int(rng.integers(0, 6))
        for _ in range(warmup):
            action = rng.uniform(action_low, action_high)
            obs, _, terminated, truncated, _ = env.step(action.astype(np.float32))
            obs = np.asarray(obs, dtype=np.float64)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed_base + item + 10_000)
                obs = np.asarray(obs, dtype=np.float64)
                break
        out.append((_snapshot(env), obs.copy()))
    return out


def _smooth_sequences(rng: np.random.Generator, low: np.ndarray, high: np.ndarray, count: int, horizon: int) -> np.ndarray:
    raw = rng.uniform(low, high, size=(count, horizon, low.shape[0]))
    if horizon > 2:
        raw[:, 1:-1, :] = 0.25 * raw[:, :-2, :] + 0.5 * raw[:, 1:-1, :] + 0.25 * raw[:, 2:, :]
    return np.clip(raw, low, high)


def _sinusoid_sequences(rng: np.random.Generator, low: np.ndarray, high: np.ndarray, count: int, horizon: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    dim = low.shape[0]
    sequences = np.zeros((count, horizon, dim), dtype=np.float64)
    amplitude = rng.uniform(0.35, 1.0, size=(count, 1, dim)) * np.maximum(np.abs(low), np.abs(high))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(count, 1, dim))
    frequency = rng.choice(np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float64), size=(count, 1, dim))
    sequences[:] = amplitude * np.sin(2.0 * np.pi * frequency * t.reshape(1, horizon, 1) + phase)
    return np.clip(sequences, low, high)


def _candidate_pool(env: gym.Env, spec: MuJoCoEnvSpec, rng: np.random.Generator, max_n: int) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float64) * spec.candidate_action_scale
    high = np.asarray(env.action_space.high, dtype=np.float64) * spec.candidate_action_scale
    dim = low.shape[0]
    zeros = np.zeros((spec.horizon, dim), dtype=np.float64)
    positive = np.tile(high, (spec.horizon, 1))
    negative = np.tile(low, (spec.horizon, 1))
    alternating = np.asarray([high if step % 2 == 0 else low for step in range(spec.horizon)], dtype=np.float64)
    fixed = np.stack([zeros, positive, negative, alternating])
    if max_n < fixed.shape[0]:
        raise ValueError("max_n must be at least the number of generic control candidates")
    random_count = max_n - fixed.shape[0]
    smooth_count = random_count // 2
    sinusoid_count = random_count - smooth_count
    actions = np.concatenate(
        [
            fixed,
            _smooth_sequences(rng, low, high, smooth_count, spec.horizon),
            _sinusoid_sequences(rng, low, high, sinusoid_count, spec.horizon),
        ],
        axis=0,
    )
    return actions[rng.permutation(actions.shape[0])]


def _cem_mean_sequence(
    env: gym.Env,
    model: StandardizedBootstrapEnsemble,
    spec: MuJoCoEnvSpec,
    start_obs: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    population: int,
    elites: int,
) -> np.ndarray:
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    mean = np.zeros((spec.horizon, low.shape[0]), dtype=np.float64)
    std = np.full_like(mean, 0.45 * (high - low))
    best_sequence = mean.copy()
    best_score = -np.inf
    for _ in range(iterations):
        samples = rng.normal(mean, std, size=(population, spec.horizon, low.shape[0]))
        samples = np.clip(samples, low, high)
        scores = _predict_returns(model, start_obs, samples).mean(axis=1)
        elite_idx = np.argsort(scores)[-elites:]
        elite_samples = samples[elite_idx]
        mean = elite_samples.mean(axis=0)
        std = np.maximum(elite_samples.std(axis=0), 0.05 * (high - low))
        if float(scores[elite_idx[-1]]) > best_score:
            best_score = float(scores[elite_idx[-1]])
            best_sequence = samples[elite_idx[-1]]
    return np.clip(best_sequence, low, high)


def _fit_calibration_gate(
    env: gym.Env,
    spec: MuJoCoEnvSpec,
    model: StandardizedBootstrapEnsemble,
    rng: np.random.Generator,
    config: MuJoCoBenchmarkConfig,
    env_id: int,
    seed: int,
) -> tuple[float, str, dict[str, float]]:
    predictions = []
    true_values = []
    snapshots = _sample_snapshots(
        env,
        rng,
        config.calibration_problems,
        seed_base=810_000 + env_id * 20_000 + seed * 100,
        train_action_scale=spec.train_action_scale,
    )
    for snapshot, obs in snapshots:
        actions = _candidate_pool(env, spec, rng, config.calibration_candidates)
        predictions.append(_predict_returns(model, obs, actions))
        true_values.append(_true_returns(env, snapshot, actions))
    beta = fit_calibration_beta(np.vstack(predictions), np.concatenate(true_values), quantile=0.85)
    validation_scores: dict[str, list[float]] = {method: [] for method in GATE_CANDIDATES}
    for problem, (prediction_matrix, true_returns) in enumerate(zip(predictions, true_values)):
        sampled_member_ids = rng.integers(0, model.ensemble_size, size=prediction_matrix.shape[0])
        for method in GATE_CANDIDATES:
            method_rng = np.random.default_rng(870_000 + env_id * 20_000 + seed * 997 + problem * 37 + len(method))
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


def _action_ood_mass(env: gym.Env, spec: MuJoCoEnvSpec, actions: np.ndarray) -> np.ndarray:
    threshold = np.maximum(np.abs(env.action_space.low), np.abs(env.action_space.high)) * spec.train_action_scale
    return np.mean(np.abs(actions) > threshold.reshape(1, 1, -1), axis=(1, 2))


def _paired(rows: list[dict[str, Any]], high_n: int) -> list[dict[str, Any]]:
    out = []
    comparisons = [
        ("gate_minus_sample", GATED_METHOD, "posterior_sample_bon"),
        ("gate_minus_lcb", GATED_METHOD, "calibrated_pessimistic_bon"),
        ("gate_minus_mean", GATED_METHOD, "mean_bon"),
        ("sample_minus_lcb", "posterior_sample_bon", "calibrated_pessimistic_bon"),
        ("sample_minus_mean", "posterior_sample_bon", "mean_bon"),
        ("ucb_minus_lcb", "ucb_bon", "calibrated_pessimistic_bon"),
        ("gate_minus_cem", GATED_METHOD, "cem_mean_mpc"),
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


def _write_macros(path: Path, config: MuJoCoBenchmarkConfig, summary_path: Path, paired_path: Path) -> None:
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
    macros: dict[str, float | int] = {
        "VFourMujocoEnvironments": len(envs),
        "VFourMujocoProblems": len(config.seeds) * config.test_problems * len(envs),
        "VFourMujocoHighN": high_n,
        "VFourMujocoEnsembleSize": config.ensemble_size,
        "VFourMujocoGateReturn": mean_metric(GATED_METHOD, "selected_true_return"),
        "VFourMujocoSampleReturn": mean_metric("posterior_sample_bon", "selected_true_return"),
        "VFourMujocoMeanReturn": mean_metric("mean_bon", "selected_true_return"),
        "VFourMujocoLCBReturn": mean_metric("calibrated_pessimistic_bon", "selected_true_return"),
        "VFourMujocoCEMReturn": mean_metric("cem_mean_mpc", "selected_true_return"),
        "VFourMujocoGateRegret": mean_metric(GATED_METHOD, "regret_to_candidate_oracle"),
        "VFourMujocoSampleRegret": mean_metric("posterior_sample_bon", "regret_to_candidate_oracle"),
        "VFourMujocoMeanRegret": mean_metric("mean_bon", "regret_to_candidate_oracle"),
        "VFourMujocoLCBRegret": mean_metric("calibrated_pessimistic_bon", "regret_to_candidate_oracle"),
        "VFourMujocoGateWinEnvs": int(gate_wins),
        "VFourMujocoLCBWinEnvs": int(lcb_wins),
        "VFourMujocoGateSampleGap": float(np.mean([float(row["true_return_gap_mean"]) for row in gate_sample_rows])),
        "VFourMujocoGateSampleCILow": float(np.mean([float(row["true_return_gap_ci95_low"]) for row in gate_sample_rows])),
        "VFourMujocoGateSampleCIHigh": float(np.mean([float(row["true_return_gap_ci95_high"]) for row in gate_sample_rows])),
        "VFourMujocoSampleLCBGap": float(np.mean([float(row["true_return_gap_mean"]) for row in sample_lcb_rows])),
        "VFourMujocoSampleLCBCILow": float(np.mean([float(row["true_return_gap_ci95_low"]) for row in sample_lcb_rows])),
        "VFourMujocoSampleLCBCIHigh": float(np.mean([float(row["true_return_gap_ci95_high"]) for row in sample_lcb_rows])),
    }
    lines = []
    for name, value in macros.items():
        rendered = str(value) if isinstance(value, int) else f"{value:.3f}"
        lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(summary_path: Path, figure_dir: Path, high_n: int) -> dict[str, Path]:
    rows = _read_csv(summary_path)
    methods = [
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
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for env_idx, env_name in enumerate(envs):
        values = [_summary_value(rows, env_name, high_n, method, "selected_true_return") for method, _ in methods]
        ax.bar(x + (env_idx - (len(envs) - 1) / 2.0) * width, values, width=width, label=env_name.replace("-v5", ""))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in methods], rotation=20, ha="right")
    ax.set_ylabel("true environment return")
    ax.set_title(f"MuJoCo benchmark selected return at N={high_n}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    return_path = figure_dir / "v4_mujoco_selected_return.png"
    fig.savefig(return_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for env_idx, env_name in enumerate(envs):
        values = [_summary_value(rows, env_name, high_n, method, "regret_to_candidate_oracle") for method, _ in methods]
        ax.bar(x + (env_idx - (len(envs) - 1) / 2.0) * width, values, width=width, label=env_name.replace("-v5", ""))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in methods], rotation=20, ha="right")
    ax.set_ylabel("regret to in-pool oracle")
    ax.set_title(f"MuJoCo benchmark regret at N={high_n}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    regret_path = figure_dir / "v4_mujoco_oracle_regret.png"
    fig.savefig(regret_path, dpi=180)
    plt.close(fig)
    return {"mujoco_return_figure": return_path, "mujoco_regret_figure": regret_path}


def _protocol(config: MuJoCoBenchmarkConfig, outputs: dict[str, Path]) -> dict[str, Any]:
    return {
        "status": "FROZEN",
        "benchmark": "Gymnasium MuJoCo continuous-control world-model planning bridge",
        "note": MUTLI_SEED_NOTE,
        "config": asdict(config),
        "env_specs": {name: asdict(MUJOCO_SPECS[name]) for name in config.env_names},
        "methods": MUJOCO_METHODS,
        "validation_gate_candidates": GATE_CANDIDATES,
        "validation_gate_tie_preference": GATE_TIE_PREFERENCE,
        "validation_gate_near_tie_margin": GATE_NEAR_TIE_MARGIN,
        "candidate_pool": "shared shuffled smoothed-random, sinusoidal, and generic control action sequences",
        "claim_gates": {
            "recognized_mujoco_envs_present": True,
            "standard_envs": list(config.env_names),
            "uses_true_environment_returns_for_final_measurement": True,
            "randomized_prior_bootstrap_ensemble": True,
            "selector_gate_fit_on_calibration_only": True,
            "finite_model_rollout_clipping": ROLLOUT_OBS_CLIP,
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


def run_mujoco_benchmark(config: MuJoCoBenchmarkConfig, output_dir: Path, figure_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for env_id, env_name in enumerate(config.env_names):
        spec = MUJOCO_SPECS[env_name]
        env = gym.make(env_name)
        try:
            for seed in config.seeds:
                rng = np.random.default_rng(900_000 + env_id * 20_000 + seed)
                env.reset(seed=910_000 + seed)
                features, targets, obs_dim = _collect_transition_data(env, rng, config.train_transitions, spec.train_action_scale)
                model = _fit_bootstrap_ensemble(
                    env_name,
                    features,
                    targets,
                    seed=920_000 + env_id * 20_000 + seed,
                    ensemble_size=config.ensemble_size,
                    ridge=config.ridge,
                    randomized_prior_scale=config.randomized_prior_scale,
                    obs_dim=obs_dim,
                )
                beta, gate_method, gate_scores = _fit_calibration_gate(env, spec, model, rng, config, env_id, seed)
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
                starts = _sample_snapshots(
                    env,
                    rng,
                    config.test_problems,
                    seed_base=930_000 + env_id * 20_000 + seed * 100,
                    train_action_scale=spec.train_action_scale,
                )
                for problem, (snapshot, start_obs) in enumerate(starts):
                    actions = _candidate_pool(env, spec, rng, config.max_n)
                    cem_action = _cem_mean_sequence(
                        env,
                        model,
                        spec,
                        start_obs,
                        rng,
                        iterations=config.cem_iterations,
                        population=config.cem_population,
                        elites=config.cem_elites,
                    )
                    cem_prediction = _predict_returns(model, start_obs, cem_action[None, ...])[0]
                    cem_true_return = _true_return(env, snapshot, cem_action)
                    cem_ood = float(_action_ood_mass(env, spec, cem_action[None, ...])[0])
                    predictions = _predict_returns(model, start_obs, actions)
                    true_returns = _true_returns(env, snapshot, actions)
                    action_ood = _action_ood_mass(env, spec, actions)
                    sampled_member_ids = rng.integers(0, config.ensemble_size, size=config.max_n)
                    for n_candidates in config.n_values:
                        pred_prefix = predictions[:n_candidates]
                        true_prefix = true_returns[:n_candidates]
                        action_ood_prefix = action_ood[:n_candidates]
                        prefix_oracle_return = float(np.max(true_prefix))
                        full_pool_oracle_return = float(np.max(true_returns))
                        for method in MUJOCO_METHODS:
                            method_rng = np.random.default_rng(940_000 + env_id * 20_000 + seed * 997 + problem * 31 + n_candidates + len(method))
                            if method == "cem_mean_mpc":
                                result_index = -1
                                uses_truth = False
                                pred_mean = float(cem_prediction.mean())
                                pred_std = float(cem_prediction.std())
                                selected_true = float(cem_true_return)
                                selected_ood = cem_ood
                                sample_over_mean = float(cem_prediction[int(method_rng.integers(0, config.ensemble_size))] - pred_mean)
                                selected_is_cem = 1.0
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
                                selected_ood = float(action_ood_prefix[result_index])
                                sample_over_mean = float(pred_prefix[result_index, sampled_member_ids[result_index]] - pred_mean)
                                selected_is_cem = 0.0
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
                                    "selected_is_cem": selected_is_cem,
                                    "calibration_beta": beta,
                                }
                            )
        finally:
            env.close()

    summary_rows = _aggregate(rows, ("env_name", "n_candidates", "method"))
    paired_rows = _paired(rows, config.max_n)
    paths = {
        "mujoco_results": output_dir / "mujoco_results.csv",
        "mujoco_summary": output_dir / "mujoco_summary.csv",
        "mujoco_paired_effects": output_dir / "mujoco_paired_effects.csv",
        "mujoco_calibration": output_dir / "mujoco_calibration.csv",
        "generated_results": output_dir / "generated_v4_mujoco_results.tex",
    }
    _write_rows(paths["mujoco_results"], rows)
    _write_rows(paths["mujoco_summary"], summary_rows)
    _write_rows(paths["mujoco_paired_effects"], paired_rows)
    _write_rows(paths["mujoco_calibration"], calibration_rows)
    _write_macros(paths["generated_results"], config, paths["mujoco_summary"], paths["mujoco_paired_effects"])
    paths.update(_plot_summary(paths["mujoco_summary"], figure_dir, config.max_n))
    protocol_path = output_dir / "protocol_freeze.json"
    paths["protocol_freeze"] = protocol_path
    protocol_path.write_text(json.dumps(_protocol(config, paths), indent=2) + "\n", encoding="utf-8")
    return paths
