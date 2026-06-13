from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def action_features(actions: FloatArray) -> FloatArray:
    """Return lightweight trajectory features for bootstrap return ensembles."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError("actions must have shape (num_candidates, horizon)")
    horizon = actions.shape[1]
    time = np.linspace(-1.0, 1.0, horizon, dtype=np.float64)
    cumsum = np.cumsum(actions, axis=1)
    velocity_proxy = np.cumsum(np.tanh(actions), axis=1)
    positive = np.maximum(actions, 0.0)
    negative = np.minimum(actions, 0.0)
    features = np.column_stack(
        [
            actions.mean(axis=1),
            actions.std(axis=1),
            actions.max(axis=1),
            actions.min(axis=1),
            actions.sum(axis=1),
            np.mean(actions**2, axis=1),
            positive.sum(axis=1),
            negative.sum(axis=1),
            np.max(np.abs(actions), axis=1),
            cumsum[:, -1],
            cumsum.max(axis=1),
            cumsum.min(axis=1),
            velocity_proxy[:, -1],
            velocity_proxy.max(axis=1),
            np.mean(np.maximum(velocity_proxy - 1.8, 0.0), axis=1),
            np.mean(np.maximum(cumsum - 2.2, 0.0), axis=1),
            actions[:, : max(1, horizon // 2)].sum(axis=1),
            actions[:, horizon // 2 :].sum(axis=1),
            actions @ time,
            actions @ (time**2),
        ]
    )
    return features.astype(np.float64)


@dataclass(frozen=True)
class BootstrapReturnEnsemble:
    weights: FloatArray
    feature_mean: FloatArray
    feature_scale: FloatArray

    def predict(self, actions: FloatArray) -> FloatArray:
        features = action_features(actions)
        normalized = (features - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(features.shape[0]), normalized])
        return (design @ self.weights.T).astype(np.float64)


def fit_bootstrap_return_ensemble(
    actions: FloatArray,
    returns: FloatArray,
    *,
    seed: int,
    ensemble_size: int = 24,
    ridge: float = 2.0e-2,
    bootstrap_fraction: float = 0.85,
) -> BootstrapReturnEnsemble:
    """Fit a CPU-light bootstrap ridge ensemble on synthetic rollout returns."""

    actions = np.asarray(actions, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError("actions must have shape (num_candidates, horizon)")
    if returns.shape != (actions.shape[0],):
        raise ValueError("returns must have one value per action sequence")
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")
    if not (0.0 < bootstrap_fraction <= 1.0):
        raise ValueError("bootstrap_fraction must be in (0, 1]")

    features = action_features(actions)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0) + 1e-6
    normalized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(features.shape[0]), normalized])
    weights = np.zeros((ensemble_size, design.shape[1]), dtype=np.float64)
    sample_size = max(design.shape[1] + 2, int(round(bootstrap_fraction * actions.shape[0])))
    rng = np.random.default_rng(seed)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    for member in range(ensemble_size):
        indices = rng.integers(0, actions.shape[0], size=sample_size)
        x = design[indices]
        y = returns[indices]
        lhs = x.T @ x + ridge * penalty
        rhs = x.T @ y
        weights[member] = np.linalg.solve(lhs, rhs)
    return BootstrapReturnEnsemble(weights=weights, feature_mean=feature_mean, feature_scale=feature_scale)
