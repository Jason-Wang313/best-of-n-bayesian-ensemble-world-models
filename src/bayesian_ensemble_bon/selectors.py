from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

METHODS = [
    "random",
    "oracle",
    "mean_bon",
    "posterior_sample_bon",
    "thompson_bon",
    "ucb_bon",
    "pessimistic_bon",
    "quantile_bon",
    "calibrated_pessimistic_bon",
]


@dataclass(frozen=True)
class SelectionResult:
    index: int
    score: float
    method: str
    uses_truth: bool = False


def fit_calibration_beta(
    predictions: FloatArray,
    true_returns: FloatArray,
    quantile: float = 0.85,
) -> float:
    """Fit a scalar residual-to-ensemble-std multiplier on calibration data."""

    means = predictions.mean(axis=1)
    stds = predictions.std(axis=1) + 1e-8
    ratios = np.abs(means - true_returns) / stds
    beta = float(np.quantile(ratios, quantile))
    return float(np.clip(beta, 0.35, 4.0))


def _argmax_score(scores: FloatArray, method: str) -> SelectionResult:
    idx = int(np.argmax(scores))
    return SelectionResult(index=idx, score=float(scores[idx]), method=method)


def select_candidate(
    predictions: FloatArray,
    true_returns: FloatArray,
    method: str,
    rng: np.random.Generator,
    beta: float = 1.0,
    sampled_member_ids: IntArray | None = None,
) -> SelectionResult:
    """Select one candidate from a candidate-by-member posterior value matrix."""

    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if predictions.ndim != 2:
        raise ValueError("predictions must be a candidate-by-member matrix")
    num_candidates, ensemble_size = predictions.shape
    if true_returns.shape != (num_candidates,):
        raise ValueError("true_returns must have one value per candidate")

    means = predictions.mean(axis=1)
    stds = predictions.std(axis=1)

    if method == "random":
        idx = int(rng.integers(0, num_candidates))
        return SelectionResult(index=idx, score=float(means[idx]), method=method)
    if method == "oracle":
        idx = int(np.argmax(true_returns))
        return SelectionResult(index=idx, score=float(true_returns[idx]), method=method, uses_truth=True)
    if method == "mean_bon":
        return _argmax_score(means, method)
    if method == "posterior_sample_bon":
        if sampled_member_ids is None:
            sampled_member_ids = rng.integers(0, ensemble_size, size=num_candidates)
        scores = predictions[np.arange(num_candidates), sampled_member_ids]
        return _argmax_score(scores, method)
    if method == "thompson_bon":
        member_id = int(rng.integers(0, ensemble_size))
        return _argmax_score(predictions[:, member_id], method)
    if method == "ucb_bon":
        return _argmax_score(means + stds, method)
    if method == "pessimistic_bon":
        return _argmax_score(means - stds, method)
    if method == "quantile_bon":
        return _argmax_score(np.quantile(predictions, 0.20, axis=1), method)
    if method == "calibrated_pessimistic_bon":
        return _argmax_score(means - beta * stds, method)

    raise AssertionError("unreachable")
