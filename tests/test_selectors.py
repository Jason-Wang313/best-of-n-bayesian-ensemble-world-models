from __future__ import annotations

import numpy as np

from bayesian_ensemble_bon.selectors import fit_calibration_beta, select_candidate


def test_oracle_is_only_truth_using_selector() -> None:
    predictions = np.array([[0.0, 0.1], [1.0, 1.2], [0.3, 0.4]])
    true_returns = np.array([3.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    oracle = select_candidate(predictions, true_returns, "oracle", rng)
    mean = select_candidate(predictions, true_returns, "mean_bon", rng)
    assert oracle.index == 0
    assert oracle.uses_truth
    assert mean.index == 1
    assert not mean.uses_truth


def test_calibrated_pessimism_can_avoid_uncertain_optimism() -> None:
    predictions = np.array(
        [
            [2.0, 2.1, 1.9, 2.0],
            [2.6, 5.8, -0.5, 2.4],
        ]
    )
    true_returns = np.array([2.0, 0.4])
    rng = np.random.default_rng(1)
    mean = select_candidate(predictions, true_returns, "mean_bon", rng)
    lcb = select_candidate(predictions, true_returns, "calibrated_pessimistic_bon", rng, beta=1.2)
    assert mean.index == 1
    assert lcb.index == 0


def test_calibration_beta_is_positive_and_clipped() -> None:
    predictions = np.array([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]])
    true_returns = np.array([0.1, -5.0, 2.1])
    beta = fit_calibration_beta(predictions, true_returns)
    assert 0.35 <= beta <= 4.0
