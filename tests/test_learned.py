from __future__ import annotations

import numpy as np

from bayesian_ensemble_bon.learned import action_features, fit_bootstrap_return_ensemble
from bayesian_ensemble_bon.world import rollout_true, sample_action_sequences


def test_action_features_are_deterministic_and_finite() -> None:
    rng = np.random.default_rng(123)
    actions = sample_action_sequences(rng, 12, 10)
    features_a = action_features(actions)
    features_b = action_features(actions)
    assert features_a.shape[0] == 12
    assert features_a.shape[1] >= 10
    np.testing.assert_allclose(features_a, features_b)
    assert np.all(np.isfinite(features_a))


def test_bootstrap_return_ensemble_predicts_member_matrix() -> None:
    rng = np.random.default_rng(321)
    train_actions = sample_action_sequences(rng, 80, 10)
    returns = rollout_true(train_actions).returns
    model = fit_bootstrap_return_ensemble(train_actions, returns, seed=7, ensemble_size=5)
    test_actions = sample_action_sequences(rng, 9, 10)
    predictions = model.predict(test_actions)
    assert predictions.shape == (9, 5)
    assert np.all(np.isfinite(predictions))
