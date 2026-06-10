from __future__ import annotations

import numpy as np

from bayesian_ensemble_bon.world import WorldConfig, make_posterior, rollout_ensemble, rollout_true, sample_action_sequences


def test_rollouts_have_expected_shapes() -> None:
    config = WorldConfig(horizon=10, ensemble_size=7)
    rng = np.random.default_rng(0)
    actions = sample_action_sequences(rng, num_candidates=11, horizon=config.horizon)
    posterior = make_posterior(1, config)
    true_rollout = rollout_true(actions)
    predictions = rollout_ensemble(actions, posterior)
    assert actions.shape == (11, 10)
    assert true_rollout.returns.shape == (11,)
    assert predictions.shape == (11, 7)
    assert np.all(np.isfinite(predictions))


def test_sampling_is_deterministic_for_seed() -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    actions_a = sample_action_sequences(rng_a, 8, 12)
    actions_b = sample_action_sequences(rng_b, 8, 12)
    np.testing.assert_allclose(actions_a, actions_b)
