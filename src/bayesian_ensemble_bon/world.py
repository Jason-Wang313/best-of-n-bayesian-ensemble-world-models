from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WorldConfig:
    """Configuration for the controlled latent world-model benchmark."""

    horizon: int = 18
    ensemble_size: int = 32
    posterior_spread: float = 1.0
    blindspot: float = 1.0
    action_scale: float = 1.0


@dataclass(frozen=True)
class PosteriorMember:
    gain: float
    damping: float
    hazard_center: float
    hazard_strength: float
    goal_shift: float
    value_bias: float
    terminal_bias: float


@dataclass(frozen=True)
class TrueRollout:
    returns: FloatArray
    max_position: FloatArray
    ood_mass: FloatArray
    hazard_mass: FloatArray
    action_energy: FloatArray


def sigmoid(x: FloatArray | float) -> FloatArray | float:
    return 1.0 / (1.0 + np.exp(-x))


def make_posterior(seed: int, config: WorldConfig) -> list[PosteriorMember]:
    """Draw a posterior ensemble around the true latent dynamics.

    The posterior is intentionally plausible but imperfect: most models are
    competent in the in-distribution region, while the high-position hazard is
    weakly identified. This is the regime in which a many-candidate maximum can
    select the most optimistic posterior member's hallucinated trajectory.
    """

    rng = np.random.default_rng(seed)
    spread = config.posterior_spread
    blindspot = config.blindspot
    members: list[PosteriorMember] = []
    for _ in range(config.ensemble_size):
        members.append(
            PosteriorMember(
                gain=float(rng.normal(0.52, 0.045 * spread)),
                damping=float(rng.normal(0.74, 0.035 * spread)),
                hazard_center=float(rng.normal(2.34, 0.16 * spread)),
                hazard_strength=float(np.exp(rng.normal(-0.10, 0.55 * spread))),
                goal_shift=float(rng.normal(0.0, 0.11 * spread)),
                value_bias=float(rng.normal(0.0, 0.55 * blindspot)),
                terminal_bias=float(rng.normal(0.0, 0.85 * blindspot)),
            )
        )
    return members


def sample_action_sequences(
    rng: np.random.Generator,
    num_candidates: int,
    horizon: int,
    action_scale: float = 1.0,
) -> FloatArray:
    """Sample smooth action sequences with rare high-excursion probes."""

    noise = rng.normal(0.0, 0.72 * action_scale, size=(num_candidates, horizon))
    actions = np.zeros_like(noise)
    for t in range(horizon):
        previous = actions[:, t - 1] if t else 0.0
        actions[:, t] = 0.62 * previous + noise[:, t]

    time = np.arange(horizon, dtype=float)
    has_probe = rng.random(num_candidates) < 0.34
    centers = rng.uniform(0.25 * horizon, 0.75 * horizon, size=num_candidates)
    widths = rng.uniform(1.7, 4.8, size=num_candidates)
    amplitudes = rng.lognormal(mean=0.52, sigma=0.42, size=num_candidates)
    directions = rng.choice(np.array([1.0, -0.55]), size=num_candidates, p=[0.78, 0.22])
    pulses = amplitudes[:, None] * directions[:, None] * np.exp(
        -0.5 * ((time[None, :] - centers[:, None]) / widths[:, None]) ** 2
    )
    actions = actions + has_probe[:, None] * pulses
    return np.clip(actions, -3.0, 3.0).astype(np.float64)


def rollout_true(actions: FloatArray) -> TrueRollout:
    """Roll out the true two-dimensional latent control system."""

    actions = np.asarray(actions, dtype=np.float64)
    num_candidates, horizon = actions.shape
    position = np.zeros(num_candidates, dtype=np.float64)
    velocity = np.zeros(num_candidates, dtype=np.float64)
    returns = np.zeros(num_candidates, dtype=np.float64)
    max_position = np.zeros(num_candidates, dtype=np.float64)
    ood_mass = np.zeros(num_candidates, dtype=np.float64)
    hazard_mass = np.zeros(num_candidates, dtype=np.float64)
    action_energy = np.mean(actions**2, axis=1)

    for t in range(horizon):
        action = actions[:, t]
        hazard_gate = sigmoid(5.0 * (position - 2.12))
        velocity = 0.75 * velocity + 0.52 * np.tanh(action) - 0.10 * position
        velocity = velocity - 0.16 * hazard_gate * velocity
        position = position + 0.38 * velocity

        goal_reward = 1.55 * np.exp(-0.5 * ((position - 1.55) / 0.43) ** 2)
        hazard_penalty = 0.32 * np.maximum(position - 2.05, 0.0) ** 2
        hazard_penalty += 1.85 * sigmoid(5.8 * (position - 2.58))
        shaping = -0.020 * position**2 - 0.060 * action**2
        returns += goal_reward - hazard_penalty + shaping

        max_position = np.maximum(max_position, position)
        ood_mass += sigmoid(4.5 * (position - 1.82))
        hazard_mass += sigmoid(5.5 * (position - 2.35))

    returns += -0.15 * np.maximum(max_position - 2.25, 0.0) ** 2
    return TrueRollout(
        returns=returns,
        max_position=max_position,
        ood_mass=ood_mass / horizon,
        hazard_mass=hazard_mass / horizon,
        action_energy=action_energy,
    )


def rollout_ensemble(actions: FloatArray, members: list[PosteriorMember]) -> FloatArray:
    """Return a candidate-by-member matrix of posterior predicted returns."""

    actions = np.asarray(actions, dtype=np.float64)
    num_candidates, horizon = actions.shape
    gain = np.asarray([member.gain for member in members], dtype=np.float64)[None, :]
    damping = np.asarray([member.damping for member in members], dtype=np.float64)[None, :]
    hazard_center = np.asarray([member.hazard_center for member in members], dtype=np.float64)[None, :]
    hazard_strength = np.asarray([member.hazard_strength for member in members], dtype=np.float64)[None, :]
    goal_shift = np.asarray([member.goal_shift for member in members], dtype=np.float64)[None, :]
    value_bias = np.asarray([member.value_bias for member in members], dtype=np.float64)[None, :]
    terminal_bias = np.asarray([member.terminal_bias for member in members], dtype=np.float64)[None, :]

    position = np.zeros((num_candidates, len(members)), dtype=np.float64)
    velocity = np.zeros_like(position)
    returns = np.zeros_like(position)
    ood_mass = np.zeros_like(position)
    max_position = np.zeros_like(position)

    for t in range(horizon):
        action = actions[:, t][:, None]
        hazard_gate = sigmoid(5.0 * (position - hazard_center))
        velocity = damping * velocity + gain * np.tanh(action) - 0.10 * position
        velocity = velocity - 0.13 * hazard_strength * hazard_gate * velocity
        position = position + 0.38 * velocity

        ood_gate = sigmoid(4.5 * (position - 1.78))
        goal_reward = 1.55 * np.exp(-0.5 * ((position - (1.55 + goal_shift)) / 0.45) ** 2)
        hazard_penalty = 0.30 * hazard_strength * np.maximum(position - 2.05, 0.0) ** 2
        hazard_penalty += 1.70 * hazard_strength * sigmoid(5.2 * (position - hazard_center))
        hallucinated_tail = value_bias * ood_gate * np.maximum(position - 1.72, 0.0) ** 2
        shaping = -0.018 * position**2 - 0.055 * action**2
        returns += goal_reward - hazard_penalty + hallucinated_tail + shaping

        ood_mass += ood_gate
        max_position = np.maximum(max_position, position)

    returns += terminal_bias * sigmoid(4.0 * (max_position - 2.0)) * np.maximum(max_position - 1.9, 0.0)
    returns += 0.08 * (ood_mass / horizon - 0.3)
    return returns.astype(np.float64)
