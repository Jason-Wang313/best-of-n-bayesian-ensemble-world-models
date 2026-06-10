from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bayesian_ensemble_bon.selectors import SelectionResult
from bayesian_ensemble_bon.world import TrueRollout


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MetricRow:
    selected_true_return: float
    selected_pred_mean: float
    selected_pred_std: float
    selected_score: float
    oracle_true_return: float
    regret_to_candidate_oracle: float
    optimism_gap: float
    sample_over_mean: float
    selected_ood_mass: float
    selected_hazard_mass: float
    selected_max_position: float
    selected_action_energy: float
    uncertainty_percentile: float
    hazard_tail_selected: float

    def as_dict(self) -> dict[str, float]:
        return {
            "selected_true_return": self.selected_true_return,
            "selected_pred_mean": self.selected_pred_mean,
            "selected_pred_std": self.selected_pred_std,
            "selected_score": self.selected_score,
            "oracle_true_return": self.oracle_true_return,
            "regret_to_candidate_oracle": self.regret_to_candidate_oracle,
            "optimism_gap": self.optimism_gap,
            "sample_over_mean": self.sample_over_mean,
            "selected_ood_mass": self.selected_ood_mass,
            "selected_hazard_mass": self.selected_hazard_mass,
            "selected_max_position": self.selected_max_position,
            "selected_action_energy": self.selected_action_energy,
            "uncertainty_percentile": self.uncertainty_percentile,
            "hazard_tail_selected": self.hazard_tail_selected,
        }


def uncertainty_percentile(stds: FloatArray, selected_index: int) -> float:
    if stds.size <= 1:
        return 0.5
    return float(np.mean(stds <= stds[selected_index]))


def evaluate_selection(
    true_rollout: TrueRollout,
    predictions: FloatArray,
    result: SelectionResult,
) -> MetricRow:
    means = predictions.mean(axis=1)
    stds = predictions.std(axis=1)
    selected = result.index
    oracle_value = float(np.max(true_rollout.returns))
    selected_true = float(true_rollout.returns[selected])
    selected_mean = float(means[selected])
    selected_std = float(stds[selected])
    return MetricRow(
        selected_true_return=selected_true,
        selected_pred_mean=selected_mean,
        selected_pred_std=selected_std,
        selected_score=float(result.score),
        oracle_true_return=oracle_value,
        regret_to_candidate_oracle=float(oracle_value - selected_true),
        optimism_gap=float(selected_mean - selected_true),
        sample_over_mean=float(result.score - selected_mean),
        selected_ood_mass=float(true_rollout.ood_mass[selected]),
        selected_hazard_mass=float(true_rollout.hazard_mass[selected]),
        selected_max_position=float(true_rollout.max_position[selected]),
        selected_action_energy=float(true_rollout.action_energy[selected]),
        uncertainty_percentile=uncertainty_percentile(stds, selected),
        hazard_tail_selected=float(true_rollout.max_position[selected] > 2.35),
    )


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.std(array, ddof=1) / math.sqrt(array.size))
