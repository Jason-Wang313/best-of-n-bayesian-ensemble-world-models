from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PLOT_METHODS = [
    ("oracle", "Candidate oracle"),
    ("mean_bon", "Posterior mean max"),
    ("posterior_sample_bon", "Sampled posterior max"),
    ("ucb_bon", "Mean + std max"),
    ("calibrated_pessimistic_bon", "Calibrated LCB"),
]


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict[str, str]], metric: str) -> dict[str, tuple[list[int], list[float]]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append((int(row["n_candidates"]), float(row[f"{metric}_mean"])))
    return {
        method: (
            [point[0] for point in sorted(points)],
            [point[1] for point in sorted(points)],
        )
        for method, points in grouped.items()
    }


def _plot_metric(
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    path: Path,
    title: str | None = None,
) -> None:
    data = _series(rows, metric)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for method, label in PLOT_METHODS:
        if method not in data:
            continue
        xs, ys = data[method]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("candidate budget N")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(summary_csv: Path, figure_dir: Path) -> dict[str, Path]:
    rows = _read_summary(summary_csv)
    outputs = {
        "true_return": figure_dir / "true_return_vs_n.png",
        "optimism": figure_dir / "optimism_gap_vs_n.png",
        "ood": figure_dir / "uncertainty_exploitation_vs_n.png",
        "regret": figure_dir / "regret_vs_n.png",
    }
    _plot_metric(rows, "selected_true_return", "selected true return", outputs["true_return"])
    _plot_metric(rows, "optimism_gap", "posterior mean minus truth", outputs["optimism"])
    _plot_metric(rows, "selected_ood_mass", "selected OOD mass", outputs["ood"])
    _plot_metric(rows, "regret_to_candidate_oracle", "regret to in-set oracle", outputs["regret"])
    return outputs
