from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_ensemble_bon.experiment import ExperimentConfig, run_experiment
from bayesian_ensemble_bon.plotting import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian ensemble posterior-tail benchmark.")
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or repo / "results" / args.preset
    figure_dir = args.figure_dir or repo / "figures" / args.preset
    config = ExperimentConfig.for_preset(args.preset)
    outputs = run_experiment(config, output_dir)
    figures = make_figures(outputs["summary"], figure_dir)
    print(f"results: {outputs['summary']}")
    for name, path in figures.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
