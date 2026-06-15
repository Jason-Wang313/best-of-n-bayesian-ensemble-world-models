from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from bayesian_ensemble_bon.mujoco_benchmark import MuJoCoBenchmarkConfig, run_mujoco_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v4 Gymnasium MuJoCo benchmark bridge.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--test-problems", type=int, default=6)
    parser.add_argument("--train-transitions", type=int, default=1200)
    parser.add_argument("--ensemble-size", type=int, default=8)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or repo / "results" / "v4_mujoco"
    figure_dir = args.figure_dir or repo / "figures" / "v4_mujoco"
    config = MuJoCoBenchmarkConfig(
        test_problems=args.test_problems,
        train_transitions=args.train_transitions,
        ensemble_size=args.ensemble_size,
    )
    outputs = run_mujoco_benchmark(config, output_dir, figure_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
