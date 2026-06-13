from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from bayesian_ensemble_bon.v3_evidence import V3Config, run_v3_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-bounded v3 evidence for posterior-tail selection.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--problems-per-condition", type=int, default=24)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or repo / "results" / "v3"
    figure_dir = args.figure_dir or repo / "figures" / "v3"
    config = V3Config(problems_per_condition=args.problems_per_condition)
    outputs = run_v3_evidence(config, output_dir, figure_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
