# Best-of-N Bayesian Ensemble World Models

This repo is a first-pass ICLR-style research artifact on Best-of-N inference for Bayesian, ensemble, posterior-sampled, and uncertainty-aware world models.

## Thesis

Best-of-N over posterior-sampled world-model values can act as an upper-tail optimizer over epistemic model error. It may select trajectories that look high-value under one plausible ensemble member while being weak under the posterior and poor under the true latent dynamics. Posterior-mean or calibrated lower-confidence selection avoids this selected-tail failure in the controlled benchmark.

## What Is Included

- `src/bayesian_ensemble_bon`: synthetic latent dynamics, posterior ensemble rollouts, selectors, metrics, experiment runner, and plotting.
- `experiments/run_benchmark.py`: deterministic smoke and full sweeps.
- `results/full`: generated CSVs and paper macros.
- `figures/full`: generated figures used by the paper.
- `docs`: literature map, 100+ row related-work matrix, hostile-prior set, novelty decision, reviewer attacks, proof audit, reproducibility notes, and final audit.
- `paper`: anonymous ICLR 2026 template source and compiled PDF.

## Quick Start

```powershell
python -m pip install -e .
pytest -q
python -m experiments.run_benchmark --preset smoke
python -m experiments.run_benchmark --preset full
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

The build script copies the final submission PDF to:

```text
C:\Users\wangz\Downloads\iclr_submission_bayesian_ensemble_world_models.pdf
```

## Main Full-Sweep Result

At `N=128`, posterior-sampled Best-of-N has selected true return `5.567` and regret `10.888` to the in-set candidate oracle. Calibrated pessimistic selection has selected true return `16.223` and regret `0.233`. Posterior-mean selection is also strong with selected true return `16.355` and regret `0.100`.

## Claim Boundary

This is a controlled diagnostic benchmark, not a real-robot or large-scale world-model result. The paper is submission-shaped and reproducible, but its strongest honest claim is mechanistic: the posterior-sampled Best-of-N aggregation rule can exploit epistemic tails.
