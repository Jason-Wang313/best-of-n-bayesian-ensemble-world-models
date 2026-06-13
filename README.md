# Posterior-Tail Selection

This repo is an anonymous ICLR-style research artifact on posterior-tail
selection in Bayesian, ensemble, posterior-sampled, and uncertainty-aware world
models.

## Thesis

A many-candidate maximum over posterior-sampled world-model values can act as an
upper-tail optimizer over epistemic model error. It may select trajectories that
look high-value under one plausible ensemble member while being weak under the
posterior and poor under the true latent dynamics. Posterior-mean or calibrated
lower-confidence selection avoids this selected-tail failure in the controlled
benchmark.

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

The build script copies the versioned final submission PDF to:

```text
C:\Users\wangz\OneDrive\Desktop\best of n bayesian ensemble world models-v2.pdf
```

## Main Full-Sweep Result

At `N=128`, sampled-posterior max selection has selected true return `5.567`
and regret `10.888` to the in-set candidate oracle. Calibrated pessimistic
selection has selected true return `16.223` and regret `0.233`. Posterior-mean
selection is also strong with selected true return `16.355` and regret `0.100`.

## Claim Boundary

This is a controlled diagnostic benchmark, not a real-robot or large-scale
world-model result. The paper is submission-shaped and reproducible, but its
strongest honest claim is mechanistic: posterior-tail aggregation can exploit
epistemic tails.
