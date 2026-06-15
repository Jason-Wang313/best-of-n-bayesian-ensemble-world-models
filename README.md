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

- `src/bayesian_ensemble_bon`: synthetic latent dynamics, posterior ensemble rollouts, Gymnasium/MuJoCo benchmark bridges, selectors, metrics, experiment runners, and plotting.
- `experiments/run_benchmark.py`: deterministic smoke and full sweeps.
- `experiments/run_v3_evidence.py`: bounded stress, learned-bootstrap, paired-effect, and calibration-coverage evidence suite.
- `experiments/run_v4_gym_benchmark.py`: Gymnasium classic-control bridge with frozen protocol artifacts.
- `experiments/run_v4_mujoco_benchmark.py`: Gymnasium MuJoCo bridge with frozen protocol artifacts.
- `results/full`: generated CSVs and paper macros.
- `results/v3`: generated stress-suite CSVs, learned-suite CSVs, paired effects, coverage diagnostics, manifest, and claim-evidence audit.
- `results/v4`: generated Gymnasium classic-control CSVs, paired effects, calibration, protocol freeze, and paper macros.
- `results/v4_mujoco`: generated Gymnasium MuJoCo CSVs, paired effects, calibration, protocol freeze, and paper macros.
- `figures/full`: generated figures used by the paper.
- `figures/v3`: generated stress, learned-bootstrap, and paired-effect figures.
- `figures/v4` and `figures/v4_mujoco`: generated recognized-environment bridge figures.
- `docs`: literature map, 100+ row related-work matrix, hostile-prior set, novelty decision, reviewer attacks, proof audit, reproducibility notes, and final audit.
- `paper`: anonymous ICLR 2026 template source and compiled PDF.

## Quick Start

```powershell
python -m pip install -e .
pytest -q
python -m experiments.run_benchmark --preset smoke
python -m experiments.run_benchmark --preset full
python -m experiments.run_v3_evidence
python -m experiments.run_v4_gym_benchmark
python -m experiments.run_v4_mujoco_benchmark
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

The build script writes the versioned final submission PDF to:

```text
paper\final\best of n bayesian ensemble world models-v4.pdf
```

After the repo has been verified, use the guarded Desktop publish flag:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1 -FinalDesktop
```

That copies the official final PDF to:

```text
C:\Users\wangz\OneDrive\Desktop\best of n bayesian ensemble world models-v4.pdf
```

## Main Full-Sweep Result

At `N=128`, sampled-posterior max selection has selected true return `5.567`
and regret `10.888` to the in-set candidate oracle. Calibrated pessimistic
selection has selected true return `16.223` and regret `0.233`. Posterior-mean
selection is also strong with selected true return `16.355` and regret `0.100`.

The expanded evidence suite adds 9 posterior stress conditions, 72 learned
bootstrap evaluation problems, paired high-`N` confidence intervals, and
calibration coverage diagnostics. In the nominal stress condition,
sampled-posterior selection is worse than calibrated LCB by `-12.455` true-return
points with 95% CI `[-16.926, -7.984]`. In the learned-bootstrap shift suite,
sampled-posterior selection is worse than calibrated LCB by `-1.548` true-return
points with 95% CI `[-2.412, -0.683]`.

The v4 evidence adds a standard Gymnasium classic-control bridge and a
Gymnasium MuJoCo bridge. On the MuJoCo bridge, the calibration-gated selector
has mean true return `1.460` and regret `3.799`, compared with posterior
sampling return `0.981` and regret `4.278`. These are recognized-environment
diagnostics, not MuJoCo or D4RL SOTA claims.

## Claim Boundary

This is a controlled diagnostic and recognized-simulator benchmark, not a
real-robot or large-scale world-model result. The strongest honest claim is
mechanistic: posterior-tail aggregation can exploit epistemic tails, and
calibrated lower-confidence or validation-gated aggregation sharply reduces the
failure in controlled stress tests while remaining visibly scoped under learned
model shift and compact MuJoCo bridges.
