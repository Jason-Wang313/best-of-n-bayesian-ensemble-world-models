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
- `experiments/run_v3_evidence.py`: bounded stress, learned-bootstrap, paired-effect, and calibration-coverage evidence suite.
- `results/full`: generated CSVs and paper macros.
- `results/v3`: generated stress-suite CSVs, learned-suite CSVs, paired effects, coverage diagnostics, manifest, and claim-evidence audit.
- `figures/full`: generated figures used by the paper.
- `figures/v3`: generated stress, learned-bootstrap, and paired-effect figures.
- `docs`: literature map, 100+ row related-work matrix, hostile-prior set, novelty decision, reviewer attacks, proof audit, reproducibility notes, and final audit.
- `paper`: anonymous ICLR 2026 template source and compiled PDF.

## Quick Start

```powershell
python -m pip install -e .
pytest -q
python -m experiments.run_benchmark --preset smoke
python -m experiments.run_benchmark --preset full
python -m experiments.run_v3_evidence
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

The build script writes the versioned final submission PDF to:

```text
paper\final\best of n bayesian ensemble world models-v3.pdf
```

After the repo has been verified, use the guarded Desktop publish flag:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1 -FinalDesktop
```

That copies the official final PDF to:

```text
C:\Users\wangz\OneDrive\Desktop\best of n bayesian ensemble world models-v3.pdf
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

## Claim Boundary

This is a controlled diagnostic benchmark, not a real-robot or large-scale
world-model result. The paper is submission-ready as a bounded mechanism study,
with the strongest honest claim being mechanistic: posterior-tail aggregation
can exploit epistemic tails, and calibrated lower-confidence aggregation sharply
reduces the failure in the controlled posterior suite while remaining visibly
limited under learned-model shift.
