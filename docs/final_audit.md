# Final V4 Audit

## Main Thesis

Posterior-tail selection in Bayesian or ensemble world-model rollouts can
amplify epistemic model-error optimism. The selected candidate may be the one
that looks best under one optimistic posterior member rather than the one that
is robust under the posterior or strong under the true environment.

## Genuine Novelty

The project does not claim to invent inference-time overoptimization, ensemble
uncertainty, pessimism, or model-based RL. The novel angle is
architecture-specific: a Bayesian or ensemble world model produces posterior
trajectory values, and sampled-posterior maximum selection changes deployment
from posterior aggregation to upper-tail selection over imagined dynamics.

## Duplicate-Risk Audit

The v4 manuscript is no longer a generic "Best-of-N theorem wrapper." It uses
the paper-specific title "Posterior-Tail Selection and Calibration-Gated
Planning in Bayesian Ensemble World Models," frames the contribution around
posterior aggregation under ensemble world models, and includes a domain-specific
mechanism, benchmark bridge, related work, limitations, and claim gates. The
only remaining best-of-n language is either ordinary method terminology or
cited prior work.

## Literature Coverage

`docs/related_work_matrix.csv` contains 113 entries. The literature package
includes:

- broad landscape map in `docs/literature_map.md`
- 30-paper serious skim set
- 22-paper deep-read threat set
- 10-paper hostile prior-work set in `docs/hostile_prior_work.md`

The strongest threats are PETS, MOPO, MOReL, COMBO, MOBILE, RAMBO, CQL,
posterior-sampling MBRL, RWM-O/RWM-U, reward overoptimization, inference-time
pessimism, and cautionary analyses of large candidate sets. The v4 paper cites
these threats in text and scopes the claim away from "pessimism is new."

## Proof Status

The formal material is a diagnostic proposition, not a sweeping theorem. It
uses a centered-residual/order-statistic argument to show how selecting the
maximum posterior-sampled score induces positive selected residual under simple
conditions. The lower-confidence repair is a deterministic margin statement.
See `docs/proof_claim_audit.md`.

## Evidence Stack

The final paper combines four evidence layers:

- controlled latent dynamics full sweep under `results/full`
- v3 posterior stress grid, learned-bootstrap shift suite, paired confidence
  intervals, calibration coverage, and claim-evidence audit under `results/v3`
- Gymnasium classic-control benchmark bridge under `results/v4`
- Gymnasium MuJoCo continuous-control benchmark bridge under
  `results/v4_mujoco`

The standard-environment bridges use true environment returns for final
measurement, shared held-out starts, frozen seeds, held-out calibration/gating,
strong fair selector baselines, in-pool oracles, and separately reported CEM or
teacher baselines. They are intentionally claim-gated as recognized-environment
diagnostics, not MuJoCo/D4RL SOTA claims.

## Strongest Empirical Results

In the full deterministic sweep at `N=128`, sampled-posterior max selection
obtains selected true return `5.567` with regret `10.888` to the in-set
candidate oracle. Posterior-mean selection obtains selected true return
`16.355` with regret `0.100`.

At high `N`, sampled-posterior selection is worse than calibrated LCB by
`-12.455` true-return points with 95% CI `[-16.926, -7.984]` in the nominal
stress condition. In the learned-bootstrap shift suite, sampled-posterior
selection is worse than calibrated LCB by `-1.548` true-return points with 95%
CI `[-2.412, -0.683]`.

On the Gymnasium MuJoCo bridge, the calibration-gated selector obtains mean true
return `1.460` and regret `3.799`, compared with posterior-sampled selection at
return `0.981` and regret `4.278`.

## Attack Summary

- **Synthetic-only rejection:** repaired by adding recognized Gymnasium
  classic-control and MuJoCo bridge evidence.
- **Known pessimism rejection:** scoped as a posterior-tail diagnostic and
  calibration/gating repair, not as a new pessimistic RL algorithm.
- **Posterior mean already strong:** acknowledged; the target failure is
  posterior-sampled or upper-confidence tail selection, and posterior mean is a
  strong baseline.
- **Benchmark leakage:** final protocol files freeze seeds, tasks, methods,
  candidate budgets, calibration splits, metrics, baselines, and hashes before
  reporting.
- **Weak baselines:** selector baselines include random, oracle, posterior
  mean, posterior sample, coherent Thompson, UCB, pessimistic LCB, quantile,
  calibrated LCB, validation-gated selection, scripted teacher, and CEM where
  applicable.
- **Overclaiming:** explicit claim gates rule out real-robot validation,
  large-scale video world-model dominance, D4RL SOTA, and MuJoCo SOTA.

## Biggest Remaining Scope Boundaries

- The standard-environment bridges are compact and CPU-bounded.
- The learned suites use ridge/bootstrap dynamics or reward ensembles rather
  than large neural video or robotics world models.
- The MuJoCo result is a mechanism diagnostic with finite candidate pools, not
  a state-of-the-art leaderboard run.
- The theory assumes simplified centered residual structure and does not prove
  a general result for arbitrary correlated learned dynamics.

These are now stated as limitations and claim gates rather than hidden risks.

## Paper-Readiness Judgment

Submission-ready as a bounded ICLR-style mechanism paper. The PDF is anonymous,
compiled in the ICLR 2026 template, 25 pages including appendix and references,
and backed by deterministic code/results. The v4 paper no longer depends on a
single synthetic curve: it includes posterior stress sweeps, learned-bootstrap
evidence, paired confidence intervals, calibration coverage, recognized
standard-environment bridges, negative controls, fair baselines, limitations,
reproducibility notes, and claim-evidence appendices.

## Verification On 2026-06-16

- `python -m compileall src experiments tests -q`: passed.
- `python -m pytest -q`: passed with 11 tests.
- `powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1 -FinalDesktop`: passed; `latexmk`/Perl fallback path used direct `pdflatex`/`bibtex` passes.
- Final `paper/main.log` scan found no unresolved citation warnings, unresolved reference warnings, rerun warnings, or overfull boxes.
- Repo PDF and visible Desktop PDF have matching SHA256:
  `EA9B3490E24FD49D663CE3AB0FE2EF12EFDDD42E6AB3DB09BF6F423D33E2754D`.
- `pdfinfo` reports 25 letter-size pages.
- Visual QA rendered all 25 pages with Poppler and inspected pages 1, 6, 7, 21,
  and 25 for title layout, main figures/tables, related work, dense appendix,
  and final checklist.

## Exact PDF Path

Local final artifact:
`paper\final\best of n bayesian ensemble world models-v4.pdf`

Visible Desktop final artifact:
`C:\Users\wangz\OneDrive\Desktop\best of n bayesian ensemble world models-v4.pdf`
