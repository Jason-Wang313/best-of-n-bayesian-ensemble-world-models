# Final Audit

## Main Thesis

Posterior-tail selection in Bayesian or ensemble world-model rollouts can
amplify epistemic model-error optimism. The selected candidate may be the one
that looks best under one optimistic posterior member rather than the one that
is robust under the posterior.

## Genuine Novelty

The project does not claim to invent inference-time overoptimization or
pessimism. The novel angle is architecture-specific: a Bayesian or ensemble
world model produces posterior trajectory values, and sampled-posterior maximum
selection changes deployment from posterior aggregation to upper-tail selection
over imagined dynamics.

## Literature Coverage

`docs/related_work_matrix.csv` contains 113 entries. The literature package includes:

- broad landscape map in `docs/literature_map.md`
- 30-paper serious skim set
- 22-paper deep-read threat set
- 10-paper hostile prior-work set in `docs/hostile_prior_work.md`

The strongest threats are PETS, MOPO, MOReL, COMBO, MOBILE, posterior-sampling MBRL, RWM-O/RWM-U, Gao et al. reward overoptimization, Huang et al. inference-time pessimism, and Yu et al. caution.

## Proof Status

The formal material is a diagnostic proposition, not a sweeping theorem. It relies on centered residuals and an order-statistic argument: selecting the maximum posterior-sampled score induces positive selected residual under simple assumptions. The lower-confidence repair claim is a deterministic margin statement. See `docs/proof_claim_audit.md`.

## Strongest Empirical Result

In the full deterministic sweep at `N=128`, sampled-posterior max selection
obtains selected true return `5.567` with regret `10.888` to the in-set
candidate oracle. Posterior-mean selection obtains selected true return
`16.355` with regret `0.100`.

## Strongest Diagnostic Result

At `N=128`, sampled-posterior max selection chooses candidates with OOD mass
`0.304`, posterior standard deviation `8.234`, sample-over-mean gap `13.726`,
and hazard-tail selection rate `0.276`. These values are the
selected-trajectory signature of epistemic tail exploitation.

## Strongest Repair Result

The calibrated lower-confidence selector obtains selected true return `16.223`, regret `0.233`, selected OOD mass `0.168`, and hazard-tail selection rate `0.004` at `N=128`. It uses held-out calibration candidates and ensemble predictions only.

## Expanded Evidence Result

The expanded suite adds 9 posterior stress conditions, 72 high-`N` posterior
stress replicates, 72 learned-bootstrap evaluation problems, paired high-`N`
confidence intervals, and calibration coverage diagnostics. In the nominal
stress condition, sampled-posterior selection is worse than calibrated LCB by
`-12.455` true-return points with 95% CI `[-16.926, -7.984]`. In the
learned-bootstrap shift suite, sampled-posterior selection is worse than
calibrated LCB by `-1.548` true-return points with 95% CI `[-2.412, -0.683]`.

## Biggest Weaknesses

- The benchmark is synthetic and hand-designed.
- The paper does not include real robotics, MuJoCo, or large-scale video world-model experiments.
- The learned suite is a bootstrap ridge return-ensemble stress test rather than a neural video or robotics world model.
- The theory is intentionally modest and does not prove a general result for arbitrary correlated learned dynamics.
- Posterior mean is already strong in the benchmark, so the repair is mainly a correction for posterior-sampled or upper-tail selection rules.

## Paper-Readiness Judgment

Submission-ready as a bounded mechanism paper. The PDF is anonymous, ICLR-template based, compiled, backed by deterministic code/results, and no longer depends on a single synthetic curve: it includes posterior stress sweeps, learned-bootstrap evidence, paired confidence intervals, calibration coverage, limitations, reproducibility notes, and a claim-evidence appendix. The remaining scope boundary is explicit rather than hidden: this is not a real-robot or large-scale world-model benchmark paper.

## Verification

- `python -m pip install -e .`: passed.
- `python -m compileall src experiments tests -q`: passed.
- `pytest -q`: passed with 9 tests.
- `python -m experiments.run_benchmark --preset smoke`: passed and generated `results/smoke` and `figures/smoke`.
- `python -m experiments.run_benchmark --preset full`: passed and generated `results/full` and `figures/full`.
- `python -m experiments.run_v3_evidence`: passed and generated `results/v3` and `figures/v3`.
- `powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1`: passed and compiled the PDF.
- Final LaTeX log has no unresolved citation or reference warnings and no overfull boxes.
- Extracted PDF text contains no project-owned Best-of-N wrapper framing; the only `best-of-n` matches are cited prior-work titles.

## Exact PDF Path

Local final artifact: `paper\final\best of n bayesian ensemble world models-v3.pdf`

Visible Desktop final artifact after guarded publish: `C:\Users\wangz\OneDrive\Desktop\best of n bayesian ensemble world models-v3.pdf`
