# V4 ICLR-Strong-Accept Execution Plan

## Paper

`best of n bayesian ensemble world models`

Current title: **Posterior-Tail Selection and Calibration-Gated Planning in Bayesian Ensemble World Models**

## Current Claim

The v3 paper argues that per-candidate posterior-sampled selection in Bayesian or ensemble world-model planning can over-select epistemic upper tails. A calibrated lower-confidence selector repairs the controlled latent-dynamics benchmark and remains stronger than sampled-posterior selection in a learned bootstrap stress suite.

## V4 Bar

The v4 standard is no longer "submission-ready controlled mechanism paper." It is "ICLR strong-accept hardening." The paper must survive a reviewer who asks why the evidence is synthetic, whether accepted world-model papers use stronger benchmarks, whether the citations are dense enough, whether posterior mean already solves the problem, and whether the method has been tuned against the final test protocol.

## ICLR Rubric Risks

- **Novelty**: Risk that reviewers view this as Best-of-N overoptimization or known pessimism. Repair: foreground posterior aggregation in world-model rollout selection, coherent Thompson-vs-per-candidate posterior sampling, and environment-level benchmark failure.
- **Significance**: Risk that synthetic results feel too small. Repair: add recognized Gymnasium control benchmarks and compare against strong fair planning baselines.
- **Technical correctness**: Risk that the selected-residual proposition is too broad. Repair: keep theory scoped and use empirical selected-tail diagnostics as primary evidence.
- **Experimental rigor**: Risk that v3 lacks standard benchmarks. Repair: add frozen real environment benchmark bridge with Pendulum-v1 and MountainCarContinuous-v0, fixed seeds, bootstrap dynamics ensembles, held-out calibration, fair selectors, teacher baselines, paired CIs, and claim gates.
- **Reproducibility**: Risk that final evidence is mixed with tuning. Repair: write a protocol-freeze JSON with code, seeds, environments, candidate budgets, metrics, baselines, thresholds, and artifact hashes.
- **Clarity**: Risk that appendix carries too much. Repair: move benchmark result and citation-backed context into the main text.

## Accepted-Paper Quality Gap

Nearby accepted/competitive world-model and model-based RL papers use recognized environment suites such as Gym MuJoCo, DMControl, D4RL, Atari 100K, Safety-Gymnasium, Meta-World, or other public benchmarks. V3 has broad controlled sweeps but no recognized environment. For v4, the minimum honest upgrade is a standard Gymnasium control benchmark bridge:

- **Pendulum-v1**: continuous-control swing-up benchmark with continuous actions.
- **MountainCarContinuous-v0**: continuous-control sparse-goal benchmark.

This is not claimed as SOTA on MuJoCo or D4RL. Its role is to remove the "synthetic-only" rejection vector by showing the selected-tail mechanism in recognized real environment dynamics with learned bootstrap world models.

## Real Benchmark Upgrade

### Protocol

- Use Gymnasium classic-control environments already available locally.
- Collect one-step transition data from each environment under fixed training-state/action distributions.
- Fit bootstrap ridge dynamics/reward ensembles.
- Use held-out calibration candidate rollouts to fit the lower-confidence multiplier.
- For each fixed test initial state, sample a shared candidate action-sequence pool and add scripted teacher candidates.
- Score candidates by learned world-model ensembles.
- Execute each selected sequence in the true Gymnasium environment from the exact same initial state to measure final return.
- Compute an in-pool candidate oracle by executing every candidate in the true environment.

### Baselines

- Random candidate.
- Candidate oracle upper bound.
- Posterior mean MPC.
- Per-candidate posterior-sampled MPC.
- Coherent Thompson-member MPC.
- Mean+std UCB.
- Uncalibrated pessimistic selector.
- Quantile selector.
- Calibrated LCB selector.
- Scripted teacher candidate/policy reference.

### Stress Cases

- Environment shift from training-state/action support to broader test starts/actions.
- Candidate budget growth: small, medium, high.
- Action OOD mass and selected ensemble standard deviation.
- Per-environment paired sampled-minus-LCB gaps.
- Teacher-candidate capture vs missed safe candidate.

### RAM-Light Strategy

- CPU-only NumPy/Gymnasium.
- Sequential environment loops.
- Small bootstrap ensembles.
- Write CSVs immediately.
- No GPU, no external model downloads, no large datasets in memory.
- Keep quality by increasing replicates modestly rather than reducing baselines.

## Adversarial Teacher Loop

Use the real benchmark as a development-time teacher:

1. Run a smoke benchmark.
2. If posterior-sampled selection is not worse anywhere, inspect diagnostics: model uncertainty, action OOD, train/test support, candidate generator.
3. If calibrated LCB fails, inspect whether failure is from under-calibration, support mismatch, teacher candidate missed by all selectors, or benchmark not exposing the mechanism.
4. Improve the mechanism only if the failure identifies a missing mechanism. Do not tune final thresholds after freeze.
5. Freeze protocol before final evidence.

## Protocol Freeze

Before final reporting, freeze:

- code version and commit pre-final hash;
- Gymnasium environment names;
- seeds;
- training sample counts;
- calibration candidate counts;
- test problems;
- horizon and candidate budgets;
- ensemble size;
- candidate generator;
- selector list;
- metrics;
- claim gates;
- artifact hashes.

Final evidence after freeze is measurement, not tuning feedback.

## Citation Repair

- Remove hidden citation styling and use visible, professional colored citation links.
- Ensure in-text citations for Gymnasium/OpenAI Gym, PETS, Dreamer/PlaNet, deep ensembles, posterior sampling, pessimistic offline MBRL, uncertainty calibration, and inference-time overoptimization.
- Add citations for standard benchmark usage and clarify that Gymnasium classic control is a benchmark bridge, not a MuJoCo/D4RL SOTA claim.

## Final Acceptance Checklist

- v4 real benchmark artifacts exist under `results/v4` and figures under `figures/v4`.
- Benchmark protocol-freeze JSON exists and hashes final artifacts.
- Manuscript includes the real benchmark result in the main text.
- Related work and benchmark paragraphs have in-text citations.
- No unresolved citations or references.
- No duplicate-template "Best-of-N wrapper" smell in title, abstract, or main contribution.
- Final PDF is at least 25 pages and visually checked.
- Desktop has `best of n bayesian ensemble world models-v4.pdf`.
- `PAPER_SOURCE_MAP.md` points to v4 for this paper.
- Old visible v3 Desktop version is absent.
- Tests, v4 audit, citation audit, and build pass.
- GitHub `main` is pushed and remote SHA verified.
