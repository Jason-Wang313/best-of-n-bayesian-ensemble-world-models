# Novelty Decision

## Chosen Contribution Type

Diagnostic plus repair.

The project is strongest as a mechanism-first diagnostic benchmark. A full theorem about arbitrary Bayesian world models would be too broad for the evidence here, and a robotics-scale empirical claim would be dishonest. The strongest contribution is:

> Posterior-tail selection over Bayesian or ensemble world-model values can act as an epistemic upper-tail optimizer. It selects trajectories that look good under at least one plausible model member, not trajectories that are robust under the posterior. A calibrated lower-confidence selector restores most of the in-set oracle value in the controlled posterior suite and remains a useful, explicitly limited baseline under learned-model shift.

## Why Not a Pure Theory Paper

The formal insight is real but simple: if candidates have zero-mean posterior score noise, maximizing a sampled score selects positive noise, and the expected selected sample residual grows with the candidate budget. The dynamics-specific part is empirical and diagnostic: the selected positive noise correlates with OOD trajectory mass and high ensemble dispersion.

## Why Not a Pure Empirical Paper

The repository contains a synthetic benchmark rather than MuJoCo, real-robot, or large-scale video world-model experiments. The empirical claim is therefore controlled and mechanistic, not benchmark dominance.

## Final Angle

Mechanism -> diagnostic -> empirical validation -> calibration-aware repair.

The ICLR paper should emphasize that the benchmark is deliberately compact and falsifiable: posterior-mean selection remains strong, posterior-sampled tail selection fails at high candidate budgets, and calibrated pessimism removes most of the selected-tail failure in the controlled posterior suite.
