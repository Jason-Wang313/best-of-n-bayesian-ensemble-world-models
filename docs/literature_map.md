# Literature Map

This is an autonomous first-pass literature package for the repo. The sweep is broad and honest rather than pretending to be a finished human 100-paper review. I treated the 100-entry CSV as the landscape triage, then used the highest-threat clusters for the serious skim and hostile-prior comparison.

## Landscape Clusters

- World models and model-based RL: PlaNet, Dreamer, DreamerV2/V3, SimPLe, MuZero, TD-MPC, PETS, MBPO, MB-MPO, VAML, PILCO.
- Ensemble and Bayesian uncertainty: PETS, deep ensembles, MC dropout, bootstrapped DQN, randomized prior functions, PSRL, Bayesian world models, uncertainty-aware robotic world models.
- Offline and pessimistic MBRL: MOPO, MOReL, COMBO, RAMBO, MOBILE, CQL, IQL, SPIBB, BCQ, BEAR.
- Best-of-N and reward-model overoptimization: Gao et al. 2023, Jinnai et al. 2024, Huang et al. 2025, Ichihara et al. 2025, Yu et al. 2026, Hsu et al. 2026.
- Calibration and uncertainty auditing: Guo et al. 2017, Ovadia et al. 2019, Kuleshov et al. 2018, conformal prediction, calibrated regression.
- Robotics and foundation world models: SayCan, RT-1, RT-2, Octo, Open X-Embodiment, RWM-O/RWM-U, Genie, IRIS.

## 30-Paper Serious Skim Set

1. PETS
2. MBPO
3. PlaNet
4. Dreamer
5. DreamerV3
6. TD-MPC
7. TD-MPC2
8. MuZero
9. MOPO
10. MOReL
11. COMBO
12. RAMBO
13. MOBILE
14. Deep Ensembles
15. MC Dropout
16. Bootstrapped DQN
17. Randomized Prior Functions
18. PSRL for continuous control
19. Bayesian world-model safe exploration
20. RWM-O/RWM-U
21. Uncertainty-guided latent model ensembles
22. Gao et al. reward overoptimization
23. Regularized Best-of-N
24. Is Best-of-N the Best of Them
25. Best-of-N through smoothing
26. From Curiosity to Caution
27. On calibration of modern neural networks
28. Can You Trust Your Model's Uncertainty
29. Accurate Uncertainties for Deep Learning
30. Trajectory Transformer

## 22-Paper Deep-Read Threat Set

The deep-read set focuses on papers that could plausibly subsume the final contribution. The result was that no single paper covers all three ingredients: posterior-sampled or ensemble world-model rollouts, inference-time posterior-tail selection pressure, and a selected-trajectory diagnostic showing amplification of epistemic model error.

- PETS and model-ensemble TRPO establish ensemble planning but treat sampling as a way to propagate uncertainty rather than as a maximum-selection failure mode.
- MOPO, MOReL, COMBO, RAMBO, and MOBILE establish pessimism against model error, mainly as offline policy optimization or value regularization.
- PSRL papers establish posterior sampling but use episode-level probability matching rather than a per-candidate posterior-sampled maximum over many trajectory options.
- RWM-O/RWM-U and Bayesian safe-exploration papers are the closest world-model systems with epistemic uncertainty, but the deployment question is uncertainty-penalized policy optimization or safety.
- Gao et al., Huang et al., Jinnai et al., Ichihara et al., Yu et al., and Hsu et al. are the closest Best-of-N theory and repair papers, but they study language reward models or verifier distributions rather than learned dynamics posterior rollouts.

## Resulting Gap

The most defensible gap is a diagnostic one: posterior-tail selection over world-model values is not just "more planning." It changes the posterior aggregation operator from expectation or lower confidence to a selected upper tail. In uncertain regions, that upper-tail operator can select trajectories whose high score is caused by one optimistic posterior member rather than by posterior-robust value.
