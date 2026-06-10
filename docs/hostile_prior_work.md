# Hostile Prior-Work Set

These are the papers most likely to be cited by a reviewer arguing that the project is not new.

| Paper | Why it is threatening | Why this repo is different |
| --- | --- | --- |
| PETS | Probabilistic ensembles and trajectory sampling are already central to model-based control. | PETS propagates uncertainty during planning. This repo studies an inference-time maximum over posterior-sampled candidate scores and measures selected-trajectory epistemic exploitation. |
| MOPO | It already penalizes model uncertainty in offline MBRL. | MOPO is a policy-optimization algorithm. The current contribution is a diagnostic and repair for Best-of-N candidate selection at inference time. |
| MOReL | It already converts unknown states into pessimistic outcomes. | MOReL builds a pessimistic MDP for offline RL. This repo isolates posterior-sample optimism under candidate selection without requiring an offline MDP construction. |
| COMBO | It already addresses model overestimation with conservative objectives. | COMBO is value-learning conservatism. The mechanism here is selected posterior upper tails across candidate rollouts. |
| MOBILE | It uses model Bellman inconsistency as an uncertainty penalty. | MOBILE is a training-time offline-RL penalty. This repo asks how to aggregate ensemble rollouts during Best-of-N inference. |
| Posterior Sampling MBRL for Continuous Control | It already formalizes posterior sampling for model-based RL. | PSRL samples a model for an episode. This repo studies many candidate trajectories and the maximum of posterior samples within one decision. |
| RWM-O/RWM-U | It is an uncertainty-aware robotic world model in the exact application family. | RWM-U uses uncertainty for offline policy robustness. It does not present a controlled Best-of-N posterior-sample selection failure. |
| Scaling Laws for Reward Model Overoptimization | It already studies Best-of-N over imperfect learned reward models. | The setting is language reward models. This repo transfers the operator insight to Bayesian ensemble world-model rollouts and adds dynamics-specific diagnostics. |
| Is Best-of-N the Best of Them | It already proves reward hacking and proposes inference-time pessimism. | It is the strongest theory threat. The current project is architecture-specific: posterior rollouts from learned dynamics with ensemble disagreement and trajectory OOD mass. |
| From Curiosity to Caution | It already uses pessimism to repair Best-of-N reward hacking. | Caution trains an error model for language responses. This repo uses posterior ensemble dispersion and calibration for world-model trajectory selection. |

## Hostile Summary

The novelty claim should not be phrased as "pessimism fixes Best-of-N" or "model uncertainty matters." Both are known. The defensible claim is narrower: in Bayesian or ensemble world models, posterior-sampled Best-of-N implements an upper-tail posterior aggregator over complete imagined trajectories, and this can amplify epistemic dynamics errors even when posterior-mean selection is near-oracle in the same candidate set.
