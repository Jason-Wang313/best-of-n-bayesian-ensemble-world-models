# Posterior-Tail Selection V2 Review Audit

Target manuscript: `Posterior-Tail Selection in Bayesian Ensemble World Models`.

Purpose: remove duplicate-wrapper risk and harden the paper as a distinct
Bayesian/ensemble world-model mechanism paper. The side-by-side risk is high
because the Desktop batch contains multiple papers previously branded around
the same candidate-budget maximum. This revision makes the identity Bayesian:
posterior samples, selected residuals, epistemic OOD mass, calibration residuals,
and lower-confidence aggregation.

## Side-by-Side Distinctness

- LLM paper: score-rank evaluation for language response pools.
- WAM paper: score-tail audits for world-action rollouts.
- Memory paper: memory-impostor rollouts from stale retrieval provenance.
- This paper: posterior-tail selection in Bayesian ensemble rollouts.

The title, abstract, contribution list, theorem role, diagnostics, and repair now
center posterior aggregation. The candidate-budget maximum remains the stressor,
not the project identity.

## 50-Round Attack Log

1. Title looks templated around the old batch theme. Fixed with posterior-tail title.
2. Abstract opens with generic inference scaling. Fixed with uncertainty-consumption failure.
3. Contribution list repeats a shared maximum-selection theorem. Fixed with selected posterior residual.
4. Reviewer asks why this is not reward-model overoptimization. Answer: dynamics posterior, OOD mass, ensemble dispersion.
5. Reviewer asks why pessimism is not known. Answer: repair is not claimed novel; diagnostic route is.
6. Posterior mean already works. Framed as boundary, not weakness hidden in appendix.
7. Synthetic benchmark is too small. Kept limitations explicit and claim scoped.
8. Could be PETS with extra plots. Related work separates uncertainty propagation from selected-tail deployment.
9. Thompson sampling analogy is misleading. Text distinguishes coherent posterior sampling from per-candidate sampled maxima.
10. Hard-coded numbers can drift. Added generated macros for mean regret, posterior std, and sample-over-mean gap.
11. Build might copy stale PDF. Build script now removes stale LaTeX artifacts before compilation.
12. Final artifact path was not visible Desktop. Build now copies v2 PDF to OneDrive Desktop.
13. Plot legend says BoN. Changed public labels to posterior-tail language.
14. README advertises old framing. Rewritten around posterior-tail selection.
15. Package metadata advertises old framing. Rewritten.
16. CLI description advertises old framing. Rewritten.
17. Code comment advertises old framing. Rewritten.
18. Final audit advertises old framing. Rewritten.
19. Reviewer attack docs use old language. Rewritten.
20. Proof audit uses old language. Rewritten.
21. The formal claim may sound too broad. It is explicitly a diagnostic proposition.
22. Independence assumption is unrealistic. Stated as mechanism sketch; benchmark measures realized selected residual.
23. LCB repair is trivial. Presented as a sanity repair, not the contribution.
24. Need oracle context. Paper reports candidate oracle and regret to in-set oracle.
25. Need posterior-mean baseline. Paper foregrounds it and admits it is strongest here.
26. Need upper-confidence baseline. Existing UCB/mean+std remains in figures and CSV.
27. Need Thompson variant. Existing Thompson-member max remains in results.
28. Need calibration evidence. Calibration beta and held-out residual path remain in code/results.
29. Need selected OOD evidence. OOD mass appears in abstract, results, figures, audit.
30. Need tail evidence. Hazard-tail selection appears in abstract, appendix, audit.
31. Need selected residual evidence. Sample-over-mean gap appears in abstract/results.
32. Need uncertainty evidence. Posterior standard deviation appears in abstract/results.
33. Reviewer could say more candidates only help oracle. Paper uses shared candidate prefixes and oracle regret.
34. Reviewer could say candidate generator is biased. Limitation acknowledges hand-designed synthetic benchmark.
35. Reviewer could ask for MuJoCo/robotics. Limitation states absent; related work positions future validation.
36. Reviewer could say no new architecture. Paper claims no new architecture.
37. Reviewer could say no real deployment recipe. Paper claims diagnostic and repair only.
38. Reviewer could say unsafe tail is constructed. Yes; controlled benchmark isolates mechanism.
39. Reviewer could say posterior is miscalibrated. Calibration repair and beta are included.
40. Reviewer could say OOD threshold is arbitrary. OOD is a continuous rollout diagnostic, not a selection criterion.
41. Reviewer could say hazard tail is arbitrary. It is a secondary diagnostic supporting the controlled mechanism.
42. Reviewer could say selected residual is tautological. The comparison to posterior mean and LCB makes it actionable.
43. Reviewer could say regret is to a sampled oracle only. Paper states in-set candidate oracle.
44. Reviewer could say figures hide variance. CSV includes stderr; text keeps claims on large mean gaps.
45. Reviewer could say results depend on N=128. Curves over N remain in figures.
46. Reviewer could say "posterior-tail" is jargon. Abstract defines it immediately.
47. Reviewer could say batch duplication remains in repo path. Public title/abstract/readme now distinguish the project.
48. Reviewer could say citations overclaim. Related work says operator warning is imported, not invented.
49. Reviewer could say submission is not main-track ready. Audit states workshop/internal readiness and main-track gaps.
50. Remaining actionable issue after this pass: run build/tests, inspect PDF visually, and push v2 artifact.
