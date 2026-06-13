# V3 Claim-Evidence Audit

This audit is generated from `results/v3` and scopes the v3 claims to CPU-light controlled evidence.

| Claim | Evidence | Boundary |
|---|---|---|
| Posterior-sampled high-N selection remains fragile under posterior stress. | Nominal stress regret at N=128: sampled posterior 12.896 vs calibrated LCB 0.441. | Controlled latent system; not a real robot benchmark. |
| The effect is not only hand-coded posterior members. | Learned bootstrap return ensemble regret at N=128: sampled posterior 14.557 vs calibrated LCB 13.009. | Lightweight learned value ensemble, not a large neural world model. |
| The high-N harm is paired, not just aggregate plotting. | Nominal paired true-return gap sampled minus LCB: -12.455 with 95% CI [-16.926, -7.984]. | Paired over synthetic candidate problems. |
| Calibration evidence is explicitly artifact-backed. | `calibration_coverage.csv` reports optimistic-miss rates and beta values for each suite/condition. | Coverage is diagnostic, not a formal conformal guarantee. |

The v3 evidence strengthens scope and rigor without raising the claim to hardware, MuJoCo, or large-scale video world-model validation.
