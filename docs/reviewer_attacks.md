# Reviewer Attacks and Responses

## Attack 1: This is just reward-model overoptimization.

Response: The operator-level analogy is acknowledged. The new part is the architecture-specific route: a posterior ensemble world model scores full imagined trajectories, and Best-of-N over posterior samples selects the most optimistic model-member rollout. The diagnostics report selected OOD mass, hazard mass, posterior standard deviation, and sample-over-mean gap.

## Attack 2: Pessimism in model-based offline RL is known.

Response: Yes. The paper does not claim to invent pessimism. It uses calibrated lower-confidence selection as a minimal repair and positions MOPO/MOReL/COMBO/RAMBO/MOBILE as direct ancestors.

## Attack 3: Synthetic experiments are not enough for ICLR.

Response: Correct as a final empirical package. The honest claim is a controlled diagnostic benchmark and runnable first-pass artifact. The paper should be judged as a mechanism paper needing real-world validation, not as a robotics SOTA submission.

## Attack 4: Posterior-mean selection already fixes the problem.

Response: That is part of the point. The failure is not "ensembles are bad." It is that posterior-sampled Best-of-N uses the wrong posterior aggregator for deployment when model error is epistemic and tail-heavy.

## Attack 5: The posterior might be miscalibrated by construction.

Response: The benchmark includes calibration on held-out candidate rollouts and reports the fitted residual-to-std multiplier. The repair uses only ensemble predictions and calibration data, not the true returns of test candidates.

## Attack 6: A simple lower confidence bound is too obvious.

Response: The method is intentionally simple. The novelty is the diagnostic that says when a lower-confidence selector is necessary for posterior-sampled world-model Best-of-N. A stronger paper would replace the scalar LCB with conformal trajectory-level coverage.
