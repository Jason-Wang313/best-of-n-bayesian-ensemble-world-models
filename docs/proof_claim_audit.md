# Proof and Claim Audit

## Formal Claim A: Upper-tail selection bias

Claim: For candidates with posterior sample scores equal to posterior means plus independent centered errors, selecting the largest sampled score induces nonnegative selected residual in expectation.

Status: Standard order-statistic intuition. The paper states it as a diagnostic proposition under explicit assumptions rather than as a theorem about all world models.

Attack: Candidate means and score errors are not independent in real learned dynamics. Response: The proposition is only a mechanism sketch. The benchmark measures the actual selected sample-over-mean gap.

## Formal Claim B: Lower-confidence selection can avoid exploit tails

Claim: If a lower-confidence score ranks every exploit-tail candidate below a safe candidate, a max selector cannot select the exploit tail.

Status: Trivial deterministic margin claim. Useful only as a repair rationale.

Attack: The margin condition is unverified in real tasks. Response: The experiment reports the empirical version: regret, OOD mass, and tail selection rate.

## Empirical Claim C: Sampled-posterior max fails in this benchmark

Claim: In the full sweep, sampled-posterior max selection at N=128 has substantially larger regret and selected OOD mass than posterior mean or calibrated pessimism.

Status: Verified by `results/full/summary.csv`.

## Empirical Claim D: Calibrated pessimism repairs the failure

Claim: Calibrated pessimistic selection reduces high-N regret from the posterior-sampled selector while staying close to the candidate oracle.

Status: Verified by `results/full/summary.csv`. It is not claimed to dominate posterior mean in this benchmark.
