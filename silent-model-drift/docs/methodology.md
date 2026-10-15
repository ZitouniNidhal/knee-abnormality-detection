# Methodology

## Measurement target

Silent Model Drift measures the conditional behavior of an API endpoint:

```text
response distribution | prompt, system prompt, request settings, tools, context, endpoint
```

It does not inspect provider weights and does not claim that a detected change is a model replacement. The endpoint is the measured system.

## Experimental design

1. Write a compact sentinel suite representing critical application behaviors.
2. Version the suite and collect a baseline with fixed request settings.
3. Collect at least several independent responses per sentinel in each window.
4. Check that the suite and request-config digests match before comparison.
5. Apply per-sentinel distribution tests and correct for multiple comparisons.
6. Repeat significant runs before operational escalation.

Use more diverse sentinels before adding many nearly identical prompts. Repeated calls to a single prompt are correlated, especially at low temperature, and should not be treated as fully independent evidence.

## Behavioral fingerprints

Raw output is canonicalized with Unicode normalization, whitespace collapse, and canonical JSON serialization where possible. A fingerprint stores a SHA-256 exact-output hash, hashed token and character n-gram sets, length summaries, and a JSON-shape hash. The default artifact omits raw output. These are privacy-conscious features, not a security boundary: low-entropy prompts or outputs can be vulnerable to dictionary attacks.

The detector computes a bounded hybrid distance in `[0, 1]` between fingerprints. It combines token overlap, character n-gram overlap, relative length, and structural agreement. The distance is intentionally transparent and inexpensive; applications with stronger semantic requirements can provide a custom fingerprint or run complementary task-level evaluations.

## Test statistic and p-values

For each sentinel, the implementation computes a Gaussian-kernel maximum mean discrepancy (MMD) between baseline and candidate fingerprints. The kernel bandwidth is the median nonzero pairwise distance in the pooled sample. A random-label permutation test produces the p-value:

```text
p = (1 + number of permuted statistics at least as large as observed)
    / (1 + number of permutations)
```

This construction remains valid under the null hypothesis of exchangeable windows and avoids a parametric model of text output. The report uses Benjamini-Hochberg adjusted p-values for prompt-level alerting. Fisher's combined p-value is reported only as a suite-level descriptive summary because sentinel outcomes can be correlated.

## Noise versus behavior change

Nondeterminism is not a nuisance to discard: it is part of endpoint behavior. The report separately tests:

- permutation-tested cross-window central separation plus medoid displacement; and
- a one-sided increase in within-window average distance.

When a distribution shift accompanies a material, significant medoid displacement, the report labels it `behavioral_shift`. When variability significantly increases without material medoid displacement, it labels it `nondeterminism_increase`. Ambiguous cases remain `distribution_shift` rather than receiving a causal explanation.

## Power planning

The generic planner assumes an equal-size two-sample comparison of a standardized mean-like effect `d` and gives the independent samples needed in each window:

```text
n = 2 × ((z(1 - α / 2) + z(power)) / d)^2
```

With `r` repeats per sentinel and intra-prompt correlation `rho`, the planner uses an effective-sample adjustment:

```text
effective observations per sentinel = r / (1 + (r - 1) × rho)
```

These equations are approximations, not a guarantee for MMD power. For an empirical study, replay historical baseline samples or use a known controlled change to estimate detection power at candidate suite sizes.

## Threats to validity

- Provider A/B routing, region, capacity, retry behavior, or safety systems may shift output without a weight update.
- API defaults, SDK versions, tool definitions, retrieval documents, and system prompts can create false positives if not locked.
- A small suite can miss narrow regressions; a broad suite increases the multiple-testing burden.
- Hash-based lexical fingerprints can miss semantic paraphrases and should be paired with task-specific checks for critical workflows.
- Statistical significance is not necessarily practical significance; use the reported effect size and impacted tags when triaging.

## References

- Gretton, A. et al. [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html), *Journal of Machine Learning Research* 13 (2012), 723–773.
- Good, P. *Permutation, Parametric and Bootstrap Tests of Hypotheses*, 3rd ed., Springer (2005).


