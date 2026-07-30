[← README](../README.md) · [← Phases 1–2](phase1-2-baselines.md) · [Phase 4 →](phase4-mlp.md)

# Phase 3 — XGBoost + Optuna

**Goal.** Hold the featurization fixed (Morgan FP + RDKit descriptors, identical
to Phase 2) and replace the model. This isolates the model-class contribution to
the test error and to the sugar bias. If the sugar bias fully disappeared with a
tuned XGBoost, the bias would be a property of Random Forest. If it persisted, it
would be a property of the features.

The answer: **partly both, with a slight predominance of feature-class** — the
bias is reduced but not eliminated, and a new compression bias on hydrophobic
outliers appears as a side-effect of stronger regularization.

---

## Setup

- **Model.** XGBoost `reg:squarederror`, `tree_method='hist'`, fixed
  `random_state=42`. Only the data split varies across multi-seed runs; the
  model's internal randomness stays constant for clean attribution.
- **Tuning.** Optuna with TPE sampler and `MedianPruner`, 100 trials, ~30 minutes
  on a Colab CPU runtime.
- **Objective.** Mean test RMSE across **5 scaffold-disjoint CV folds** built from
  the seed=42 training set. Validation and test sets of the outer split are never
  observed during tuning. The CV folds use greedy bin-packing (largest scaffold
  groups first, each placed in the currently smallest fold) so folds end up of
  comparable size with disjoint scaffold sets.
- **Search space (9 hyperparameters).** `n_estimators ∈ [200, 2000]`,
  `max_depth ∈ [3, 10]`, `learning_rate ∈ [0.01, 0.3]` (log),
  `subsample ∈ [0.5, 1.0]`, `colsample_bytree ∈ [0.4, 1.0]`,
  `min_child_weight ∈ [1, 10]`, `reg_lambda ∈ [10⁻³, 10]` (log),
  `reg_alpha ∈ [10⁻³, 10]` (log), `gamma ∈ [10⁻³, 5]` (log).
- **Final eval.** Best params (selected once on seed=42) re-fit on each of the 5
  outer scaffold splits `[42, 0, 1, 2, 3]`.

---

## Results

| Evaluation protocol | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Random split (seed=42) | 0.550 | 0.392 | +0.936 |
| Scaffold split (seed=42) | 0.834 | 0.644 | +0.807 |
| Scaffold split (5-seed mean) | **0.862 ± 0.026** | **0.657 ± 0.023** | **+0.807 ± 0.017** |

*Numbers from `reports/phase3_summary.json`.*

**Phase 2 → Phase 3 paired delta (same seed = same split):**

| Seed | P2 RMSE (RF) | P3 RMSE (XGB-tuned) | Δ RMSE |
|---|---|---|---|
| 42 | 1.019 | 0.834 | −0.184 |
| 0 | 1.119 | 0.867 | −0.252 |
| 1 | 1.028 | 0.838 | −0.190 |
| 2 | 1.010 | 0.862 | −0.148 |
| 3 | 1.132 | 0.907 | −0.225 |

**Mean ± std: ΔRMSE = −0.200 ± 0.036, P3 wins on 5/5 seeds.**

The variance of the improvement (0.036) is *smaller* than the within-phase std of
either P2 (0.053) or P1 (0.126). The gain is structural: tuned XGBoost beats RF
by roughly the same amount on every scaffold split, regardless of whether the
split is easy (seed=2) or hard (seed=3, seed=0).

---

## Optuna diagnostics

The optimization history (`reports/figures/03_optuna_history.png`) shows two
phases: random exploration in trials 0–10 (CV RMSE bouncing between 0.78 and
0.95), then a sharp drop to ~0.780 around trial 11 once TPE converges on the
promising region, followed by a long plateau with a final small improvement to
~0.775 around trial 84. **100 trials were sufficient**: the last ~15 produced no
further improvement. On future ADMET endpoints, 60–80 trials would likely be
enough.

The CV-best RMSE of 0.775 vs test RMSE of 0.834 on seed=42 gives a **CV→test gap
of 0.06**, well within reasonable seed-to-seed variance. No sign that Optuna
overfit the inner CV folds.

---

## Hyperparameter importance (FANOVA)

Decomposition of the variance in CV RMSE across the 100 completed trials:

| Hyperparameter | FANOVA importance |
|---|---|
| `max_depth` | ~0.38 |
| `reg_alpha` | ~0.27 |
| `min_child_weight` | ~0.08 |
| `colsample_bytree` | ~0.08 |
| `gamma` | ~0.06 |
| `subsample` | ~0.05 |
| `learning_rate` | ~0.04 |
| `reg_lambda` | ~0.02 |
| `n_estimators` | ~0.02 |

1. **`max_depth` and `reg_alpha` together account for ~65% of the variance.**
   Tree depth controls model capacity; `reg_alpha` is L1 regularization that
   drives weights to exactly zero — effectively automatic feature selection over
   the 217 RDKit descriptors. With 2265 features and ~900 training molecules per
   fold, controlling capacity and shrinking the feature set are the two dominant
   levers.
2. **`learning_rate` and `n_estimators` are surprisingly low.** TPE found a good
   "many trees + moderate LR" region early; most marginal tuning happened around
   regularization, not the gradient-boosting schedule.

For future tabular-cheminformatics tuning, prioritising `max_depth`, `reg_alpha`,
`min_child_weight` and `colsample_bytree` while leaving the other five at
sensible defaults would likely capture >85% of the gain at a quarter of the
budget.

---

## What happened to the sugar bias

Per-regime breakdown on the seed=42 scaffold test set:

> **⚠️ Pending: exact values.** Phase 2 per-regime residuals below are
> approximate. Fill from `reports/phase2_stratified.json` once generated (see
> [Phases 1–2](phase1-2-baselines.md#stratified-error-by-logs-region)). Remove
> this notice once filled.

| Regime | n | P2 mean residual | P3 mean residual | P2 RMSE | P3 RMSE |
|---|---|---|---|---|---|
| hydrophilic (logS > −2) | 18 | ~−0.9 | −0.450 | 1.49 | 0.825 |
| moderate (−4 < logS ≤ −2) | 43 | ~+0.0 | +0.018 | 0.69 | 0.732 |
| hydrophobic (logS ≤ −4) | 52 | ~+0.1 | +0.219 | 0.94 | 0.913 |

**The sugar bias is attenuated but still present.** Mean residual on the
hydrophilic regime improved from ~−0.9 to −0.45 — roughly halving the systematic
under-prediction — but the sign is still negative and the magnitude still well
outside what random error would produce (MAE = 0.625 on n=18). Of the top-5 worst
residuals on seed=42, 1 is still a glycoside (vs 3 in Phase 2). The tuned XGBoost
has weakened the "high MW + many rings → low logS" heuristic without removing it.

**A new bias has appeared on the hydrophobic side.** Mean residual on the
hydrophobic regime moved from ~+0.1 (≈unbiased in P2) to **+0.22**: the tuned
model now systematically over-predicts solubility on the most hydrophobic
compounds. Two of the top-5 worst residuals on seed=42 are extreme hydrophobes
pulled too close to the bulk of the distribution (an anthraquinone at
logS = −5.19 predicted at −2.66; a tri-aryl ether at logS = −8.6 predicted
at −6.4).

This is a textbook regularization side-effect. High optimal `reg_alpha` plus
moderate `max_depth` plus low `min_child_weight` all push the model toward
simpler, smoother predictions that compress toward the training mean. RMSE wins
because the bulk of ESOL sits in the moderate region where compression helps; the
tails pay the price.

---

## Reading the model-class vs feature-class question

Phase 3 holds the features fixed and changes the model. The result:

- The sugar bias **reduces by ~50%** (hydrophilic mean residual ~−0.9 → −0.45).
  This part was model-class — RF without regularization committed harder to the
  spurious "high MW + many rings" rule than XGBoost-with-L1 does.
- The sugar bias **does not disappear**. The remaining ~−0.45 is the part the
  features themselves cannot avoid. The "MW + ring count → low logS" signal is
  too statistically dominant in the training set for any reasonable tabular model
  to fully resist on out-of-distribution polyhydroxy compounds.

**The remaining ~−0.45 hydrophilic residual is the bar ChemProp must beat** to
justify graph-based modelling on chemically heterogeneous datasets.

[Phase 4](phase4-mlp.md) tests a different cross-section of the same question: it
holds the *featurization* of Phase 1 fixed (Morgan FP only) and changes the model
class instead, from RF to a tuned MLP.
