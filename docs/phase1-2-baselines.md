[← README](../README.md) · [Methodology](methodology.md) · [Phase 3 →](phase3-xgboost.md)

# Phases 1–2 — Random Forest baselines

Two phases at constant model class, varying only the featurization. Phase 2
improves aggregate RMSE by 33% and introduces a structurally-localised failure
mode that aggregate RMSE does not reveal.

---

## Phase 1 — Random Forest on Morgan FP

**Featurization.** Morgan / ECFP4 fingerprint, radius 2, 2048 bits, computed with
`AllChem.GetMorganFingerprintAsBitVect`. The standard "minimum viable" QSAR
featurization: dense substructure information, no global molecular properties.

**Results:**

| Evaluation protocol | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Random split (seed=42) | 1.167 | 0.899 | 0.712 |
| Scaffold split (seed=42) | 1.618 | 1.230 | 0.273 |
| Scaffold split (5-seed mean) | **1.591 ± 0.126** | **1.239 ± 0.084** | **0.344 ± 0.066** |

**The +0.4 R² gap between random and scaffold split** quantifies the cost of true
structural generalization on this dataset. Reporting only random-split metrics,
as some early QSAR papers did, would systematically inflate apparent performance.

### Comparison with published baselines

RF on Morgan FP for ESOL is reported at RMSE ≈ 1.05–1.10 in MoleculeNet (Wu et
al. 2018) and the ChemProp paper (Yang et al. 2019), as **single-run** numbers on
the **non-deduplicated** dataset. The 5-seed estimate here on deduplicated data
is 1.591 ± 0.126.

Four methodological choices differ between those numbers and this one:

1. deduplication of the dataset (11 groups, ~1% of molecules),
2. balanced vs pure scaffold split,
3. five seeds vs one,
4. Random Forest hyperparameters.

The published numbers are reproducible by running a single scaffold split on the
original duplicated dataset. **The individual contribution of each of the four
choices to the ~0.5 RMSE gap has not been measured, and no causal attribution is
claimed.** A factorial ablation isolating each is the current open work item; see
[Open questions](../README.md#open-questions).

---

## Phase 2 — Random Forest on Morgan FP + RDKit 2D descriptors

**Featurization.** Morgan FP (2048 bits) **concatenated** with all available
RDKit 2D descriptors via `Descriptors._descList` (217 features in current RDKit),
giving a feature matrix of shape `(N, 2265)`. NaN/inf values from descriptor
failures are handled by **median imputation fitted on the training fold only**
(no leakage). On clean ESOL the imputer is a no-op (217/217 columns retained),
but the pattern transports to noisier ChEMBL data.

**Hyperparameters identical to Phase 1.** Only the featurization changes.

**Results:**

| Evaluation protocol | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Random split (seed=42) | 0.638 | 0.434 | 0.914 |
| Scaffold split (seed=42) | 1.019 | 0.765 | 0.712 |
| Scaffold split (5-seed mean) | **1.062 ± 0.053** | **0.788 ± 0.037** | **0.705 ± 0.041** |

**Phase 1 → Phase 2 paired delta (same seed = same split):**

| Protocol | Δ RMSE | Δ R² |
|---|---|---|
| Random split (seed=42) | −0.529 | +0.202 |
| Scaffold (seed=42) | −0.599 | +0.439 |
| Scaffold (5-seed paired) | −0.529 ± 0.158 | +0.361 ± 0.095 |

**Phase 2 wins on 5/5 scaffold seeds for both RMSE and R².** The improvement is
not driven by a single lucky seed.

**Why such a large gain?** Aqueous solubility is dominated by global molecular
properties — particularly LogP (Crippen) and topological polar surface area
(TPSA), with smaller contributions from MW and H-bond donor counts. These are
explicitly encoded in the RDKit 2D descriptor set but only implicitly recoverable
from Morgan substructure fingerprints. Phase 2 hands the model the right features
for the problem.

**Where the gain is largest.** The improvement is most pronounced precisely on
the seeds where Phase 1 suffered most. Seed 2: P1 RMSE = 1.773, P2 RMSE = 1.010,
ΔRMSE = −0.763. Seed 0 (where P1 was already mildly OK): P1 = 1.395, P2 = 1.119,
ΔRMSE = −0.276. The descriptors function as a safety net on out-of-distribution
scaffolds.

---

## Where Phase 2 still fails — and where it is *worse* than Phase 1

The aggregate RMSE improvement (1.591 → 1.062) hides a structural trade-off.
Per-molecule analysis on the seed=42 scaffold test set reveals two distinct error
regimes.

### Stratified error by logS region

> **⚠️ Pending: exact values.** The table below is populated with approximate
> values. Exact per-regime residuals require running the diagnostic cell of
> `notebooks/02b_baseline_rf_morgan_plus_desc.ipynb` and persisting
> `reports/phase2_stratified.json` in the schema of `phase5_stratified.json`.
> Remove this notice once filled.

| Region | n | P1 RMSE | P2 RMSE | Δ | P2 wins |
|---|---|---|---|---|---|
| very hydrophobic (logS < −7) | ~5 | ~3.4 | ~2.5 | −0.9 | mostly |
| hydrophobic (−7 ≤ logS < −4) | ~28 | ~1.6 | ~0.9 | −0.7 | strong |
| moderate (−4 ≤ logS < −2) | ~50 | ~1.0 | ~0.7 | −0.3 | yes |
| hydrophilic (logS ≥ −2) | ~30 | ~1.0 | ~1.5 | **+0.5** | **NO** |

**Phase 2 wins on 70/113 molecules (62%)** of the test set on scaffold seed=42.
The remaining 38% — where Phase 2 is *strictly worse* than Phase 1 —
concentrates almost entirely in the hydrophilic region.

Note that the extreme bins carry few molecules (n ≈ 5 in the most hydrophobic
bin). Bootstrap intervals on the regime-level means are outstanding work; the
conclusions below should be read as directional until they exist.

### The failure mode: sugars and glycosides

Three of the top-5 worst Phase 2 residuals on the seed=42 test set are sugars or
glycosides:

| # | Type | MW | true logS | P1 pred | P2 pred | P2 residual |
|---|---|---|---|---|---|---|
| 1 | trisaccharide | 504.4 | −0.41 | −0.87 | −4.24 | +3.83 |
| 2 | aryl disaccharide | 446.4 | −0.74 | −2.38 | −4.26 | +3.52 |
| 3 | poly-aryl ether | 376.5 | −8.60 | −4.01 | −5.63 | −2.97 |
| 4 | natural-product-like ring | 281.4 | −1.13 | −2.42 | −3.53 | +2.40 |
| 5 | aryl glucoside | 286.3 | −0.85 | −2.18 | −3.20 | +2.35 |

**Why this happens.** RDKit 2D descriptors give the model access to MW, ring
count, heavy atom count — features highly correlated with low solubility **on
most of ESOL**, which is dominated by drug-like aromatic compounds. Phase 2 has
internalized a strong "high MW + many rings → low logS" heuristic that is
statistically correct on the majority class but **systematically wrong on
poly-hydroxylated outliers** like sugars: same MW range, many polar −OH groups,
opposite solubility.

Morgan FP alone (Phase 1) lacked this heuristic. It made weaker, more average
predictions on these molecules and ended up closer to the truth almost by
accident. The improvement from Phase 1 to Phase 2 was not free.

### Why this matters

This is not an artifact of overfitting — it is a real limitation of
feature-engineered global descriptors on chemically heterogeneous datasets.
Strong inductive biases that are correct on the majority class can backfire on
minority structural families. In applicability-domain terms: the model's domain
of reliable prediction is narrower than its aggregate metric suggests, and the
boundary is chemotype-shaped.

Two implications for the project:

1. **The bias is partly model-class, partly feature-class.** Phase 3 tests this
   directly by keeping the featurization fixed and replacing RF with a tuned
   XGBoost. Phase 5 (ChemProp / D-MPNN) is the cleaner test for the feature-class
   part: it learns the molecular representation end-to-end without privileging
   hand-crafted global descriptors, so it has no built-in "MW → logS" prior to
   misapply.
2. **Aggregate RMSE is not enough.** Stratified error analysis by chemical class
   and by target range is essential: a model with lower mean error can still be
   unsafe on specific structural families.
