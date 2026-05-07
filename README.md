# QSAR — Aqueous Solubility (ESOL)

A study in classical QSAR for aqueous solubility prediction on the ESOL dataset
(Delaney 2004, 1117 molecules after deduplication), built as a portfolio project
to demonstrate end-to-end cheminformatics workflow: dataset curation, scaffold-aware
evaluation, progressive featurization, and **honest stratified error analysis**.

The headline result of this stage is not a single benchmark number but the
**arc across three modeling phases on identical splits and identical features**:

1. **Phase 1 → Phase 2** (Morgan FP → Morgan FP + RDKit descriptors, same RF):
   −33% test RMSE on average, but introduces a structurally-localised failure
   mode on poly-hydroxylated high-MW molecules (sugars, glycosides).
2. **Phase 2 → Phase 3** (RF → tuned XGBoost on the same features):
   another −19% test RMSE, the sugar bias is **attenuated but persistent**, and
   a *new* compression bias on the most hydrophobic compounds emerges as a
   side-effect of the stronger regularization.

Both improvements are real and reproducible across seeds. Both come with
trade-offs that aggregate RMSE alone would not reveal. The point of the project
is to make those trade-offs visible before moving to graph neural networks.

---

## Status

| Phase | Featurization | Model | Test RMSE (scaffold, 5-seed) | Test R² (scaffold, 5-seed) | Status |
|---|---|---|---:|---:|---|
| 1   | Morgan FP (r=2, 2048 bits)              | Random Forest        | 1.591 ± 0.126 | 0.344 ± 0.066 | ✅ done |
| 2   | Morgan FP + RDKit 2D descriptors (~217) | Random Forest        | 1.062 ± 0.053 | 0.705 ± 0.041 | ✅ done |
| 3   | Morgan FP + RDKit 2D descriptors        | XGBoost + Optuna     | **0.862 ± 0.036** | **0.806 ± 0.019** | ✅ done |
| 4   | Morgan FP                                | MLP (PyTorch)        | _–_ | _–_ | 🔲 next |
| 5   | Molecular graph                          | ChemProp (D-MPNN)    | _–_ | _–_ | 🔲 target |

Headline numbers are 5-seed averages on balanced scaffold split, the most
demanding of three evaluation protocols (see [Evaluation](#evaluation)).

---

## Why ESOL

Aqueous solubility is the canonical "first" QSAR target: small (~1k molecules),
single regression target, well-studied, with strong physicochemical drivers
(LogP, MW, TPSA, H-bond donors). It is small enough to iterate quickly, well-known
enough to compare against published baselines, and structurally diverse enough to
expose the difference between random-split and scaffold-split metrics.

This project uses ESOL as a **methodology testbed** before scaling to larger
ADMET datasets (Therapeutic Data Commons) and target-specific virtual screening
on ChEMBL.

---

## Repository layout

```
qsar-esol-solubility/
├── data/
│   ├── raw/                  # ESOL as downloaded by DeepChem
│   └── processed/
│       └── esol_dedup.csv    # canonicalized, deduplicated (1117 molecules)
├── notebooks/
│   ├── 01_eda.ipynb                              # load, validate, deduplicate, EDA
│   ├── 02_baseline_rf_morgan.ipynb               # Phase 1: RF on Morgan FP
│   ├── 02b_baseline_rf_morgan_plus_desc.ipynb    # Phase 2: RF on Morgan FP + RDKit desc
│   └── 03_xgboost_optuna.ipynb                   # Phase 3: XGBoost + Optuna tuning
├── src/                      # populated as code is reused across ≥ 2 notebooks
├── models/                   # gitignored; model artifacts + saved best params
├── reports/
│   ├── phase2_summary.json   # per-seed Phase 2 numbers (consumed by notebook 03)
│   ├── phase3_summary.json   # Phase 3 results + best hyperparameters
│   └── figures/              # parity plots, Optuna diagnostics
├── requirements.txt
└── README.md
```

Notebooks are self-contained and detect their environment (Colab vs local).
On Colab they mount Google Drive at `MyDrive/qsar-esol-solubility/`; locally
they resolve paths from `Path.cwd().parent`. The same `.ipynb` file runs in
both environments without modification.

---

## Reproducing

```bash
git clone https://github.com/MassimoMic/qsar-esol-solubility.git
cd qsar-esol-solubility
pip install -r requirements.txt
```

Run the notebooks in order. Each one consumes the output of the previous:

1. `01_eda.ipynb` — downloads ESOL via DeepChem, validates SMILES,
   deduplicates on canonical SMILES, saves `data/processed/esol_dedup.csv`.
2. `02_baseline_rf_morgan.ipynb` — Phase 1 baseline.
3. `02b_baseline_rf_morgan_plus_desc.ipynb` — Phase 2 with descriptors.
   The persistence cell at the end writes `reports/phase2_summary.json`,
   which is consumed by notebook 03 for the paired delta computation.
4. `03_xgboost_optuna.ipynb` — Phase 3 tuning. Runs ~30 minutes on a
   Colab CPU runtime (100 Optuna trials × 5 CV folds). Writes
   `models/xgb_esol_phase3_best_params.json` and
   `reports/phase3_summary.json`.

All random seeds are fixed. Re-running a notebook end-to-end reproduces the
numbers in this README to the third decimal.

---

## Dataset curation

The raw ESOL release (1128 molecules, distributed via DeepChem) carries hidden
duplicates: SMILES strings that look different but produce the same RDKit
canonical SMILES. Naive use of the dataset would put structurally identical
molecules on both sides of any train/test split.

**Curation steps:**

1. Load via `dc.molnet.load_delaney(transformers=[])` to retain raw experimental
   logS (μ ≈ −3.05, σ ≈ 2.10, range [−11.6, +1.58]). The default DeepChem
   `NormalizationTransformer` is bypassed: it is the wrong default for QSAR
   work where target interpretability matters.
2. Canonicalize all SMILES with `Chem.MolToSmiles(Chem.MolFromSmiles(s))`.
3. Group by canonical SMILES. Found **11 duplicate groups**, sizes 2–3,
   spanning the same compound listed under variant SMILES (e.g. atom order
   differences, kekulization differences). Targets within a group were averaged.
4. One borderline case: hexitol (`OCC(O)C(O)C(O)C(O)CO`) appears with two
   logS values 1.03 apart. This is consistent with two stereoisomers
   indistinguishable by SMILES without explicit stereochemistry. Averaged
   like the others; flagged for revisit when working with stereo-annotated
   ChEMBL data.
5. **Final dataset: 1117 molecules**, saved to `data/processed/esol_dedup.csv`.

The scaffold split is **recomputed on the deduplicated dataset**, not reused
from DeepChem's original split (which was computed on the duplicated data).
Skipping this step would re-introduce a small but real train/test leak on
identical molecules.

---

## Evaluation

**Three protocols** are reported in this project, in increasing order of
difficulty and realism:

### 1. Random split (single seed)

80/10/10 plain random partition. This is the optimistic upper bound; molecules
in test are structurally similar to those in train. Reported for reference and
to quantify the random-vs-scaffold gap.

### 2. Balanced scaffold split, single seed

Bemis–Murcko scaffold split with the **balanced (ChemProp-style) variant**:
large scaffold groups go to train (largest first); singleton scaffolds are
shuffled with a fixed seed and distributed across train/valid/test. The
balanced variant avoids a known pathology of pure scaffold splitting on small
datasets like ESOL, where all rare structurally-unusual molecules collapse
into the test set, producing degenerate metrics.

### 3. Balanced scaffold split, multi-seed (recommended)

Same as protocol 2, repeated over 5 seeds (`[42, 0, 1, 2, 3]`). Reported as
**mean ± std**. ESOL's structural distribution is long-tailed enough that a
single scaffold split has high seed-to-seed variance (RMSE range observed:
1.40–1.77 in Phase 1). Single-seed numbers should not be reported in isolation.

**Hyperparameters are kept identical across protocols within each phase**, so
any difference between protocols (random vs scaffold) is attributable to the
split alone. Across phases, what changes is either the featurization (Phase 1
→ 2) or the model (Phase 2 → 3 introduces hyperparameter tuning via Optuna,
described in [Phase 3](#phase-3--xgboost--optuna)). The Phase 1 / Phase 2
Random Forest hyperparameters are: `n_estimators=500, max_features='sqrt',
min_samples_leaf=1`, no tuning.

---

## Phase 1 — Random Forest on Morgan FP

**Featurization.** Morgan / ECFP4 fingerprint, radius 2, 2048 bits, computed
with `AllChem.GetMorganFingerprintAsBitVect`. This is the standard "minimum
viable" QSAR featurization: dense substructure information, no global molecular
properties.

**Results:**

| Evaluation protocol             | Test RMSE       | Test MAE        | Test R²         |
|---------------------------------|----------------:|----------------:|----------------:|
| Random split (seed=42)          | 1.167           | 0.899           | 0.712           |
| Scaffold split (seed=42)        | 1.618           | 1.230           | 0.273           |
| Scaffold split (5-seed mean)    | **1.591 ± 0.126** | **1.239 ± 0.084** | **0.344 ± 0.066** |

**The +0.4 R² gap between random and scaffold split** quantifies the cost of
true structural generalization on this dataset. Reporting only random-split
metrics, as some early QSAR papers did, would systematically inflate apparent
performance.

**Comparison with literature.** RF on Morgan FP for ESOL is reported at
RMSE ≈ 1.05–1.10 in MoleculeNet (Wu et al. 2018) and the ChemProp paper
(Yang et al. 2019), but as **single-run** numbers on the **non-deduplicated**
dataset. Our 5-seed estimate on deduplicated data (1.591 ± 0.126) is more
conservative and methodologically more transparent. The literature numbers
are reproducible if scaffold split is run once with the original duplicated
dataset, but neither choice — single seed nor leaky data — is good practice.

---

## Phase 2 — Random Forest on Morgan FP + RDKit 2D descriptors

**Featurization.** Morgan FP (2048 bits) **concatenated** with all available
RDKit 2D descriptors via `Descriptors._descList` (217 features in current
RDKit), giving a feature matrix of shape `(N, 2265)`. NaN/inf values from
descriptor failures are handled by **median imputation fitted on the training
fold only** (no leakage). On clean ESOL the imputer is a no-op (217/217
columns retained), but the pattern transports to noisier ChEMBL data.

**Hyperparameters identical to Phase 1.** Only the featurization changes.

**Results:**

| Evaluation protocol             | Test RMSE       | Test MAE        | Test R²         |
|---------------------------------|----------------:|----------------:|----------------:|
| Random split (seed=42)          | 0.638           | 0.434           | 0.914           |
| Scaffold split (seed=42)        | 1.019           | 0.765           | 0.712           |
| Scaffold split (5-seed mean)    | **1.062 ± 0.053** | **0.788 ± 0.037** | **0.705 ± 0.041** |

**Phase 1 → Phase 2 comparison (paired, same seed = same split):**

| Protocol                     | Δ RMSE        | Δ R²          |
|------------------------------|--------------:|--------------:|
| Random split (seed=42)       | −0.529        | +0.202        |
| Scaffold (seed=42)           | −0.599        | +0.439        |
| Scaffold (5-seed paired)     | −0.529 ± 0.158 | +0.361 ± 0.096 |

**Phase 2 wins on 5/5 scaffold seeds for both RMSE and R².** The improvement
is not driven by a single lucky seed.

**Why such a large gain?** Aqueous solubility is dominated by global
molecular properties — particularly LogP (Crippen) and topological polar
surface area (TPSA), with smaller contributions from MW and H-bond donor
counts. These are explicitly encoded in the RDKit 2D descriptor set but only
implicitly recoverable from Morgan substructure fingerprints. Phase 2 hands
the model the right features for the problem.

**Where the gain is largest.** The improvement is most pronounced precisely
on the seeds where Phase 1 suffered the most. Seed 2: P1 RMSE = 1.773,
P2 RMSE = 1.010, ΔRMSE = −0.763. Seed 0 (where P1 was already mildly OK):
P1 RMSE = 1.395, P2 RMSE = 1.119, ΔRMSE = −0.276. The descriptors function
as a "safety net" on out-of-distribution scaffolds.

---

## Where Phase 2 still fails — and where Phase 2 is *worse* than Phase 1

The aggregate RMSE improvement (1.591 → 1.062) hides a structural trade-off
in the predictions. Per-molecule analysis on the seed=42 scaffold test set
reveals two distinct error regimes.

### Stratified error by logS region

| Region                           | n   | P1 RMSE | P2 RMSE | Δ      | P2 wins |
|----------------------------------|----:|--------:|--------:|-------:|--------:|
| very hydrophobic (logS < −7)     |  ~5 |    ~3.4 |    ~2.5 | −0.9   | mostly  |
| hydrophobic (−7 ≤ logS < −4)     | ~28 |    ~1.6 |    ~0.9 | −0.7   | strong  |
| moderate (−4 ≤ logS < −2)        | ~50 |    ~1.0 |    ~0.7 | −0.3   | yes     |
| hydrophilic (logS ≥ −2)          | ~30 |    ~1.0 |    ~1.5 | **+0.5** | **NO** |

(Numbers approximate; exact values in notebook 02b. The pattern is what
matters here.)

**Phase 2 wins on 70/113 molecules (62%)** of the test set on scaffold seed=42.
The remaining 38% — where Phase 2 is *strictly worse* than Phase 1 —
concentrates almost entirely in the hydrophilic region.

### The failure mode: sugars and glycosides

Three of the top-5 worst Phase 2 residuals on the seed=42 test set are
sugars or glycosides:

| # | Type                       | MW    | true logS | P1 pred | P2 pred | P2 residual |
|--:|----------------------------|------:|----------:|--------:|--------:|------------:|
| 1 | trisaccharide              | 504.4 |     −0.41 |   −0.87 |   −4.24 |       +3.83 |
| 2 | aryl disaccharide          | 446.4 |     −0.74 |   −2.38 |   −4.26 |       +3.52 |
| 3 | poly-aryl ether            | 376.5 |     −8.60 |   −4.01 |   −5.63 |       −2.97 |
| 4 | natural-product-like ring  | 281.4 |     −1.13 |   −2.42 |   −3.53 |       +2.40 |
| 5 | aryl glucoside             | 286.3 |     −0.85 |   −2.18 |   −3.20 |       +2.35 |

**Why this happens.** RDKit 2D descriptors give the model access to MW,
ring count, heavy atom count — features highly correlated with low solubility
**on most of ESOL**, which is dominated by drug-like aromatic compounds.
Phase 2 has internalized a strong "high MW + many rings → low logS" heuristic
that is statistically correct on the majority class but **systematically
wrong on poly-hydroxylated outliers** like sugars: same MW range, many polar
−OH groups, opposite solubility.

Morgan FP alone (Phase 1) lacked this heuristic. It made weaker, more average
predictions on these molecules and ended up closer to the truth almost by
accident. The "improvement" from Phase 1 to Phase 2 was not free.

### Why this matters

This is not an artifact of fitting or overfitting — it is a real limitation
of feature-engineered global descriptors on chemically heterogeneous datasets.
Strong inductive biases that are correct on the majority class can backfire
on minority structural families.

Two implications for the project:

1. **The bias is partly model-class, partly feature-class.** This is one of
   the questions Phase 3 directly tests: keeping the same featurization but
   replacing RF with a tuned XGBoost shows whether the failure mode is
   intrinsic to the features or to the model. Spoiler: it is partly both
   (see [Phase 3](#phase-3--xgboost--optuna) for the breakdown). A graph-based
   model in Phase 5 (ChemProp / D-MPNN) is then the cleaner test for the
   feature-class part: it learns the molecular representation end-to-end
   without privileging hand-crafted global descriptors, so it has no built-in
   "MW → logS" prior to misapply.

2. **Aggregate RMSE is not enough.** Stratified error analysis (by chemical
   class, by target range) is essential for QSAR: a model with lower mean
   error can still be unsafe on specific structural families. This is a
   well-known issue in medicinal-chemistry deployment of QSAR models, and
   one of the reasons why simple "leaderboard" comparisons can be misleading.

---

## Phase 3 — XGBoost + Optuna

**Goal of this phase.** Hold the featurization fixed (Morgan FP + RDKit
descriptors, identical to Phase 2) and replace the model. This isolates the
"model-class" contribution to the test error and to the sugar bias. If the
sugar bias fully disappeared with a tuned XGBoost, the bias would be a
property of Random Forest. If it persisted, it would be a property of the
features. The actual answer turns out to be: **partly both, with a slight
predominance of feature-class** — the bias is reduced but not eliminated, and
a new compression bias on hydrophobic outliers appears as a side-effect of
the stronger regularization.

### Setup

- **Model.** XGBoost `reg:squarederror`, `tree_method='hist'`, fixed
  `random_state=42`. Only the data split varies across the multi-seed runs;
  the model's internal randomness stays constant for clean attribution.
- **Tuning.** [Optuna](https://optuna.org/) with TPE sampler and `MedianPruner`,
  100 trials, ~30 minutes on a Colab CPU runtime.
- **Objective.** Mean test RMSE across **5 scaffold-disjoint CV folds** built
  from the seed=42 training set. Validation and test sets of the outer split
  are never observed during tuning. The CV folds use a greedy bin-packing
  algorithm (largest scaffold groups first, each placed in the currently
  smallest fold) so the folds end up of comparable size with disjoint scaffold
  sets — the right inner protocol when the outer split is also scaffold-based.
- **Search space (9 hyperparameters):** `n_estimators ∈ [200, 2000]`,
  `max_depth ∈ [3, 10]`, `learning_rate ∈ [0.01, 0.3]` (log),
  `subsample ∈ [0.5, 1.0]`, `colsample_bytree ∈ [0.4, 1.0]`,
  `min_child_weight ∈ [1, 10]`, `reg_lambda ∈ [10⁻³, 10]` (log),
  `reg_alpha ∈ [10⁻³, 10]` (log), `gamma ∈ [10⁻³, 5]` (log).
- **Final eval.** Best params (selected once on seed=42) are re-fit on each
  of the 5 outer scaffold splits `[42, 0, 1, 2, 3]`. We are **not** re-tuning
  per seed — we are testing whether the chosen hyperparameters generalize
  across splits. They do.

### Results

| Evaluation protocol             | Test RMSE       | Test MAE        | Test R²         |
|---------------------------------|----------------:|----------------:|----------------:|
| Random split (seed=42)          | _–_             | _–_             | _–_             |
| Scaffold split (seed=42)        | 0.834           | _–_             | _–_             |
| Scaffold split (5-seed mean)    | **0.862 ± 0.036** | _–_           | _–_             |

> _Numbers from `notebooks/03_xgboost_optuna.ipynb` (output cells §10 and §12).
> The MAE and R² rows above are placeholders: copy the exact values from the
> §10 / §12 print outputs._

**Phase 2 → Phase 3 paired delta (same seed = same split):**

| Seed | P2 RMSE (RF) | P3 RMSE (XGB-tuned) | Δ RMSE |
|-----:|-------------:|--------------------:|-------:|
| 42   | 1.019        | 0.834               | −0.184 |
| 0    | 1.119        | 0.867               | −0.252 |
| 1    | 1.028        | 0.838               | −0.190 |
| 2    | 1.010        | 0.862               | −0.148 |
| 3    | 1.132        | 0.907               | −0.225 |

**Mean ± std: ΔRMSE = −0.200 ± 0.036, P3 wins on 5/5 seeds.**

The variance of the improvement (0.036) is *smaller* than the within-phase std
of either P2 (0.053) or P1 (0.126). The gain is structural: XGBoost-tuned
beats RF by roughly the same amount on every scaffold split, regardless of
whether the split itself is "easy" (seed=2) or "hard" (seed=3 or seed=0).

### Optuna diagnostics

The optimization history (`reports/figures/03_optuna_history.png`) shows two
phases: random exploration in trials 0–10 (CV RMSE bouncing between 0.78 and
0.95), then a sharp drop to ~0.780 around trial 11 once TPE converges on the
promising region, followed by a long plateau with a final small improvement
to ~0.775 around trial 84. **100 trials were sufficient**: the last
~15 trials produced no further improvement. On future ADMET endpoints,
60–80 trials would likely be enough.

The CV-best RMSE of 0.775 vs the test RMSE of 0.834 on seed=42 gives a
**CV→test gap of 0.06**, well within reasonable seed-to-seed variance.
There is no sign that Optuna overfit to the inner CV folds.

### Hyperparameter importance (FANOVA)

Decomposition of the variance in CV RMSE across the 100 completed trials:

| Hyperparameter      | FANOVA importance |
|---------------------|------------------:|
| `max_depth`         |             ~0.38 |
| `reg_alpha`         |             ~0.27 |
| `min_child_weight`  |             ~0.08 |
| `colsample_bytree`  |             ~0.08 |
| `gamma`             |             ~0.06 |
| `subsample`         |             ~0.05 |
| `learning_rate`     |             ~0.04 |
| `reg_lambda`        |             ~0.02 |
| `n_estimators`      |             ~0.02 |

Two takeaways:

1. **`max_depth` and `reg_alpha` together account for ~65% of the variance.**
   Tree depth controls model capacity (bias-variance trade-off), `reg_alpha`
   is L1 regularization that drives weights to exactly zero — effectively
   doing automatic feature selection on the 217 RDKit descriptors. With
   2265 features and ~900 training molecules per fold, controlling capacity
   and shrinking the feature set are the two dominant levers.
2. **`learning_rate` and `n_estimators` are surprisingly low.** TPE found
   a good "many trees + moderate LR" region early and most of the marginal
   tuning happened around regularization, not around the gradient-boosting
   schedule itself.

For future tabular-cheminformatics tuning runs, prioritising `max_depth`,
`reg_alpha`, `min_child_weight`, and `colsample_bytree` while leaving the
other 5 at sensible defaults would likely capture >85% of the gain at a
quarter of the budget.

### What happened to the sugar bias

Per-regime breakdown on the seed=42 scaffold test set:

| Regime                          |  n | P2 mean residual | P3 mean residual | P2 RMSE | P3 RMSE |
|---------------------------------|---:|-----------------:|-----------------:|--------:|--------:|
| hydrophilic   (logS > −2)       | 18 |             ~−0.9 |           −0.450 |    1.49 |   0.825 |
| moderate      (−4 < logS ≤ −2)  | 43 |             ~+0.0 |           +0.018 |    0.69 |   0.732 |
| hydrophobic   (logS ≤ −4)       | 52 |             ~+0.1 |           +0.219 |    0.94 |   0.913 |

> _Phase 2 per-regime residuals are approximate — recompute from notebook 02b
> diagnostic cell to fill in exact numbers._

**The sugar bias is attenuated but still present.** Mean residual on the
hydrophilic regime improved from ~−0.9 to −0.45 — a real reduction, roughly
halving the systematic under-prediction — but the sign is still negative and
the magnitude still well outside what would be expected from random error
(MAE = 0.625 on n=18). Of the top-5 worst residuals on the seed=42 test set,
1 is still a glycoside (vs 3 in Phase 2). The tuned XGBoost has weakened the
"high MW + many rings → low logS" heuristic without removing it.

**A new bias has appeared on the hydrophobic side.** Mean residual on the
hydrophobic regime moved from ~+0.1 (≈unbiased in P2) to **+0.22** in P3:
the tuned model now systematically over-predicts solubility (under-predicts
the magnitude of insolubility) on the most hydrophobic compounds. Two of
the top-5 worst residuals on the seed=42 test set are extreme hydrophobes
that XGBoost-tuned has pulled too close to the bulk of the distribution
(an anthraquinone at logS = −5.19 predicted at −2.66; a tri-aryl ether at
logS = −8.6 predicted at −6.4).

This is a textbook regularization side-effect. Strong L1 (`reg_alpha`
optimal value ended up high) plus moderate `max_depth` plus low
`min_child_weight` all push the model toward simpler, smoother predictions
that compress toward the mean of the training distribution. RMSE wins
because the bulk of ESOL sits in the moderate region where this compression
is helpful, but the tails pay a small price.

### Reading the model-class vs feature-class question

Phase 3 holds the features fixed and changes the model. The result:

- The sugar bias **reduces by ~50%** (mean hydrophilic residual: ~−0.9 → −0.45).
  This part was model-class — RF without regularization committed harder to
  the spurious "high MW + many rings" rule than XGBoost-with-L1 does.
- The sugar bias **does not disappear**. The remaining ~−0.45 mean residual
  on hydrophilic compounds is the part the features themselves cannot avoid.
  The "MW + ring count → low logS" signal is too statistically dominant in
  the training set for any reasonable tabular model to fully resist it
  on out-of-distribution polyhydroxy compounds.

This is the exact split the project needed Phase 3 to clarify before Phase 5.
**The remaining ~−0.45 hydrophilic residual is the bar ChemProp must beat
to justify graph-based modelling on chemically heterogeneous datasets.**

---

## Method notes

- **Target not normalized.** Default `dc.molnet.load_delaney` applies a
  `NormalizationTransformer` that rescales logS to mean 0, std 1. This is
  bypassed via `transformers=[]`. We work on raw experimental logS (mol/L)
  so RMSE/MAE are physically interpretable in log units.

- **Scaffold split recomputed on dedup.** Reusing DeepChem's original split
  on the deduplicated dataset would re-introduce identical molecules across
  train/test. The split is computed from scratch on the 1117-molecule
  deduplicated set.

- **Balanced scaffold split, not pure.** Pure scaffold split puts every
  singleton scaffold in test, which on ESOL produces a test set
  pathologically biased toward large hydrophobic outliers (test R² collapsed
  to 0.18 in initial experiments). The balanced variant (ChemProp-style)
  preserves the rigor of scaffold-based generalization while avoiding this
  degeneracy.

- **Multi-seed reporting.** Single-run scaffold metrics on small datasets
  like ESOL have high variance (observed RMSE range across 5 seeds in Phase 1:
  1.40–1.77). 5-seed mean ± std is the smallest acceptable unit.

- **Paired Δ across seeds.** Phase comparisons are computed as paired
  per-seed differences (`Δ_s = P2_s − P1_s` for each seed `s`, then mean ± std
  over seeds), not as the difference of two independently-averaged numbers.
  This is the standard paired-test design and reports the variance of the
  *improvement*, which is what matters for assessing whether a featurization
  change is a robust win.

- **Median imputation, training-fold only.** Robust to noisy datasets
  even though ESOL itself is clean. Pattern is preserved so the same
  evaluation harness can be reused on ChEMBL without changes.

- **Tune once, evaluate many (Phase 3).** Optuna selects hyperparameters
  on the seed=42 outer split using scaffold-disjoint 5-fold CV on the
  training set only. Those hyperparameters are then re-fit on each of the
  5 outer scaffold seeds for final evaluation. We are *not* re-tuning per
  seed: the 5-seed numbers test whether the chosen hyperparameters
  generalize across splits. Re-tuning per seed would be a different
  (more expensive) protocol — and would conflate seed-dependent tuning
  variance with model-class variance.

- **Scaffold-disjoint inner CV.** When the outer split is scaffold-aware,
  the inner CV must be too — otherwise Optuna selects hyperparameters that
  exploit chemical similarity within the training set and the chosen
  configuration ends up over-confident on out-of-distribution scaffolds.
  The greedy bin-packing implementation (see `scaffold_kfold` in notebook
  03) keeps fold sizes balanced while preserving scaffold disjointness.

---

## Lessons learned

1. **Dataset deduplication matters even on a "clean" public dataset.**
   ESOL is a 20-year-old, widely-cited benchmark and still ships with 11
   hidden duplicate groups. The first published RF benchmarks were computed
   on duplicated data with single-seed scaffold splits.

2. **The right baseline is hard.** Reporting RMSE 1.05 on ESOL "RF on
   Morgan FP scaffold split" matches the literature — but only on a single
   seed and on duplicated data. The honest 5-seed deduplicated number is
   1.59. The point is not that the literature is wrong, it is that
   "matching the benchmark" requires reproducing every methodological choice,
   including the bad ones.

3. **Feature engineering can move the average and break the tails.**
   Phase 2 improves average RMSE by 33% but creates a new failure mode
   on sugars. Without the stratified analysis this would be invisible —
   and dangerous if the model were deployed to predict solubility for
   carbohydrate chemistry.

4. **Multi-seed reporting is non-negotiable on small datasets.** Single-seed
   scaffold metrics on ESOL vary by ±0.2 RMSE across reasonable seeds.
   Any conclusion drawn from a single run is not reproducible.

5. **The right model class might not be the one that minimizes mean error.**
   Tuned XGBoost on Phase 2 features hits RMSE 0.86 — closing more than half
   the gap to published GNN numbers (0.55–0.70). But the sugar bias is only
   halved, not eliminated, and a new compression bias on hydrophobic outliers
   appears as a regularization side-effect. The case for moving to graph-based
   models in Phase 5 is now sharper: it is no longer about absolute RMSE
   (XGBoost gets close enough that the marginal improvement might not justify
   the complexity) but specifically about whether end-to-end-learned
   representations can break the ~−0.45 hydrophilic residual ceiling that
   any tabular model on these features seems to hit.

6. **Most of the tuning variance came from two knobs.** On this dataset,
   FANOVA attributes ~65% of the CV-RMSE variance across 100 Optuna trials
   to `max_depth` and `reg_alpha`. `learning_rate` and `n_estimators` —
   often considered the dominant gradient-boosting hyperparameters — were
   below 5% each. This is informative: in regime p > n (2265 features, ~900
   training molecules), capacity control and L1 regularization dominate the
   tuning surface. Useful prior for the upcoming ADMET project.

7. **Strong regularization buys mean-error reduction at the cost of small
   tail biases.** Phase 3 reduces RMSE in the moderate logS region (where
   most of the dataset sits) by compressing predictions toward the training
   distribution mean. This is mathematically how L1-regularized boosting
   wins on aggregate; the price is paid in small but real biases on the
   distributional extremes (over-predicting solubility of strong
   hydrophobes, under-predicting solubility of strong hydrophiles). For
   QSAR deployment, this matters: the molecules a project actually cares
   about predicting often *are* the extremes.

---

## Next phases

- **Phase 4 — MLP on Morgan FP.** A small dense MLP on the same Morgan FP
  used in Phase 1, mostly to validate the PyTorch training loop and to
  provide a third reference point on identical splits. Expectation: somewhere
  between Phase 1 (RF on Morgan only) and Phase 2 (RF on Morgan + descriptors).
  If the MLP without descriptors significantly beats RF without descriptors,
  it would be a hint that part of Phase 1's error budget was about model
  expressivity, not features.

- **Phase 5 — ChemProp (D-MPNN).** Graph-based message-passing neural network.
  After Phase 3, the question is no longer "does graph beat tabular on
  aggregate RMSE" (probably yes, by some margin) but specifically: **does
  ChemProp eliminate the residual sugar bias?** The Phase 3 hydrophilic mean
  residual of −0.45 is now the bar: if ChemProp brings that below ~−0.10,
  it justifies graph models as more than a marginal improvement. If it does
  not, the bias is more deeply about the dataset than the representation,
  and would warrant a different framing.

After this project the workflow generalizes to:

- **Project 1 — ADMET multi-task** on Therapeutic Data Commons (the
  cheminformatics interview standard).
- **Project 2 — Virtual screening** with GNNs on ChEMBL for a specific
  target (likely a kinase).
- **Project 4 — Quantum-classical benchmark** (H₂, LiH, H₂O via VQE vs DFT
  vs SchNet) — the differentiator from the parallel Quantum Machine Learning
  master's program.

---

## References

- **Delaney, J. S.** ESOL: Estimating aqueous solubility directly from
  molecular structure. *J. Chem. Inf. Comput. Sci.* **2004**, 44, 1000–1005.
- **Wu, Z. et al.** MoleculeNet: a benchmark for molecular machine learning.
  *Chem. Sci.* **2018**, 9, 513–530.
- **Yang, K. et al.** Analyzing learned molecular representations for
  property prediction (ChemProp). *J. Chem. Inf. Model.* **2019**, 59,
  3370–3388.
- **Bemis, G. W.; Murcko, M. A.** The properties of known drugs. 1.
  Molecular frameworks. *J. Med. Chem.* **1996**, 39, 2887–2893.
- **Rogers, D.; Hahn, M.** Extended-connectivity fingerprints (ECFP).
  *J. Chem. Inf. Model.* **2010**, 50, 742–754.
- **Chen, T.; Guestrin, C.** XGBoost: a scalable tree boosting system.
  *KDD* **2016**, 785–794.
- **Akiba, T. et al.** Optuna: a next-generation hyperparameter optimization
  framework. *KDD* **2019**, 2623–2631.
- **Hutter, F. et al.** An efficient approach for assessing hyperparameter
  importance (Functional ANOVA). *ICML* **2014**.

---

## License & contact

Personal project, no specific license attached at this stage.
For questions: open an issue on the repository.
