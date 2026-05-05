# QSAR — Aqueous Solubility (ESOL)

A study in classical QSAR for aqueous solubility prediction on the ESOL dataset
(Delaney 2004, 1117 molecules after deduplication), built as a portfolio project
to demonstrate end-to-end cheminformatics workflow: dataset curation, scaffold-aware
evaluation, progressive featurization, and **honest stratified error analysis**.

The headline result of this stage is not a benchmark number — it is the discovery
that **adding global molecular descriptors to a substructure fingerprint reduces
test RMSE by 33% on average but introduces a systematic, structurally-localised
failure mode** on poly-hydroxylated high-MW molecules (sugars, glycosides). Both
the improvement and the trade-off are real and worth understanding before moving
to graph neural networks.

---

## Status

| Phase | Featurization | Model | Test RMSE (scaffold, 5-seed) | Test R² (scaffold, 5-seed) | Status |
|---|---|---|---:|---:|---|
| 1   | Morgan FP (r=2, 2048 bits)              | Random Forest        | 1.591 ± 0.126 | 0.344 ± 0.066 | ✅ done |
| 2   | Morgan FP + RDKit 2D descriptors (~217) | Random Forest        | **1.062 ± 0.053** | **0.705 ± 0.041** | ✅ done |
| 3   | Morgan FP + RDKit 2D descriptors        | XGBoost + Optuna     | _–_ | _–_ | 🔲 next |
| 4   | Morgan FP                                | MLP (PyTorch)        | _–_ | _–_ | 🔲 |
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
│   └── 02b_baseline_rf_morgan_plus_desc.ipynb    # Phase 2: RF on Morgan FP + RDKit desc
├── src/                      # populated as code is reused across ≥ 2 notebooks
├── models/                   # gitignored; model artifacts
├── reports/figures/          # parity plots, diagnostics
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

**Hyperparameters are identical across protocols and across phases** so any
difference in metrics is attributable to featurization alone. Random Forest
hyperparameters: `n_estimators=500, max_features='sqrt', min_samples_leaf=1`,
no tuning. Tuning is deferred to Phase 3 (XGBoost + Optuna).

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

1. **A graph-based model (ChemProp / D-MPNN) should mitigate this.** It learns
   the molecular representation end-to-end without privileging hand-crafted
   global descriptors, so it has no built-in "MW → logS" prior to misapply.
   This is the next phase of the project and the reason the project does not
   stop at the Phase 2 number.

2. **Aggregate RMSE is not enough.** Stratified error analysis (by chemical
   class, by target range) is essential for QSAR: a model with lower mean
   error can still be unsafe on specific structural families. This is a
   well-known issue in medicinal-chemistry deployment of QSAR models, and
   one of the reasons why simple "leaderboard" comparisons can be misleading.

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

5. **The right model class for the problem might not be the one that
   minimizes mean error.** A classical RF with global descriptors hits
   RMSE 1.06; published GNNs hit RMSE 0.55–0.70. But the sugar problem
   suggests RF with descriptors has a structural bias that no amount of
   tuning will fix. The case for moving to graph-based models is partly
   about lower error and partly about removing this bias.

---

## Next phases

- **Phase 3 — XGBoost + Optuna.** Same featurization as Phase 2, hyperparameter
  search to quantify the gap between RF (no tuning) and a tuned gradient
  boosting model. Expected: small additional gain, mostly in the moderate
  region; sugars still a problem.

- **Phase 4 — MLP on Morgan FP.** A small dense MLP, mostly to validate the
  PyTorch training loop and to provide a third reference point on identical
  data and identical splits.

- **Phase 5 — ChemProp (D-MPNN).** Graph-based message-passing neural network.
  The point is not just lower RMSE but **whether the sugar failure mode
  disappears**. If it does, that is the strongest argument for moving to
  graph models on chemically heterogeneous datasets.

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

---

## License & contact

Personal project, no specific license attached at this stage.
For questions: open an issue on the repository.
