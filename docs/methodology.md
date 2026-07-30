[← README](../README.md)

# Methodology

Dataset curation, evaluation protocols, and the design decisions behind them.

---

## Why ESOL

Aqueous solubility is the canonical "first" QSAR target: small (~1k molecules),
single regression target, well-studied, with strong physicochemical drivers
(LogP, MW, TPSA, H-bond donors). It is small enough to iterate quickly,
well-known enough to compare against published baselines, and structurally
diverse enough to expose the difference between random-split and scaffold-split
metrics.

This project uses ESOL as a **methodology testbed** before scaling to larger
ADMET datasets and target-specific virtual screening.

---

## Dataset curation

The raw ESOL release (1128 molecules, distributed via DeepChem) carries hidden
duplicates: SMILES strings that look different but produce the same RDKit
canonical SMILES. Naive use of the dataset would put structurally identical
molecules on both sides of any train/test split.

**Curation steps:**

1. Load via `dc.molnet.load_delaney(transformers=[])` to retain raw experimental
   logS (μ ≈ −3.05, σ ≈ 2.10, range [−11.6, +1.58]). The default DeepChem
   `NormalizationTransformer` is bypassed: it is the wrong default for QSAR work
   where target interpretability matters.
2. Canonicalize all SMILES with `Chem.MolToSmiles(Chem.MolFromSmiles(s))`.
3. Group by canonical SMILES. Found **11 duplicate groups**, sizes 2–3, spanning
   the same compound listed under variant SMILES (atom order differences,
   kekulization differences). Targets within a group were averaged.
4. One borderline case: hexitol (`OCC(O)C(O)C(O)C(O)CO`) appears with two logS
   values 1.03 apart. This is consistent with two stereoisomers indistinguishable
   by SMILES without explicit stereochemistry. Averaged like the others; flagged
   for revisit when working with stereo-annotated ChEMBL data.
5. **Final dataset: 1117 molecules**, saved to `data/processed/esol_dedup.csv`.

The scaffold split is **recomputed on the deduplicated dataset**, not reused from
DeepChem's original split (which was computed on the duplicated data). Skipping
this step would re-introduce a train/test leak on identical molecules.

> **Scope note.** Deduplication is one of four methodological choices that
> separate the numbers reported here from commonly published ESOL baselines; the
> others are split variant, seed count, and model hyperparameters. Their
> individual contributions have not been separated. See
> [Open questions](../README.md#open-questions).

---

## Evaluation protocols

**Three protocols** are reported, in increasing order of difficulty and realism.

### 1. Random split (single seed)

80/10/10 plain random partition. This is the optimistic upper bound; molecules in
test are structurally similar to those in train. Reported for reference and to
quantify the random-vs-scaffold gap.

### 2. Balanced scaffold split, single seed

Bemis–Murcko scaffold split with the **balanced (ChemProp-style) variant**: large
scaffold groups go to train (largest first); singleton scaffolds are shuffled
with a fixed seed and distributed across train/valid/test. The balanced variant
avoids a known pathology of pure scaffold splitting on small datasets like ESOL,
where all rare structurally-unusual molecules collapse into the test set,
producing degenerate metrics.

### 3. Balanced scaffold split, multi-seed (recommended)

Same as protocol 2, repeated over 5 seeds `[42, 0, 1, 2, 3]`, reported as
**mean ± std**. ESOL's structural distribution is long-tailed enough that a
single scaffold split has high seed-to-seed variance (RMSE range observed:
1.40–1.77 in Phase 1). Single-seed numbers should not be reported in isolation.

**Hyperparameters are kept identical across protocols within each phase**, so any
difference between protocols (random vs scaffold) is attributable to the split
alone. Across phases, what changes is either the featurization or the model.

Phase 1 / Phase 2 Random Forest hyperparameters: `n_estimators=500`,
`max_features='sqrt'`, `min_samples_leaf=1`, no tuning.

---

## Design decisions

**Target not normalized.** Default `dc.molnet.load_delaney` applies a
`NormalizationTransformer` that rescales logS to mean 0, std 1. This is bypassed
via `transformers=[]`, so RMSE/MAE stay physically interpretable in log units.
For PyTorch training in Phase 4 the target is standardized *internally* within
`train_mlp`, fitted on the training fold only, and predictions are
inverse-transformed before any metric is computed — a numerical stabilization
choice that does not affect the reported logS-space metrics.

**Scaffold split recomputed on dedup.** Reusing DeepChem's original split on the
deduplicated dataset would re-introduce identical molecules across train/test.
The split is computed from scratch on the 1117-molecule set.

**Balanced scaffold split, not pure.** Pure scaffold split puts every singleton
scaffold in test, which on ESOL produces a test set pathologically biased toward
large hydrophobic outliers (test R² collapsed to 0.18 in initial experiments).
The balanced variant preserves the rigor of scaffold-based generalization while
avoiding this degeneracy.

**Multi-seed reporting.** Single-run scaffold metrics on small datasets have high
variance. 5-seed mean ± std is the smallest acceptable unit.

**Paired Δ across seeds.** Phase comparisons are computed as paired per-seed
differences (`Δ_s = P2_s − P1_s` for each seed `s`, then mean ± std over seeds),
not as the difference of two independently-averaged numbers. This is the standard
paired-test design and reports the variance of the *improvement*, which is what
matters for assessing whether a change is a robust win.

**Median imputation, training-fold only.** Robust to noisy datasets even though
ESOL itself is clean. The pattern is preserved so the same evaluation harness can
be reused on ChEMBL without changes.

**Tune once, evaluate many (Phases 3–5).** Optuna selects hyperparameters on the
seed=42 outer split using scaffold-disjoint 5-fold CV on the training set only.
Those hyperparameters are then re-fit on each of the 5 outer scaffold seeds. We
are *not* re-tuning per seed: the 5-seed numbers test whether the chosen
hyperparameters generalize across splits. Re-tuning per seed would conflate
seed-dependent tuning variance with model-class variance.

**Scaffold-disjoint inner CV.** When the outer split is scaffold-aware, the inner
CV must be too — otherwise Optuna selects hyperparameters that exploit chemical
similarity within the training set and the chosen configuration ends up
over-confident on out-of-distribution scaffolds. The greedy bin-packing
implementation (`scaffold_kfold` in `src/splits.py`) keeps fold sizes balanced
while preserving scaffold disjointness.

**BatchNorm + variable batch size (Phase 4).** When the Optuna search space
includes both `batch_size` (categorical) and `use_batchnorm` (boolean), and the
training set size has `len % batch_size == 1` for some combination, `BatchNorm1d`
raises *Expected more than 1 value per channel when training* on the last batch.
Resolved in `src/training.py` by setting `drop_last=True` on the train DataLoader
**only when** the trailing batch would have size 1; otherwise `drop_last=False`
is preserved so the model sees the full training set.
