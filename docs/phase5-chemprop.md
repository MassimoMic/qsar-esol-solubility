[← README](../README.md) · [← Phase 4](phase4-mlp.md) · [Lessons learned →](lessons-learned.md)

# Phase 5 — ChemProp D-MPNN (graph representation)

Phase 5 swaps the feature-engineering side completely: instead of computing
Morgan fingerprints and optional RDKit descriptors and feeding them into a flat
model, it hands the *molecular graph* to a directed message-passing neural
network that learns its own representation end-to-end. No hand-crafted
descriptors; the input is SMILES plus the per-atom and per-bond features ChemProp
computes internally (atomic number, degree, formal charge, hybridization,
aromaticity, chirality, bond type, ring membership, stereo).

This phase was framed in advance as "the resolution" of the Phase 4 compression
bias. Reality is more nuanced.

---

## Setup

- **Framework.** ChemProp v2.2.3 + PyTorch Lightning 2.6 + Optuna 3.5
- **Tuning.** 30 Optuna trials × 5-fold scaffold-disjoint CV on the seed=42
  training set (same protocol as Phases 3 and 4)
- **Sampler.** TPE. **Pruner.** MedianPruner (n_startup=5, n_warmup_steps=2)
- **Search space (6 hyperparameters).** `depth ∈ {2,3,4,5}`,
  `mp_hidden_dim ∈ {200,300,400,500}`, `mp_dropout ∈ [0, 0.4]`,
  `ffn_hidden_dim ∈ {200,300,500,700}`, `ffn_num_layers ∈ {1,2,3}`,
  `max_lr ∈ [3e-4, 3e-3]` (log)
- **Fixed.** batch_size=50, init_lr=final_lr=1e-4, warmup_epochs=2,
  max_epochs=150, patience=30
- **Final evaluation.** 5-seed scaffold outer with best params (~15 min, Kaggle T4)
- **Compute.** Optuna study ~45 min on Kaggle T4 (background execution, SQLite
  resume); 5-seed outer ~3 min

The compute infrastructure required a switch from Colab Free (GPU quota exhausted
mid-study) to Kaggle. Documented in [Lessons learned](lessons-learned.md).

---

## Results

| Evaluation protocol | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Scaffold split (seed=42) | 1.159 | 0.882 | +0.661 |
| Scaffold split (5-seed mean) | **1.148 ± 0.030** | **0.882 ± 0.021** | **+0.656 ± 0.036** |

The aggregate sits between Phase 4 (1.436) and Phase 2 (1.062). It beats Phase 4
by a clear margin but **does not match** the hand-engineered tabular baselines:

| Comparison vs Phase 5 | ΔRMSE (paired) | Direction | Interpretation |
|---|---|---|---|
| vs P4 (MLP Morgan) | −0.287 ± 0.098 | P5 wins 5/5 | graph beats fingerprint |
| vs P3 (XGB Morgan+desc) | +0.287 ± 0.042 | P3 wins 5/5 | tabular beats graph |
| vs P2 (RF Morgan+desc) | +0.087 ± 0.070 | P2 wins 4/5 | even untuned RF beats graph |
| vs P1 (RF Morgan) | −0.442 ± 0.110 | P5 wins 5/5 | full trajectory ≈28% RMSE |

**Numerical symmetry.** The two key deltas are **indistinguishable in magnitude**
(−0.287 ± 0.098 and +0.287 ± 0.042). Switching Morgan → graph gains
approximately what dropping global descriptors costs at fixed model class. Given
the error bars, the equality to three decimals should be read as coincidence
rather than law; what is defensible is that the two effects are of the same size
on a dataset of this scale. With a larger dataset, the graph would likely pull
ahead. This is the cleanest empirical statement of the feature-class vs
model-class question the project has been chasing.

---

## Context in the literature

That a tuned D-MPNN does not beat a tuned tabular model with global descriptors
on ~1k molecules is **consistent with independent benchmarking**, not an
idiosyncrasy of this project:

- **Jiang et al. (2021), *J. Cheminform.* 13:12** — systematic comparison of four
  descriptor-based models (SVM, XGBoost, RF, DNN) against four graph-based models
  (GCN, GAT, MPNN, Attentive FP) on 11 MoleculeNet datasets with multiple random
  splits. Descriptor-based models occupy the top three ranks on **24 of 33
  dataset×rank slots (73%)** and produce the best model on 6 of 11 datasets —
  **ESOL among them**. SVM gives the best test RMSE on ESOL (≈0.57 on random
  split). Graph models pull ahead only on larger or multi-task datasets.

- **Notwell & Wood (2023), arXiv:2310.00174** — on the 22 TDC ADMET benchmarks,
  gradient-boosted decision trees (CatBoost) with ECFP + Avalon + ErG
  fingerprints plus 200 molecular descriptors consistently match or outperform
  recently published GNN methods. Adding a GNN-derived fingerprint as additional
  features improves results further — a sign that graph and descriptor features
  carry **complementary** rather than redundant information.

- **Broccatelli et al. (2021), arXiv:2111.13964** — Genentech/Roche internal ADME
  datasets, four GNN variants benchmarked against tabular baselines with
  whole-molecule descriptors. All GNNs beat the fingerprints-only baseline; only
  GAT shows a small consistent improvement over the descriptor-augmented
  baseline. MPNN — the architecture closest to ChemProp D-MPNN — does **not**
  exceed the descriptor-augmented tabular model.

Phase 5 falls precisely in line: on a ~1k-molecule single-task regression target
where global physicochemical descriptors are highly predictive of the endpoint
(LogP/TPSA for solubility), 2D graph representations recover most but not all of
the information descriptors carry. The 0.29 RMSE gap between P3 and P5 is the
*quantitative measure* of that residual descriptor advantage on this dataset.

The regime in which graph representations are expected to outperform — larger
datasets (>10k molecules), multi-task setups, endpoints less directly tied to
bulk physicochemical properties — makes ESOL Phase 5 a calibrated baseline rather
than a negative result.

*For context on the random-split comparison: this project's Phase 3 reaches
RMSE 0.550 on random split, marginally better than the ≈0.57 SVM figure Jiang et
al. report. Random-split numbers are deprecated throughout this project and the
comparison is offered only for orientation.*

---

## Cross-seed stability

Phase 5's std of **0.030** RMSE is the smallest in the project — comparable to
Phase 3 (0.026), tighter than P4 (0.102), P2 (0.053) or P1 (0.126). ChemProp's
failures are highly *deterministic*: models trained on different scaffold splits
converge to nearly identical predictions for the same molecules, which is exactly
the signature expected when the ceiling is set by the representation rather than
by training stochasticity.

---

## Stratified residual analysis

Phase 5 was assessed against three targets stated before the phase was run, with
an explicit decision rule.

| Pattern | Phase 4 baseline | Phase 5 result | Target | Verdict |
|---|---|---|---|---|
| hydrophilic mean residual | −0.674 | **+0.143** | ≤ \|0.10\| | ✗ (off by 0.04) |
| hydrophobic mean residual | +1.813 | **+0.517** | ≤ \|0.50\| | ✗ (off by 0.02) |
| polyol mean residual | +0.488 | **+0.481** | ≤ \|0.50\| | ✓ (essentially unchanged) |

**Two patterns moved a lot. One did not.**

- **Hydrophilic bias** was reduced **79% in magnitude** and inverted in sign: P4
  systematically under-predicted hydrophiles by 0.67 logS; P5 over-predicts them
  by 0.14, just missing the |0.10| target. A real architectural improvement.

- **Hydrophobic bias** was reduced **71% in magnitude**, from +1.81 to +0.52,
  just missing the |0.50| target. Phase 4's worst-10 contained coronene at
  +5.09 logS (a 10⁵ overestimate of solubility); the same molecules in Phase 5
  still have residuals around +2 to +3 — large, but no longer pathological.

- **Polyol bias** is **essentially unchanged**: +0.488 → +0.481. The |0.50|
  target is formally hit by accident — the bias did not improve, it was already
  at the threshold. This is the most informative null result of the phase.

**Caveat on n.** The extreme regimes carry few molecules. Bootstrap intervals on
the regime-level means are outstanding work; see
[Open questions](../README.md#open-questions).

---

## Interpretation: the bias splits into two components

Phase 4's finding was *"compression bias at the extremes is feature-class, not
model-class"*. Phase 5 refines this into three statements:

1. **A directional (sign-related) component is expressivity-class.** The learned
   graph representation, with no help from global descriptors, reduces both tails
   by ~70–80% in magnitude. The model class matters.

2. **A magnitude (variance-recovery) component is feature-class.** Even after the
   architectural improvement, average RMSE remains worse than tabular baselines
   with access to global descriptors. The 0.29 RMSE gap vs P3 is the
   *quantitative price* of not having LogP, TPSA and MW directly available. The
   2D molecular graph alone, however cleverly aggregated, does not reconstruct
   global physicochemical quantities from atom-level patterns at this dataset
   size.

3. **The polyol-aromatic case is neither.** It does not move with model class
   (Phase 5 did not fix it), and Phases 2/3 did not exhibit it at all — suggesting
   it is neither model-class nor 2D-feature-class. The most plausible reading is
   that it is **3D/solvation-class**: discriminating riboflavin-like
   polyol-aromatic hybrids from glucose-like sugars requires the 3D arrangement of
   hydroxyls relative to the aromatic system and the solvation shell they form.
   2D graph representations (Morgan, ChemProp) and 2D descriptors (RDKit) all
   miss this. Resolving it would require explicit 3D conformer ensembles (SchNet,
   PaiNN, equivariant GNNs) or solubility-specific physics such as SMD solvation
   energies as auxiliary features.

---

## Top-10 worst residuals: how the failure mode evolved

The Phase 5 top-10 (pooled 5-seed, deduplicated by canonical SMILES) overlaps
substantially with Phase 4's but is less extreme: the worst residual is ~+3.5
(still a hyper-hydrophobic PAH) against Phase 4's +5.09 on coronene. The same
chemotypes recur — PAHs, organochlorines, lipophilic ethers, the C₂₇H₄₂O₃
steroidal sapogenin, DDE — with consistently smaller magnitude. The model has
learned to push hyper-hydrophobes less extremely toward the training mean, but it
still pushes them inward.

Per-seed test predictions: `reports/phase5_summary.json`. Regime/polyol
breakdowns and the top-10 distinct list: `reports/phase5_stratified.json`.
Reproducing code: `notebooks/08_phase5_stratified.ipynb`.

---

## Verdict against the pre-registered decision rule

The rule stated before running the phase was: *if ChemProp hits all three targets
(hydrophilic, hydrophobic, polyol), graph representation is empirically validated
as the right next step on logS QSAR.*

Phase 5 hits one of three formally and narrowly misses two. **Under the strict
rule the verdict is WEAK**: graph representation alone does not resolve the
compression bias.

Under a more nuanced reading the verdict is **PARTIAL**: the bias has two
components, direction and magnitude, and architecture resolves the first but not
the second. This is a scientifically richer finding than "ChemProp wins" or
"ChemProp fails" — it makes explicit the *kind* of information each featurization
carries, and identifies the polyol-aromatic case as a genuinely 3D problem.

> **On the word "pre-registered."** The targets and decision rule were fixed in
> the project's working notes before Phase 5 was run, but were not committed to
> the repository with a verifiable timestamp beforehand. Readers cannot
> independently confirm the ordering. Future phases will commit the target file
> as a dated artifact prior to running.
