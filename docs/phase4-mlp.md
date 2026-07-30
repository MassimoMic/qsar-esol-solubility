[← README](../README.md) · [← Phase 3](phase3-xgboost.md) · [Phase 5 →](phase5-chemprop.md)

# Phase 4 — MLP (PyTorch) on Morgan FP

**Goal.** Drop the RDKit descriptors and switch from a tree ensemble to a neural
network, keeping only the Morgan fingerprint as input. This isolates the model
expressivity axis on the *same* representation as Phase 1, and builds the PyTorch
infrastructure (training loop, early stopping, target standardization, Optuna
integration) reused by Phase 5.

The question was: *if the MLP without descriptors beats RF without descriptors,
part of Phase 1's error was about model expressivity, not features.* The MLP
**does** beat RF on aggregate RMSE (ΔRMSE = −0.155), but in doing so it reveals a
more interesting failure mode than Phase 3 showed: a strongly symmetric
compression bias on both tails of the solubility distribution.

> **⚠️ Confound to resolve.** The Phase 1 RF is **untuned**; the Phase 4 MLP is
> tuned with 100 Optuna trials. The −0.155 delta therefore conflates model class
> with tuning budget, and the expressivity conclusion below is not yet clean. A
> Morgan-only RF tuned on the same budget is outstanding work; see
> [Open questions](../README.md#open-questions).

---

## Setup

- **Model.** Configurable feed-forward MLP:
  `Input → [Linear → BatchNorm1d → ReLU → Dropout] × n_hidden → Linear → 1`.
  Hidden sizes, dropout rate and `use_batchnorm` are part of the search space.
- **Featurization.** Morgan FP radius=2, 2048 bits. **No descriptors**, by design.
- **Target standardization.** y standardized internally (train fold only);
  predictions inverse-transformed before any metric is computed.
- **Optimizer.** AdamW, `ReduceLROnPlateau` on valid RMSE (factor 0.5,
  patience 10, min_lr 1e-6).
- **Early stopping.** Patience 20 epochs on valid RMSE; best checkpoint restored.
- **Tuning.** 100 Optuna TPE + MedianPruner trials on scaffold-disjoint 5-fold CV
  from the seed=42 training set. Per-trial budget reduced to 100 epochs /
  patience 15 (~7 min total on a Colab T4); final 5-seed evaluation uses the full
  200 / 20 budget.
- **Search space (7 hyperparameters).** `n_hidden_layers ∈ {1,2,3}`,
  `hidden_dim_1 ∈ {128,256,512,1024}` (subsequent layers halve),
  `dropout ∈ [0.0, 0.5]`, `lr ∈ [10⁻⁵, 10⁻²]` (log),
  `weight_decay ∈ [10⁻⁷, 10⁻³]` (log), `batch_size ∈ {32,64,128,256}`,
  `use_batchnorm ∈ {True, False}`.

---

## Results

| Evaluation protocol | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Scaffold split (seed=42) | 1.399 | 1.108 | +0.456 |
| Scaffold split (5-seed mean) | **1.436 ± 0.102** | **1.133 ± 0.084** | **+0.466 ± 0.045** |

**Best hyperparameters** (CV-RMSE = 1.618):

```
hidden_dims:    (512, 256)
dropout:        0.320
lr:             0.003
weight_decay:   2.31e-5
batch_size:     32
use_batchnorm:  True
```

**Phase 1 → Phase 4 paired delta** (same featurization, different model class):

| Seed | P1 RMSE (RF, untuned) | P4 RMSE (MLP, tuned) | Δ RMSE |
|---|---|---|---|
| 42 | 1.618 | 1.399 | −0.219 |
| 0 | 1.395 | 1.293 | −0.102 |
| 1 | 1.526 | 1.445 | −0.081 |
| 2 | 1.773 | 1.609 | −0.164 |
| 3 | 1.641 | 1.433 | −0.208 |

**Mean ± std: ΔRMSE = −0.155 ± 0.055, P4 wins 5/5. ΔR² = +0.122 ± 0.046, 5/5.**

The paired std (0.055) is roughly **3× tighter** than the unpaired std would be
(`sqrt(0.126² + 0.102²) ≈ 0.162`), because most seed-to-seed variance is shared
between the two phases — hard scaffold splits are hard for both models. This is
the canonical illustration of why paired delta is the right comparison protocol
on small datasets.

---

## Optuna diagnostics

The 100-trial study completed in 6.9 minutes on a Colab T4 (3.3 s per trial
average; MedianPruner truncated unpromising trials early). Best CV-RMSE = 1.618,
found at trial 56.

The **CV-RMSE of 1.618 is markedly higher than the final 5-seed test RMSE of
1.436**. Expected: the CV folds use only ~80% of the outer training set, and
CV-RMSE averages over 5 scaffold-disjoint folds where at least one is typically
harder than the outer test split. The two are not directly comparable; what
matters is that the chosen hyperparameters **generalize** across the 5 outer
splits with low variance (std 0.102).

---

## Hyperparameter importance (FANOVA)

The most striking diagnostic of the phase:

| Hyperparameter | FANOVA importance |
|---|---|
| `use_batchnorm` | **0.731** |
| `lr` | 0.118 |
| `n_hidden_layers` | 0.041 |
| `weight_decay` | 0.034 |
| `hidden_dim_1` | 0.032 |
| `dropout` | 0.024 |
| `batch_size` | 0.015 |

**A single binary decision — BatchNorm on or off — accounts for 73% of the
CV-RMSE variance across 100 trials.** Everything else combined is under 27%.
Trials without BatchNorm cluster around CV-RMSE 1.8–2.0; trials with BatchNorm
cluster around 1.6 with limited further improvement available.

1. **BatchNorm is non-optional in this regime.** With 894 training molecules and
   2048-dimensional sparse features, per-layer normalization is what holds the
   gradient signal together. For neural QSAR work on similarly-sized datasets,
   BatchNorm should be defaulted on and only ablated for a specific reason.
2. **Once BatchNorm is on, the rest of the search space matters very little.** A
   50-trial run with `use_batchnorm=True` fixed would have reached essentially
   the same result.

This contrasts sharply with Phase 3, where `max_depth` and `reg_alpha` explained
~65% of the variance and at least four other hyperparameters contributed
meaningfully. **The leverage points of neural-net and tree-ensemble tuning are
fundamentally different**, and the FANOVA pattern is a more useful prior than
rules of thumb.

---

## Where Phase 4 really lives: the compression bias

Aggregate RMSE 1.436 hides the actual story. Stratified by logS regime, pooled
across 5 seeds (565 test predictions):

| Regime | n | Mean residual | MAE | RMSE |
|---|---|---|---|---|
| Hydrophilic (logS > −2) | 104 | **−0.674** | 1.050 | 1.298 |
| Moderate (−5 ≤ logS ≤ −2) | 335 | **+0.552** | 0.885 | 1.144 |
| Hydrophobic (logS < −5) | 126 | **+1.813** | 1.859 | 2.103 |

A **symmetric compression bias around the moderate regime**, much stronger than
Phase 3's:

- Hydrophilic molecules under-predicted (predicted less soluble than they are)
  by −0.67 on average.
- Hydrophobic molecules over-predicted by +1.81 on average.
- The moderate regime, holding 59% of test molecules, carries a smaller positive
  bias of +0.55.

Phase 3 showed hydrophilic −0.45 and hydrophobic +0.22 — both biases are
dramatically worse here. Aggregate RMSE only looks competitive because the
dominant moderate regime is where the bias is smallest.

---

## Worst-10 residuals: a chemotype audit

The ten distinct molecules with the largest mean absolute residual across the
pooled 5-seed test predictions. The pool contains 565 predictions over 194
distinct test molecules (mean 2.91 appearances per molecule; 174 of 194 appear in
≥2 seed test folds, 126 in ≥3), so most entries are tested against multiple
training subsets.

| # | Molecule | CAS | Class | y_true | y_pred | Residual | Seeds |
|---|---|---|---|---|---|---|---|
| 1 | **Coronene** (C₂₄H₁₂, 7 fused aromatic rings) | 191-07-1 | PAH | −9.33 | −4.24 | **+5.09** | 1/5 |
| 2 | **Mirex** (dodecachloropentacyclodecane) | 2385-85-5 | Organochlorine pesticide | −6.80 | −2.81 | +3.99 | 3/5 |
| 3 | **Etofenprox** (pyrethroid, C₂₅H₂₈O₃) | 80844-07-1 | Aromatic tri-ether | −8.60 | −4.64 | +3.96 | 3/5 |
| 4 | **Fluoranthene** (C₁₆H₁₀) | 206-44-0 | PAH | −8.49 | −5.13 | +3.36 | 2/5 |
| 5 | **C₂₇H₄₂O₃ steroidal sapogenin** (spirostan-3-ol, 414.6 Da; *not* diosgenin) | — | Steroid sapogenin | −7.32 | −3.97 | +3.35 | 2/5 |
| 6 | **Benzo[ghi]perylene** (C₂₂H₁₂) | 191-24-2 | PAH | −9.02 | −5.72 | +3.30 | 1/5 |
| 7 | **Diethylstilbestrol** (C₁₈H₂₀O₂) | 56-53-1 | Stilbene diphenol | −4.95 | −2.00 | +2.95 | 4/5 |
| 8 | **Piroxicam** (C₁₅H₁₃N₃O₄S) | 36322-90-4 | Sulfonamide drug | −4.16 | −1.24 | +2.92 | 4/5 |
| 9 | **Succinimide** (C₄H₅NO₂) | 123-56-8 | Cyclic imide | +0.30 | −2.60 | **−2.90** | 3/5 |
| 10 | **p,p'-DDE** (C₁₄H₈Cl₄) | 72-55-9 | Organochlorine | −6.90 | −4.05 | +2.85 | 4/5 |

### Identity verification methodology

Identities established by combining (1) RDKit canonical SMILES + InChIKey
computation, (2) SMARTS pattern matching against a curated chemotype catalogue
covering PAHs, organochlorines, steroids/triterpenes, sugars/polyols,
succinimides, sulfonamides, stilbenes, diaryl ethers, isoalloxazines and
nucleosides, and (3) PubChem InChIKey lookup.

Nine of ten entries are confirmed in PubChem with full IUPAC names and CAS
registry numbers. Entry #5 is identified to the **steroidal sapogenin
spirostan-3-ol class** (C₂₇H₄₂O₃, MolWt 414.6, skeleton hash `CPMZWWWXIWHTMU`) on
the basis of molecular formula, mass and structural inspection. A direct InChIKey
lookup against PubChem returns no match, and structural comparison against
canonical diosgenin (C₂₇H₄₂O₃, skeleton hash `DVOCDRFUYUFIIH`) shows divergent 2D
connectivity — so the ESOL compound is **a related sapogenin isomer, not
diosgenin itself**.

Authoritative table: `reports/phase4_worst10.json`. Structure grid:
`reports/figures/04_phase4_worst10.png`. Reproducing notebook:
`notebooks/07_phase4_worst10_analysis.ipynb`.

### Pattern: 9 hydrophobic + 1 hydrophilic, all compressed toward the mean

- **3 PAHs** (coronene, fluoranthene, benzo[ghi]perylene): heavy π-systems, true
  logS −8.5 to −9.3, predicted near −5. **The largest single error is coronene at
  +5.09 logS** — equivalent to predicting 60 µM solubility for a compound that is
  actually 0.5 nM soluble (a 10⁵ overestimate).
- **2 organochlorine pesticides** (mirex, DDE): true logS −6.8 to −6.9, predicted
  near −3.
- **1 pyrethroid** (etofenprox): true logS −8.6, predicted near −4.6.
- **1 steroidal sapogenin**: true logS −7.3, predicted near −4.0.
- **1 sulfonamide drug** (piroxicam) and **1 stilbene-diphenol** (DES): drug-like,
  smaller residuals.
- **1 hydrophilic small molecule** (succinimide): the only positive-logS entry.
  True +0.30, predicted −2.60 — pushed strongly *toward* hydrophobicity, the exact
  opposite direction of the others. Compression bias on the hydrophilic tail.

**Cross-seed consistency.** For the 8 worst-10 molecules appearing in ≥2 seeds,
the std of predictions across seeds is small (typically 0.15–0.45 logS) compared
to the residual itself (2.85–5.09). The failures are **structural to the
featurization, not stochastic to the training run** — different MLPs trained on
different subsets converge to the same wrong prediction.

### Mechanistic reading

Morgan FP is a *presence-of-pattern* encoding: one bit per substructure motif.
**The same bit fires for naphthalene (2 fused aromatic rings) and for coronene
(7).** The fingerprint captures qualitative substructure, not quantitative extent.
But solubility at the extremes is governed by quantity: the *number* of fused
rings, the *count* of chlorine atoms, the *length* and branching of an alkyl tail,
the *size* of a fused ring system. With access only to qualitative bits, the model
has no choice but to anchor predictions near the training mean for any molecule
outside the moderate regime.

---

## The polyol diagnostic — and why the sugar bias disappeared

The sugar bias of Phases 2/3 is genuinely gone, but was replaced by a different
polyol failure mode that the aggregate hydrophilic mean residual was hiding.

Screening the pooled test predictions for molecules with ≥3 free hydroxyl groups
(SMARTS `[OX2H]`, RDKit-validated, not a regex over SMILES):

- **43 polyol predictions across 14 unique molecules.**
- Mean residual: **+0.488** (positive — over-prediction of solubility).
- Median: **+0.715**.

The sign is **opposite** to Phase 2 (≈−0.9, sugars under-predicted) and Phase 3
(≈−0.45, same direction). Phase 4 *over*-predicts polyol solubility on average.

The five most badly mis-predicted polyols:

- **Riboflavin (vitamin B₂)** — true logS −3.69, predicted ≈−1.7 (residual +2.0).
  Four hydroxyls on a ribityl chain, attached to a voluminous pteridine aromatic
  scaffold.
- **Tubercidin/aristeromycin-like nucleoside** — −1.95, predicted −0.29
  (residual +1.66). Three hydroxyls on ribose, aromatic purine base.
- **Polyhydroxylated flavone** (luteolin-type) — −3.62, predicted −2.00
  (residual +1.62). Four phenolic OH on a tri-ring aromatic skeleton.

The five *most accurately* predicted polyols are, by contrast, **simple
saccharides and glycosides** — arbutin, sucrose, a chloral hydrate glycoside —
all with residuals under 0.15.

The MLP predicts canonical sugars well; what it cannot handle are **hybrids** —
molecules with multiple OH groups attached to an aromatic or fused-ring scaffold.
It has internalized an unsophisticated heuristic: *count hydroxyls, count aromatic
atoms, sum with appropriate sign*. This works on pure cases (glucose: only OH and
sp³ carbons → very soluble; a PAH: only aromatic carbons → very insoluble). It
breaks on molecules where the contributions cannot simply be summed — riboflavin
has the OH count of a tetraol but the polarity behaviour of an aromatic system.

The OH oxygen in glucose and the OH oxygen in luteolin are identical Morgan bits,
but they live in chemically completely different neighborhoods. A message-passing
GNN can encode that neighborhood directly in the atom embedding; Morgan FP cannot.
This is the motivation for [Phase 5](phase5-chemprop.md).

---

## Reading the model-class vs feature-class question, again

Phase 3 concluded the sugar bias was *partly* model-class and *partly*
feature-class. Phase 4 sharpens this:

- **The sugar bias of Phases 2/3 was largely model-class.** A different model
  class with completely different regularization mechanisms (dropout, BN, weight
  decay, AdamW) eliminated the under-prediction direction entirely.
- **The compression bias at the tails is largely feature-class.** It is *worse*
  in Phase 4 than in Phase 3 despite tuning to convergence, a different model
  class, and different regularization. It does not respond to model improvements
  on Morgan-only features.
- **The polyol-aromatic hybrid bias** is a Phase 4 refinement: a class-conditional
  failure where the learned heuristic is too simple. Also feature-class — Morgan
  cannot represent the chemical context of a substructure.

Phase 5 is framed accordingly: **does graph representation simultaneously reduce
both tails of the compression bias *and* resolve the polyol-aromatic case?**
Quantitative targets, set before running it:

- Hydrophilic mean residual: from −0.674 toward |0.10|.
- Hydrophobic mean residual: from +1.813 toward |0.50|.
- Polyol-aromatic mean residual: from ≈+0.49 toward |0.50|.
