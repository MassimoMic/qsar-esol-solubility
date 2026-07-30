[← README](../README.md) · [Methodology](methodology.md)

# Lessons learned

Thirteen items, including the ones that cost time.

---

**1. Dataset deduplication matters even on a "clean" public dataset.** ESOL is a
20-year-old, widely-cited benchmark and still ships with 11 hidden duplicate
groups. The first published RF benchmarks were computed on duplicated data with
single-seed scaffold splits. (How much of the reported-vs-honest gap this
actually accounts for is a separate question, and one this project has not yet
answered — see [Open questions](../README.md#open-questions).)

**2. The right baseline is hard.** Reporting RMSE 1.05 on ESOL for "RF on Morgan
FP, scaffold split" matches the literature — but only on a single seed and on
duplicated data. The 5-seed deduplicated number is 1.59. The point is not that
the literature is wrong; it is that "matching the benchmark" requires reproducing
every methodological choice, including the questionable ones.

**3. Feature engineering can move the average and break the tails.** Phase 2
improves average RMSE by 33% but creates a new failure mode on sugars. Without
stratified analysis this would be invisible — and dangerous if the model were
deployed to predict solubility for carbohydrate chemistry.

**4. Multi-seed reporting is non-negotiable on small datasets.** Single-seed
scaffold metrics on ESOL vary by ±0.2 RMSE across reasonable seeds. Any
conclusion drawn from a single run is not reproducible.

**5. The right model class might not be the one that minimizes mean error.**
Tuned XGBoost on Phase 2 features hits RMSE 0.86 — closing more than half the gap
to published GNN numbers (0.55–0.70). But the sugar bias is only halved, and a
new compression bias on hydrophobic outliers appears as a regularization
side-effect. Phase 4 sharpens the point on the opposite side: a tuned MLP on
Morgan-only features beats RF on aggregate RMSE while developing a much
*stronger* symmetric compression bias on both tails. Aggregate metrics and
model-class improvements can move in opposite directions on the parts of the
distribution that matter.

**6. Most tuning variance came from very few knobs — but which knobs depends on
the model class.** On Phase 3 (XGBoost), FANOVA attributes ~65% of CV-RMSE
variance to `max_depth` and `reg_alpha`. On Phase 4 (MLP), FANOVA attributes
**73% to a single binary**: `use_batchnorm`. Translating between architectures is
not automatic: tree-ensemble tuning is about capacity control and L1;
neural-net tuning on small datasets is about whether the gradient signal can
stabilize at all.

**7. Strong regularization buys mean-error reduction at the cost of tail biases.**
Phase 3 reduces RMSE in the moderate logS region (where most of the dataset sits)
by compressing predictions toward the training mean. This is mathematically how
L1-regularized boosting wins on aggregate; the price is paid in small but real
biases at the distributional extremes. For QSAR deployment this matters: the
molecules a project actually cares about predicting often *are* the extremes.

**8. Compression bias is a property of the featurization, not just of the model —
and the literature agrees.** Phase 3 showed mild compression as a side-effect of
`reg_alpha`. Phase 4 — completely different model class, completely different
regularization mechanism — shows *worse* compression on both tails. The bias is
not what the model does to fit; it is what the features cannot encode. Morgan FP
captures qualitative substructure presence, not quantitative extent (ring count,
halogen count, alkyl chain length), and the extremes of the logS distribution are
precisely governed by quantity.

Phase 5 refines this further. ChemProp's learned graph representation reduces the
*directional* compression bias at both tails by 70–80%, but the *magnitude* gap
against Phase 3 is of the same size as the gap P4 → P5 closes. The bias
decomposes into two coupled components: a directional one that is partially
expressivity-class (resolvable by better architecture) and a magnitude one that is
feature-class (requires global descriptors). The polyol-aromatic case responds to
neither, suggesting a 3D/solvation-class problem out of reach of 2D
representations entirely.

This is **consistent with independent benchmarks on small molecular datasets**.
Jiang et al. (2021) found descriptor-based models best on 6 of 11 MoleculeNet
datasets — ESOL among them — occupying 73% of top-3 ranks overall. Notwell & Wood
(2023) reach the same conclusion across the 22 TDC ADMET benchmarks, with the
additional observation that adding a GNN-derived fingerprint to a descriptor +
fingerprint baseline *further improves* performance: the representations carry
complementary, not redundant, information.

**9. Stratified analysis must be paired with targeted sub-class diagnostics.** The
Phase 4 aggregate hydrophilic residual of −0.67 looked like a continuation of the
Phase 2/3 sugar bias. A targeted polyol screen (≥3 free OH via SMARTS) revealed
that the bias on canonical sugars had inverted in sign, and that the residual
hydrophilic error was concentrated on a different sub-class: polyol-aromatic
hybrids (riboflavin, polyhydroxyflavones, nucleosides). A single stratified
pattern can hide two failure modes that partially cancel in the average. Whenever
a stratified analysis reveals a directional bias, validate it with at least one
chemically-defined sub-class query.

**10. Persist per-seed results to JSON after every notebook.** The pattern *"each
notebook writes a summary JSON in `reports/`"* makes paired deltas between phases
automatic and eliminates the need to re-run upstream notebooks to recover lost
numbers. This project learned the cost of the alternative the hard way: the
Phase 1 → Phase 4 comparison originally fell back on an unpaired delta because no
`phase1_summary.json` had been persisted, and a dedicated housekeeping pass was
needed before `06_paired_deltas.ipynb` could compute all cross-phase paired deltas
in one shot. Cheap to write upstream; expensive to recover downstream. From
Phase 4 onward every notebook persists its own canonical summary; Phases 1–3 were
retrofitted to match.

**11. Wrapper functions need to be tested with non-default parameters before being
handed to Optuna.** Phase 5 lost ~3 hours to a wrapper bug that only manifested
for `mp_hidden_dim ≠ 300`: the predictor FFN's `input_dim` was not propagated from
the message-passing layer's `d_h`, producing `RuntimeError: mat1 and mat2 shapes
cannot be multiplied (50x200 and 300x300)` on every trial whose suggested hidden
dimension was not the default. The sanity check ran with default parameters and
passed; the bug only surfaced once Optuna started exploring. **Rule:** any wrapper
intended for hyperparameter search must be tested explicitly with parameter values
that *differ* from the defaults along every dimension before its first Optuna call.

**12. ChemProp v2 + Lightning have several non-obvious API frictions worth
recording.** Three issues consumed an evening during Phase 5 setup:

- `pl.seed_everything(seed, workers=True)` installs a DataLoader reinstantiator
  that conflicts with ChemProp v2's internal sampler argument (`TypeError: got
  multiple values for argument 'sampler'`). Fix: use a plain
  `torch.manual_seed` + numpy + python `random` seeder
  (`src.training.seed_everything`) instead.
- `TrainingBatch` objects from ChemProp v2's DataLoader have no `.to(device)`
  method. Use `lightning.pytorch.Trainer.predict()` for inference instead of
  manual loader iteration; it handles device placement internally.
- `Draw.MolsToGridImage(useSVG=False)` returns a `PIL.Image`, a `bytes` object or
  an `IPython.core.display.Image` depending on the installed RDKit version. Use
  duck typing (`hasattr(img, 'save')`, `hasattr(img, 'data')`,
  `isinstance(img, bytes)`) to handle all three.

These are not bugs to be fixed in the libraries; they are rough edges to be
navigated. Worth keeping a personal list for the next graph-based project.

**13. Compute infrastructure choices are part of methodology.** The Phase 5 Optuna
study (30 trials × 5-fold CV × ~3 min/fit ≈ 7.5 hours) exceeded Colab Free's daily
GPU quota mid-run. The recovery path was migrating to Kaggle (30 GPU-hours/week,
background execution with email notification, 9-hour session cap) with
**SQLite-based Optuna storage** for resumability across sessions. This is not just
an operational note: it changes what kind of experiments are feasible in what kind
of week. For larger follow-on work, compute will be the binding constraint rather
than ideation, and the default should be Kaggle for heavy training and Colab for
interactive debugging.
