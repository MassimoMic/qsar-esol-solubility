[← README](../README.md)

# Reproducing

```bash
git clone https://github.com/MassimoMic/qsar-esol-solubility.git
cd qsar-esol-solubility
pip install -r requirements.txt
```

Notebooks are self-contained and detect their environment. On Colab they mount
Google Drive at `MyDrive/qsar-esol-solubility/`; locally they resolve paths from
`Path.cwd().parent`. The same `.ipynb` file runs in both environments without
modification.

Run in order — each notebook consumes the output of the previous one.

---

## 1. `01_eda.ipynb`

Downloads ESOL via DeepChem, validates SMILES, deduplicates on canonical SMILES,
saves `data/processed/esol_dedup.csv` (1117 molecules). See
[Methodology → Dataset curation](methodology.md#dataset-curation).

## 2. `02_baseline_rf_morgan.ipynb`

Phase 1 baseline: Random Forest on Morgan FP. The persistence cell at the end
writes `reports/phase1_summary.json` in canonical schema.

## 3. `02b_baseline_rf_morgan_plus_desc.ipynb`

Phase 2: Random Forest on Morgan FP + RDKit 2D descriptors. Writes
`reports/phase2_summary.json`.

> **Outstanding:** the diagnostic cell in this notebook needs to persist
> `reports/phase2_stratified.json` (schema of `phase5_stratified.json`) so the
> per-regime tables in [Phases 1–2](phase1-2-baselines.md) and
> [Phase 3](phase3-xgboost.md) can be filled with exact values instead of
> approximations.

## 4. `03_xgboost_optuna.ipynb`

Phase 3 tuning. ~30 minutes on a Colab CPU runtime (100 Optuna trials × 5 CV
folds). Writes `models/xgb_esol_phase3_best_params.json` and
`reports/phase3_summary.json`.

## 5. `04_mlp_morgan.ipynb`

Phase 4 tuning. ~10 minutes on a Colab T4 GPU runtime (100 Optuna trials × 5 CV
folds ≈ 7 min, plus ~1 min for the 5-seed final evaluation). Writes
`models/mlp_esol_phase4_best_params.json` and `reports/phase4_summary.json`.
Requires `torch>=2.0`.

## 6. `05_chemprop_dmpnn_kaggle.ipynb`

Phase 5 ChemProp tuning. Designed for **Kaggle GPU runtime** — Colab Free quota is
insufficient for the full 30-trial × 5-fold study (~7–12 hours). Adapts paths and
uses SQLite-resumable Optuna storage to survive Kaggle's 9-hour session cap.
Writes `reports/phase5_summary.json` with inline per-seed test predictions (no
separate predictions file needed). Requires `chemprop>=2.0,<3.0`.

## 7. `06_paired_deltas.ipynb`

Reads all `reports/phase{N}_summary.json` files and emits cross-phase paired
ΔRMSE / ΔR² tables. Writes `reports/paired_deltas.md` (drop-in markdown) and
`reports/paired_deltas.json` (structured data). Runs in seconds.

## 8. `07_phase4_worst10_analysis.ipynb`

Re-fits the Phase 4 MLP on all 5 seeds to recover test predictions, deduplicates
by canonical SMILES, runs RDKit SMARTS-based chemical class detection, and
verifies molecular identities via PubChem InChIKey lookup. Writes
`reports/phase4_worst10.json` and `reports/figures/04_phase4_worst10.png`.

## 9. `08_phase5_stratified.ipynb`

Reads `phase5_summary.json` (no re-training needed, thanks to inline predictions),
computes regime and polyol stratified residuals, top-10 worst distinct molecules,
cross-phase worst-10 overlap with Phase 4, and emits the verdict against the three
stated targets. Writes `reports/phase5_stratified.json`.

---

## Determinism

All random seeds are fixed. Re-running a notebook end-to-end reproduces the
numbers in the [README](../README.md) to the third decimal on CPU; on GPU the
third decimal may move ±0.005 due to non-deterministic cuDNN atomic operations.
