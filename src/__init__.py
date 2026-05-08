"""
QSAR ESOL — shared infrastructure.

Modules:
    splits         : scaffold splitting (balanced, k-fold)
    featurization  : Morgan FP, RDKit descriptors, median imputation
    metrics        : RMSE / MAE / R² evaluation helpers
    training       : PyTorch MLP training + Optuna tuning + 5-seed evaluation
    reporting      : JSON summary persistence and paired-delta analysis

Notebooks 01/02/02b/03 currently keep their definitions inline for historical
reproducibility. From Phase 4 onward, notebooks should import from this package.
"""
