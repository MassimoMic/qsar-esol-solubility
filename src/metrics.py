"""
Evaluation metrics for QSAR regression.

Single function `evaluate` matches the dict shape used in notebooks
01–03, so JSON summaries from this package and from the notebooks are
schema-compatible.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: Optional[str] = None,
    include_arrays: bool = True,
) -> Dict:
    """Compute RMSE / MAE / R² and optionally include the raw arrays.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions, same shape as y_true.
        split_name: Optional label ('train', 'valid', 'test') passed through
                    to the output dict — useful when assembling summary
                    tables across splits.
        include_arrays: If True, includes y_true and y_pred in the output
                        dict for downstream stratified-error analysis.
                        Set False when serializing many runs to JSON to
                        keep file size small.

    Returns:
        {
            'split': str | None,
            'RMSE':  float,
            'MAE':   float,
            'R2':    float,
            'y_true': np.ndarray   (if include_arrays),
            'y_pred': np.ndarray   (if include_arrays),
        }
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = {
        "split": split_name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
    if include_arrays:
        out["y_true"] = y_true
        out["y_pred"] = y_pred
    return out


def aggregate_seed_runs(per_seed_metrics: list[Dict]) -> Dict:
    """Aggregate per-seed metric dicts into mean ± std.

    Args:
        per_seed_metrics: list of dicts, each containing 'RMSE', 'MAE', 'R2'.

    Returns:
        {
            'RMSE_mean': ..., 'RMSE_std': ...,
            'MAE_mean':  ..., 'MAE_std':  ...,
            'R2_mean':   ..., 'R2_std':   ...,
            'n_seeds': int,
        }
    """
    rmses = np.array([r["RMSE"] for r in per_seed_metrics])
    maes = np.array([r["MAE"] for r in per_seed_metrics])
    r2s = np.array([r["R2"] for r in per_seed_metrics])
    return {
        "RMSE_mean": float(rmses.mean()),
        "RMSE_std": float(rmses.std()),
        "MAE_mean": float(maes.mean()),
        "MAE_std": float(maes.std()),
        "R2_mean": float(r2s.mean()),
        "R2_std": float(r2s.std()),
        "n_seeds": len(per_seed_metrics),
    }
