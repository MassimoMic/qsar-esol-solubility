"""
Featurization for ESOL QSAR.

Provides three feature representations used across phases:
    Morgan FP / ECFP4   : substructure-based, dense binary
    RDKit 2D descriptors: ~217 global molecular properties
    Combined            : [Morgan | descriptors], used in Phases 2 and 3

Plus median imputation helpers (fit on training fold only, no leakage).
On clean datasets like ESOL imputation is essentially a no-op, but the
pattern is kept here so the same pipeline transports to noisier datasets
like ChEMBL.

Extracted verbatim from notebook 02b. The descriptor list is computed at
import time using `Descriptors._descList` and depends on the RDKit version
in use; on RDKit 2024+, len(DESC_LIST) == 217.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors


# Compute once at import. The set of descriptors RDKit exposes is a
# function of the installed RDKit version; freezing it here makes the
# featurization deterministic for a given environment.
DESC_LIST = [(name, fn) for name, fn in Descriptors._descList]
DESC_NAMES = [name for name, _ in DESC_LIST]


# --- Morgan / ECFP4 fingerprint --------------------------------------------


def morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan / ECFP fingerprint as int8 NumPy array.

    Args:
        mol: RDKit Mol object.
        radius: Atom-environment radius. radius=2 → ECFP4.
        n_bits: Output bit-vector length.

    Returns:
        np.ndarray of shape (n_bits,), dtype int8, values in {0, 1}.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_morgan(
    df: pd.DataFrame, mol_col: str = "mol", radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """Stack Morgan fingerprints for all rows of a dataframe.

    Returns:
        np.ndarray of shape (len(df), n_bits), dtype int8.
    """
    return np.stack([morgan_fingerprint(m, radius, n_bits) for m in df[mol_col]])


# --- RDKit 2D descriptors --------------------------------------------------


def rdkit_descriptors(mol) -> np.ndarray:
    """Compute all RDKit 2D descriptors for a single molecule.

    Returns NaN for descriptor functions that fail or produce non-finite
    values, to be handled downstream by the median imputer.

    Returns:
        np.ndarray of shape (len(DESC_LIST),), dtype float64.
    """
    out = np.full(len(DESC_LIST), np.nan, dtype=np.float64)
    if mol is None:
        return out
    for j, (_, fn) in enumerate(DESC_LIST):
        try:
            v = fn(mol)
            if not np.isfinite(v):
                v = np.nan
            out[j] = v
        except Exception:
            out[j] = np.nan
    return out


def featurize_rdkit_descriptors(
    df: pd.DataFrame, mol_col: str = "mol"
) -> np.ndarray:
    """Stack RDKit 2D descriptors for all rows of a dataframe."""
    return np.stack([rdkit_descriptors(m) for m in df[mol_col]])


def featurize_combined(
    df: pd.DataFrame,
    mol_col: str = "mol",
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Concatenated [Morgan FP | RDKit 2D descriptors] feature matrix.

    Phase 2 and Phase 3 featurization. Output dtype is float64 (descriptors
    dominate the type promotion); fingerprint columns are 0.0 / 1.0.
    """
    fp = featurize_morgan(df, mol_col, radius, n_bits)
    desc = featurize_rdkit_descriptors(df, mol_col)
    return np.hstack([fp, desc]).astype(np.float64)


# --- Median imputation -----------------------------------------------------


def fit_median_imputer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a median imputer on the training fold.

    Columns that are entirely NaN in training are dropped — they carry no
    information that a median can recover. Other columns get the column-wise
    median (over non-NaN values) as their fill value.

    Args:
        X_train: Training feature matrix, may contain NaN/inf.

    Returns:
        (medians, keep_mask):
            medians : np.ndarray of column medians for kept columns
            keep_mask: boolean array of length X_train.shape[1] indicating
                       which columns are retained.
    """
    col_all_nan = np.isnan(X_train).all(axis=0)
    keep = ~col_all_nan
    medians = np.nanmedian(X_train[:, keep], axis=0)
    return medians, keep


def apply_imputer(
    X: np.ndarray, medians: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """Apply a fitted median imputer to a feature matrix.

    Drops the columns excluded by `keep`, then fills any remaining NaN with
    the corresponding column median. inf values are preserved (caller should
    have handled them upstream); only NaN is filled.
    """
    Xk = X[:, keep]
    nan_mask = np.isnan(Xk)
    if nan_mask.any():
        Xk = Xk.copy()
        Xk[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    return Xk
