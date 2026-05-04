"""Featurization molecolare: Morgan fingerprint + descrittori RDKit 2D."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP-like fingerprint a lunghezza fissa.

    Radius=2 (≈ECFP4) è il default più usato in QSAR.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_descriptors(smiles: str) -> np.ndarray:
    """~210 descrittori 2D di RDKit (logP, MW, TPSA, conteggi, ecc.)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(Descriptors.descList), np.nan)
    return np.array([fn(mol) for _, fn in Descriptors.descList], dtype=float)


def descriptor_names() -> list[str]:
    return [name for name, _ in Descriptors.descList]


def featurize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    use_morgan: bool = True,
    use_descriptors: bool = True,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Combina Morgan FP e descrittori in una matrice X."""
    feats = []
    for smi in df[smiles_col]:
        parts = []
        if use_morgan:
            parts.append(morgan_fingerprint(smi, radius=radius, n_bits=n_bits))
        if use_descriptors:
            parts.append(rdkit_descriptors(smi))
        feats.append(np.concatenate(parts))
    X = np.vstack(feats)
    # Pulizia: descrittori RDKit possono dare inf/nan su molecole strane
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X
