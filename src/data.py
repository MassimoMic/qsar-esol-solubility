"""Data loading e splitting per il dataset ESOL.

Lo scaffold split divide le molecole in base al loro scaffold di Bemis-Murcko.
Questo simula meglio lo scenario reale: il modello viene valutato su scheletri
chimici mai visti in training, evitando R² gonfiati dal random split.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ESOL_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/delaney-processed.csv"
)


def load_esol(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Carica ESOL, scaricandolo se non presente in locale.

    Returns
    -------
    DataFrame con colonne ['smiles', 'y'] dove y = log solubility (mol/L).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    local = data_dir / "delaney-processed.csv"

    if not local.exists():
        df = pd.read_csv(ESOL_URL)
        df.to_csv(local, index=False)
    else:
        df = pd.read_csv(local)

    df = df.rename(
        columns={"smiles": "smiles", "measured log solubility in mols per litre": "y"}
    )
    return df[["smiles", "y"]].reset_index(drop=True)


def _scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold come SMILES canonico."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split scaffold-based: scaffold più frequenti -> train, rari -> test.

    Returns
    -------
    Tuple di indici (train_idx, valid_idx, test_idx).
    """
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-6

    scaffolds: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(df[smiles_col]):
        scaffolds[_scaffold(smi)].append(i)

    # Ordina i set di scaffold per dimensione decrescente: i big bucket
    # vanno in train, i piccoli (più "rari") in valid/test.
    scaffold_sets = sorted(scaffolds.values(), key=lambda s: (len(s), s[0]), reverse=True)

    n = len(df)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)

    train_idx, valid_idx, test_idx = [], [], []
    for sset in scaffold_sets:
        if len(train_idx) + len(sset) <= n_train:
            train_idx.extend(sset)
        elif len(valid_idx) + len(sset) <= n_valid:
            valid_idx.extend(sset)
        else:
            test_idx.extend(sset)

    rng = np.random.default_rng(seed)
    return (
        rng.permutation(train_idx),
        rng.permutation(valid_idx),
        rng.permutation(test_idx),
    )
