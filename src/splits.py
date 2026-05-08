"""
Scaffold-aware dataset splitting.

Two functions:
    scaffold_split_balanced : ChemProp-style train/valid/test split where
                              singleton scaffolds are distributed (not all
                              pushed to test). The recommended outer split.
    scaffold_kfold          : Scaffold-disjoint k-fold for inner CV during
                              hyperparameter tuning. Used when the outer
                              split is also scaffold-based.

Both are extracted verbatim from notebooks 02b and 03 to preserve numerical
identity with the published Phase 1/2/3 results.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(mol, include_chirality: bool = False) -> str:
    """Bemis–Murcko scaffold of a molecule as canonical SMILES.

    Args:
        mol: RDKit Mol object.
        include_chirality: If True, includes stereo information in the
            scaffold key. Default False, matching the notebooks.

    Returns:
        Canonical SMILES string of the scaffold (may be empty for acyclic
        molecules — RDKit returns '' in that case, which acts as its own
        equivalence class).
    """
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )


def scaffold_split_balanced(
    df: pd.DataFrame,
    mol_col: str = "mol",
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Balanced (ChemProp-style) Bemis–Murcko scaffold split.

    Strategy:
      1. Group molecules by scaffold.
      2. Sort scaffold groups of size > 1 by descending size; assign them
         to train first (largest first), so common scaffolds populate train.
      3. Shuffle singleton-scaffold groups with the given seed and append.
      4. Greedily fill train up to its target fraction, then valid, then
         test gets the remainder.

    This avoids the pure-scaffold-split pathology where every singleton
    scaffold ends up in test, producing degenerate metrics on small
    long-tailed datasets like ESOL.

    Args:
        df: DataFrame indexed 0..n-1 (caller's responsibility to reset_index
            beforehand if needed).
        mol_col: Column name holding RDKit Mol objects.
        train_frac, valid_frac, test_frac: Must sum to 1.0.
        seed: Controls the order in which singleton scaffolds are visited,
              and therefore the seed-to-seed variance of the split.

    Returns:
        (train_idx, valid_idx, test_idx): three lists of integer row indices
        into df. Sizes approximate the requested fractions but are not exact
        because scaffold groups are atomic (a group is never split across
        subsets).
    """
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-6

    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffolds[get_scaffold(row[mol_col])].append(idx)

    big_groups = [g for g in scaffolds.values() if len(g) > 1]
    singleton_groups = [g for g in scaffolds.values() if len(g) == 1]

    # Deterministic ordering of big groups: by size desc, ties broken by
    # first index for stability across pandas versions.
    big_groups.sort(key=lambda g: (len(g), g[0]), reverse=True)
    rng = random.Random(seed)
    rng.shuffle(singleton_groups)

    ordered_groups = big_groups + singleton_groups

    n_total = len(df)
    n_train_target = int(train_frac * n_total)
    n_valid_target = int(valid_frac * n_total)

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for group in ordered_groups:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid_target:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, valid_idx, test_idx


def scaffold_kfold(
    df_train: pd.DataFrame,
    mol_col: str = "mol",
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Scaffold-disjoint k-fold split for inner CV.

    Strategy (greedy bin-packing):
      1. Group all molecules in df_train by scaffold.
      2. Sort groups by descending size; shuffle ties with the given seed.
      3. Walk groups in order, assigning each to the currently-smallest fold.

    This keeps fold sizes comparable while guaranteeing disjoint scaffold
    sets across folds — the right inner-CV protocol when the outer split
    is also scaffold-based, otherwise hyperparameters can exploit scaffold
    similarity and over-fit.

    Args:
        df_train: Training subset DataFrame, expected to be reset_index'd.
        mol_col: Column with RDKit Mol objects.
        n_folds: Number of folds.
        seed: Tie-break shuffle seed.

    Returns:
        List of (cv_train_idx, cv_valid_idx) tuples. Indices are positional
        into df_train (i.e. 0..len(df_train)-1, *not* original df indices).
    """
    scaffolds = defaultdict(list)
    for pos, row in enumerate(df_train.itertuples(index=False)):
        mol = getattr(row, mol_col)
        scaffolds[get_scaffold(mol)].append(pos)

    groups = list(scaffolds.values())
    # Sort by size descending; rng-shuffle within size buckets for tie-break.
    rng = random.Random(seed)
    groups.sort(key=lambda g: len(g), reverse=True)

    # Re-shuffle within each size bucket so two runs with different seeds
    # produce different fold assignments among same-sized groups.
    by_size: dict[int, list] = defaultdict(list)
    for g in groups:
        by_size[len(g)].append(g)
    shuffled: List[List[int]] = []
    for size in sorted(by_size.keys(), reverse=True):
        bucket = by_size[size]
        rng.shuffle(bucket)
        shuffled.extend(bucket)

    fold_assignments: List[List[int]] = [[] for _ in range(n_folds)]
    for group in shuffled:
        # Assign group to the fold with the fewest molecules.
        smallest = min(range(n_folds), key=lambda k: len(fold_assignments[k]))
        fold_assignments[smallest].extend(group)

    # Build (train, valid) tuples for each fold.
    splits: List[Tuple[List[int], List[int]]] = []
    for k in range(n_folds):
        cv_valid = sorted(fold_assignments[k])
        cv_train = sorted(
            i for j in range(n_folds) if j != k for i in fold_assignments[j]
        )
        splits.append((cv_train, cv_valid))

    return splits
