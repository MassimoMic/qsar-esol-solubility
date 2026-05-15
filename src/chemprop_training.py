"""
ChemProp v2 D-MPNN training wrapper for Phase 5.

Mirrors the contract of `src.training.train_mlp` so that the 5-seed
evaluation orchestrator (`evaluate_5seed_scaffold` in src/training.py) can
plug ChemProp in by name, without changes to the orchestrator itself.

Key differences from `train_mlp`:
  - Input is SMILES (list of strings), not a feature matrix.
  - Training uses ChemProp's MoleculeDataset/featurizer pipeline and a
    Lightning Trainer under the hood.
  - Target standardization is delegated to ChemProp's
    `MoleculeDataset.normalize_targets` (StandardScaler internally).
  - Inference returns predictions in original logS units (the predictor's
    UnscaleTransform unwraps automatically).

The wrapper returns the same dict schema as `train_mlp`, so downstream
code is identical:
    {
        'predict':         callable(smiles_list) -> np.ndarray (logS units),
        'best_valid_rmse': float (in logS space),
        'history':         [{'epoch','train_loss','valid_rmse'}, ...],
        'stopped_at':      int,
        'config':          dict of architecture hyperparams,
    }

Tested with: chemprop==2.2.3, pytorch-lightning==2.6.1, torch==2.10.0+cu128.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Lazy import of ChemProp + Lightning so module import doesn't fail if
# someone loads src/ before installing chemprop. The training functions
# import these at call-time.


def _import_chemprop():
    """Lazy import: raise a clear error if chemprop isn't installed."""
    try:
        from chemprop import data, featurizers, models, nn
        from lightning import pytorch as pl
    except ImportError as e:
        raise ImportError(
            "ChemProp v2 and PyTorch Lightning are required for "
            "src.chemprop_training. Install with: "
            "pip install 'chemprop>=2.0,<3.0'"
        ) from e
    return data, featurizers, models, nn, pl


def _build_datapoints(smiles: Sequence[str], y: np.ndarray):
    """Build a list of ChemProp MoleculeDatapoint from SMILES + targets."""
    data, _, _, _, _ = _import_chemprop()
    return [
        data.MoleculeDatapoint.from_smi(smi, np.array([float(target)]))
        for smi, target in zip(smiles, y)
    ]


def train_chemprop(
    smiles_train: Sequence[str],
    y_train: np.ndarray,
    smiles_valid: Sequence[str],
    y_valid: np.ndarray,
    *,
    # --- D-MPNN architecture hyperparams ---
    depth: int = 3,
    mp_hidden_dim: int = 300,
    mp_dropout: float = 0.0,
    # --- Output FFN hyperparams ---
    ffn_hidden_dim: int = 300,
    ffn_num_layers: int = 1,
    ffn_dropout: float = 0.0,
    # --- Optimization hyperparams ---
    max_lr: float = 1e-3,
    init_lr: float = 1e-4,
    final_lr: float = 1e-4,
    warmup_epochs: int = 2,
    batch_size: int = 50,
    # --- Training schedule ---
    max_epochs: int = 150,
    patience: int = 30,
    # --- Reproducibility ---
    seed: int = 42,
    verbose: bool = False,
    # --- Trainer overrides (rare) ---
    accelerator: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict:
    """Train one D-MPNN until early stopping on validation loss.

    Args:
        smiles_train, y_train: training SMILES strings + logS targets.
        smiles_valid, y_valid: validation SMILES + targets.
        depth: number of message-passing iterations.
        mp_hidden_dim: hidden dim of the bond-message passing layer.
        mp_dropout: dropout in the message passing.
        ffn_hidden_dim: hidden dim of the readout FFN.
        ffn_num_layers: depth of the readout FFN (1 = linear head).
        ffn_dropout: dropout in the readout FFN.
        max_lr, init_lr, final_lr: Noam schedule LR boundaries.
        warmup_epochs: linear warmup duration (in epochs).
        batch_size: training batch size.
        max_epochs: training cap.
        patience: epochs without val_loss improvement before stopping.
        seed: torch + python + numpy seed.

    Returns:
        Dict with 'predict' (callable), 'best_valid_rmse' (in logS units),
        'history', 'stopped_at', 'config'.
    """
    data, featurizers, models, nn, pl = _import_chemprop()

    # --- 1. Seed everything ---
    # pl.seed_everything(seed, workers=True)

    # Lightning's `workers=True` flag installs a DataLoader wrapper that
    # clashes with ChemProp v2's internal sampler argument; use the standard
    # seed_everything from src.training (torch + numpy + python only).
    from src.training import seed_everything
    seed_everything(seed)

    # --- 2. Build datapoints + datasets ---
    train_dpoints = _build_datapoints(smiles_train, y_train)
    valid_dpoints = _build_datapoints(smiles_valid, y_valid)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_dpoints, featurizer)
    valid_dset = data.MoleculeDataset(valid_dpoints, featurizer)

    # Target scaling: fit on train, propagate to valid
    scaler = train_dset.normalize_targets()
    valid_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(
        train_dset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    valid_loader = data.build_dataloader(
        valid_dset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    # --- 3. Build the D-MPNN model ---
    mp = nn.BondMessagePassing(
        depth=depth,
        d_h=mp_hidden_dim,
        dropout=mp_dropout,
    )
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    predictor = nn.RegressionFFN(
        input_dim=mp_hidden_dim,        # ← FIX: propaga d_h da MPNN
        n_layers=ffn_num_layers,
        hidden_dim=ffn_hidden_dim,
        dropout=ffn_dropout,
        output_transform=output_transform,
    )
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    mpnn = models.MPNN(
        mp,
        agg,
        predictor,
        batch_norm=True,
        metrics=metric_list,
        warmup_epochs=warmup_epochs,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )

    # --- 4. Trainer ---
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=[early_stop],
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=log_dir or "/tmp/chemprop_lightning",
    )

    trainer.fit(mpnn, train_loader, valid_loader)

    # --- 5. Compute val RMSE in original logS units ---
    val_predictions = trainer.predict(mpnn, valid_loader)
    val_preds = torch.cat(val_predictions, dim=0).cpu().numpy().flatten()
    val_y_orig = np.asarray(y_valid).astype(np.float64)
    val_rmse_logS = float(np.sqrt(np.mean((val_preds - val_y_orig) ** 2)))

    # --- 6. Build closure predictor for new SMILES ---
    def predict(smiles_in: Sequence[str]) -> np.ndarray:
        """Return predictions in original logS units."""
        # Dummy targets (predict doesn't use them)
        dummy_y = np.zeros(len(smiles_in), dtype=np.float32)
        dp_in = _build_datapoints(smiles_in, dummy_y)
        dset_in = data.MoleculeDataset(dp_in, featurizer)
        # NOTE: do NOT call normalize_targets on inference data — the
        # UnscaleTransform inside the predictor handles the inverse.
        loader_in = data.build_dataloader(
            dset_in, batch_size=batch_size, num_workers=0, shuffle=False
        )
        preds = trainer.predict(mpnn, loader_in)
        return torch.cat(preds, dim=0).cpu().numpy().flatten()

    # --- 7. Pack the return dict (same schema as train_mlp) ---
    return {
        "predict": predict,
        "best_valid_rmse": val_rmse_logS,
        "stopped_at": trainer.current_epoch,
        "history": [],  # ChemProp's Trainer doesn't expose easy per-epoch hist
        "config": {
            "depth": depth,
            "mp_hidden_dim": mp_hidden_dim,
            "mp_dropout": mp_dropout,
            "ffn_hidden_dim": ffn_hidden_dim,
            "ffn_num_layers": ffn_num_layers,
            "ffn_dropout": ffn_dropout,
            "max_lr": max_lr,
            "init_lr": init_lr,
            "final_lr": final_lr,
            "warmup_epochs": warmup_epochs,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
        },
        "model": mpnn,
        "trainer": trainer,
    }


__all__ = ["train_chemprop"]
