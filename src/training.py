"""
PyTorch MLP training, Optuna tuning, and 5-seed scaffold evaluation.

Four concentric layers:

    Layer 1 — SolubilityMLP : configurable feed-forward model
    Layer 2 — train_mlp     : train one model with given hyperparameters
                              until early stopping on validation RMSE
    Layer 3 — tune_mlp_optuna : 100-trial Optuna study with scaffold-aware
                                inner k-fold CV on the training set
    Layer 4 — evaluate_5seed_scaffold : model-agnostic 5-seed scaffold
                                        evaluation harness; takes a
                                        train_fn and produces per-seed +
                                        aggregate metrics ready for
                                        save_summary_json

Layer 4 is intentionally model-agnostic so Phase 5 (ChemProp, in
src/chemprop_training.py) can plug in its own train_fn and reuse the
same evaluation protocol verbatim.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .featurization import apply_imputer, fit_median_imputer
from .metrics import aggregate_seed_runs, evaluate
from .splits import scaffold_split_balanced


# =========================================================================
# Reproducibility
# =========================================================================


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) deterministically.

    Sets cudnn deterministic / benchmark=False. A small amount of
    non-determinism may remain on GPU due to atomic algorithms; for
    publication-grade reproducibility this is usually acceptable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# Layer 1 — Model
# =========================================================================


class SolubilityMLP(nn.Module):
    """Feed-forward MLP for QSAR regression on flat features.

    Architecture:
        Input → [ Linear → (BatchNorm) → ReLU → Dropout ] × n_hidden → Linear → 1

    All hidden layer sizes, dropout rate, and use_batchnorm are configurable
    so the same class is the search-space target for Optuna.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty sequence")

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

        # Stash config for later inspection / serialization.
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": tuple(hidden_dims),
            "dropout": float(dropout),
            "use_batchnorm": bool(use_batchnorm),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =========================================================================
# Layer 2 — Train one model
# =========================================================================


@dataclass
class _TargetScaler:
    """Standardize y to (mean=0, std=1), fit on training values only."""

    mean: float = 0.0
    std: float = 1.0

    def fit(self, y: np.ndarray) -> "_TargetScaler":
        self.mean = float(np.mean(y))
        self.std = float(np.std(y))
        if self.std < 1e-8:
            self.std = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


@dataclass
class _EarlyStopper:
    patience: int
    best: float = float("inf")
    counter: int = 0
    best_epoch: int = -1
    best_state: Optional[Dict] = None

    def step(self, current: float, model: nn.Module, epoch: int) -> bool:
        """Return True if training should stop."""
        if current < self.best:
            self.best = current
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train (shuffled) and valid (sequential) DataLoaders."""
    Xtr_t = torch.from_numpy(X_train).float()
    ytr_t = torch.from_numpy(y_train).float()
    Xva_t = torch.from_numpy(X_valid).float()
    yva_t = torch.from_numpy(y_valid).float()

    # Drop the last training batch if it would contain only one sample.
    # BatchNorm1d (and similar) raise ValueError on batches of size 1 because
    # variance is undefined. This can happen when len(X_train) % batch_size == 1
    # — a sporadic Optuna trial pathology that's not worth aborting the trial.
    # On all other batch sizes drop_last=False is preserved (full coverage).
    drop_last = (len(X_train) % batch_size == 1)
    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    # Eval batch size large to amortize forward cost; size is dataset size
    # if it's small enough, otherwise capped.
    eval_bs = min(len(X_valid), 1024)
    valid_loader = DataLoader(
        TensorDataset(Xva_t, yva_t),
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, valid_loader


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true_concat, y_pred_concat) in the model's output space."""
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy()
        ps.append(pred)
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    hidden_dims: Sequence[int] = (512, 256),
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 20,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 10,
    min_lr: float = 1e-6,
    standardize_target: bool = True,
    device: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """Train one MLP until early stopping on validation RMSE.

    The target is internally standardized (mean=0, std=1, fit on train)
    when standardize_target=True. Returned RMSE/predictions are always in
    the original logS space.

    Returns:
        {
            'model':           trained nn.Module loaded with best weights,
            'config':          model architecture config dict,
            'best_valid_rmse': float (in logS space),
            'best_epoch':      int,
            'stopped_at':      int (last epoch run),
            'history':         [{'epoch','train_loss','valid_rmse','lr'}, ...],
            'target_scaler':   _TargetScaler instance (or None),
            'predict':         callable(np.ndarray) -> np.ndarray in logS space,
        }
    """
    seed_everything(seed)
    device_t = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Target standardization
    scaler: Optional[_TargetScaler] = None
    if standardize_target:
        scaler = _TargetScaler().fit(y_train)
        y_train_s = scaler.transform(y_train).astype(np.float32)
        y_valid_s = scaler.transform(y_valid).astype(np.float32)
    else:
        y_train_s = y_train.astype(np.float32)
        y_valid_s = y_valid.astype(np.float32)

    X_train_f = X_train.astype(np.float32)
    X_valid_f = X_valid.astype(np.float32)

    train_loader, valid_loader = _make_loaders(
        X_train_f, y_train_s, X_valid_f, y_valid_s, batch_size
    )

    model = SolubilityMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
    ).to(device_t)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_lr,
    )
    loss_fn = nn.MSELoss()
    stopper = _EarlyStopper(patience=patience)

    history: List[Dict] = []
    last_epoch = 0
    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, loss_fn, device_t
        )
        # Valid RMSE in *original* logS space (de-standardize before metric)
        y_va_s, p_va_s = _eval_one_epoch(model, valid_loader, device_t)
        if scaler is not None:
            y_va = scaler.inverse_transform(y_va_s)
            p_va = scaler.inverse_transform(p_va_s)
        else:
            y_va, p_va = y_va_s, p_va_s
        valid_rmse = float(np.sqrt(np.mean((y_va - p_va) ** 2)))

        scheduler.step(valid_rmse)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_rmse": valid_rmse,
                "lr": float(current_lr),
            }
        )
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(
                f"  epoch {epoch:>3} | train_loss={train_loss:.4f}  "
                f"valid_rmse={valid_rmse:.4f}  lr={current_lr:.2e}"
            )

        if stopper.step(valid_rmse, model, epoch):
            if verbose:
                print(
                    f"  early stop at epoch {epoch} "
                    f"(best valid_rmse={stopper.best:.4f} @ epoch {stopper.best_epoch})"
                )
            break

    # Restore best weights
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    def _predict(X: np.ndarray) -> np.ndarray:
        """Predict in original logS space."""
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(device_t)
            preds = model(x).detach().cpu().numpy()
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
        return preds

    return {
        "model": model,
        "config": model.config,
        "best_valid_rmse": float(stopper.best),
        "best_epoch": int(stopper.best_epoch),
        "stopped_at": int(last_epoch),
        "history": history,
        "target_scaler": scaler,
        "predict": _predict,
    }


# =========================================================================
# Layer 3 — Optuna tuning
# =========================================================================


@dataclass
class MLPSearchSpace:
    """Default Optuna search space for the MLP. Override per project."""

    n_hidden_layers_choices: Tuple[int, ...] = (1, 2, 3)
    hidden_dim_choices: Tuple[int, ...] = (128, 256, 512, 1024)
    dropout_low: float = 0.0
    dropout_high: float = 0.5
    lr_low: float = 1e-5
    lr_high: float = 1e-2
    weight_decay_low: float = 1e-7
    weight_decay_high: float = 1e-3
    batch_size_choices: Tuple[int, ...] = (32, 64, 128, 256)
    use_batchnorm_choices: Tuple[bool, ...] = (True, False)


def _suggest_params(trial, space: MLPSearchSpace) -> Dict:
    """Sample hyperparameters for one Optuna trial."""
    n_layers = trial.suggest_categorical(
        "n_hidden_layers", list(space.n_hidden_layers_choices)
    )
    # Hidden sizes: first layer free, subsequent halve (common heuristic
    # to keep the search space tractable; could be relaxed if needed).
    h1 = trial.suggest_categorical(
        "hidden_dim_1", list(space.hidden_dim_choices)
    )
    if n_layers == 1:
        hidden_dims: Tuple[int, ...] = (h1,)
    elif n_layers == 2:
        hidden_dims = (h1, max(h1 // 2, 32))
    else:
        hidden_dims = (h1, max(h1 // 2, 32), max(h1 // 4, 16))

    return {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float(
            "dropout", space.dropout_low, space.dropout_high
        ),
        "lr": trial.suggest_float(
            "lr", space.lr_low, space.lr_high, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", space.weight_decay_low, space.weight_decay_high, log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", list(space.batch_size_choices)
        ),
        "use_batchnorm": trial.suggest_categorical(
            "use_batchnorm", list(space.use_batchnorm_choices)
        ),
    }


def tune_mlp_optuna(
    df_train_full: pd.DataFrame,
    featurize_fn: Callable[[pd.DataFrame], np.ndarray],
    target_col: str,
    cv_splits: List[Tuple[List[int], List[int]]],
    n_trials: int = 100,
    space: Optional[MLPSearchSpace] = None,
    impute: bool = True,
    max_epochs: int = 200,
    patience: int = 20,
    device: Optional[str] = None,
    seed: int = 42,
    pruner_n_startup_trials: int = 10,
    pruner_n_warmup_steps: int = 0,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    show_progress: bool = True,
):
    """Optuna study with scaffold-aware k-fold CV.

    The cv_splits argument is a list of (train_idx, valid_idx) tuples
    over positional indices into df_train_full — exactly the output of
    splits.scaffold_kfold(df_train_full, ...). This keeps tuning agnostic
    to scaffold-vs-random choice.

    Args:
        df_train_full: Outer training DataFrame (must have mol col +
                       target_col). Inner CV folds are *positional indices*
                       into this DataFrame.
        featurize_fn: callable df -> np.ndarray feature matrix. Called once
                      on the full training set; CV folds index into the
                      resulting matrix.
        target_col: Name of the target column in df_train_full.
        cv_splits: Output of scaffold_kfold — list of (train_idx, valid_idx).
        n_trials: Optuna trial budget.
        space: Search space. Defaults to MLPSearchSpace().
        impute: If True, fits median imputer on each CV training fold.
        max_epochs, patience: Forwarded to train_mlp.
        device: 'cuda' / 'cpu' / None (auto).
        seed: Sampler seed (TPE) and per-trial training seed base.
        pruner_n_startup_trials: Optuna MedianPruner config.
        pruner_n_warmup_steps: Optuna MedianPruner config (epochs reported
                               via trial.report before pruning is enabled).
        study_name, storage: Optional Optuna persistence parameters.
        show_progress: Whether to print Optuna's progress bar.

    Returns:
        (study, best_params): the completed Optuna Study object and the
        best_params dict ready to feed to train_mlp (note hidden_dims is
        already a tuple, not separate n_hidden_layers + hidden_dim_1 keys).
    """
    import optuna  # imported here so the module import is cheap
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    if space is None:
        space = MLPSearchSpace()

    # Featurize the outer training set once. CV folds are positional.
    X_full = featurize_fn(df_train_full)
    y_full = df_train_full[target_col].values.astype(np.float64)

    def _objective(trial):
        params = _suggest_params(trial, space)
        fold_rmses: List[float] = []
        for fold_id, (tr_idx, va_idx) in enumerate(cv_splits):
            X_tr_raw = X_full[tr_idx]
            X_va_raw = X_full[va_idx]
            y_tr = y_full[tr_idx]
            y_va = y_full[va_idx]

            if impute:
                medians, keep = fit_median_imputer(X_tr_raw)
                X_tr = apply_imputer(X_tr_raw, medians, keep)
                X_va = apply_imputer(X_va_raw, medians, keep)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            res = train_mlp(
                X_tr, y_tr, X_va, y_va,
                hidden_dims=params["hidden_dims"],
                dropout=params["dropout"],
                use_batchnorm=params["use_batchnorm"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                max_epochs=max_epochs,
                patience=patience,
                device=device,
                seed=seed + fold_id,  # distinct init per fold
                verbose=False,
            )
            fold_rmses.append(res["best_valid_rmse"])

            # Report intermediate result so MedianPruner can act between folds.
            mean_so_far = float(np.mean(fold_rmses))
            trial.report(mean_so_far, step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses))

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(
        n_startup_trials=pruner_n_startup_trials,
        n_warmup_steps=pruner_n_warmup_steps,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=show_progress)

    raw = study.best_params
    n_layers = raw["n_hidden_layers"]
    h1 = raw["hidden_dim_1"]
    if n_layers == 1:
        hidden_dims: Tuple[int, ...] = (h1,)
    elif n_layers == 2:
        hidden_dims = (h1, max(h1 // 2, 32))
    else:
        hidden_dims = (h1, max(h1 // 2, 32), max(h1 // 4, 16))

    best_params = {
        "hidden_dims": hidden_dims,
        "dropout": raw["dropout"],
        "lr": raw["lr"],
        "weight_decay": raw["weight_decay"],
        "batch_size": raw["batch_size"],
        "use_batchnorm": raw["use_batchnorm"],
    }
    return study, best_params


# =========================================================================
# Layer 4 — Generic 5-seed scaffold evaluator (model-agnostic)
# =========================================================================


def evaluate_5seed_scaffold(
    df: pd.DataFrame,
    target_col: str,
    featurize_fn: Callable[[pd.DataFrame], np.ndarray],
    train_fn: Callable,
    train_kwargs: Dict,
    seeds: Sequence[int] = (42, 0, 1, 2, 3),
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    impute: bool = True,
    smiles_col: Optional[str] = "smiles",
    verbose: bool = True,
) -> Dict:
    """5-seed balanced-scaffold evaluation, model-agnostic.

    Same protocol used in Phase 1, 2, 3. Once written here, every future
    model class just plugs in by providing a `train_fn` with signature
    (X_tr, y_tr, X_va, y_va, **train_kwargs) -> dict with key 'predict'.

    Args:
        df: Deduplicated dataset DataFrame with 'mol' column and target_col.
        target_col: Target column name (e.g. 'logS').
        featurize_fn: callable df -> np.ndarray feature matrix.
        train_fn: callable (X_tr, y_tr, X_va, y_va, **kwargs) -> dict.
                  Must return {'predict': callable(X) -> np.ndarray, ...}.
        train_kwargs: dict of hyperparameters fed verbatim to train_fn.
        seeds: Outer scaffold seeds. Default = standard 5-seed sequence.
        train_frac, valid_frac, test_frac: Outer split fractions.
        impute: If True, fit/apply median imputer on each split.
        smiles_col: If set and present, included in per-seed prediction
                    arrays for downstream stratified analysis. None to skip.
        verbose: Print per-seed line as it completes.

    Returns:
        {
            'per_seed': [
                {'seed': s, 'RMSE': r, 'MAE': m, 'R2': r2,
                 'y_true': np.ndarray, 'y_pred': np.ndarray,
                 'test_smiles': list[str] | None,
                 'train_size': int, 'valid_size': int, 'test_size': int,
                 'train_extra': dict   # whatever train_fn returned beyond 'predict'
                },
                ...
            ],
            'aggregate': aggregate_seed_runs output,
            'config': {'seeds': ..., 'train_kwargs': ...},
        }
    """
    per_seed: List[Dict] = []
    for s in seeds:
        tr_idx, va_idx, te_idx = scaffold_split_balanced(
            df, train_frac=train_frac, valid_frac=valid_frac,
            test_frac=test_frac, seed=s,
        )
        df_tr = df.loc[tr_idx].reset_index(drop=True)
        df_va = df.loc[va_idx].reset_index(drop=True)
        df_te = df.loc[te_idx].reset_index(drop=True)

        X_tr_raw = featurize_fn(df_tr)
        X_va_raw = featurize_fn(df_va)
        X_te_raw = featurize_fn(df_te)
        y_tr = df_tr[target_col].values.astype(np.float64)
        y_va = df_va[target_col].values.astype(np.float64)
        y_te = df_te[target_col].values.astype(np.float64)

        if impute:
            medians, keep = fit_median_imputer(X_tr_raw)
            X_tr = apply_imputer(X_tr_raw, medians, keep)
            X_va = apply_imputer(X_va_raw, medians, keep)
            X_te = apply_imputer(X_te_raw, medians, keep)
        else:
            X_tr, X_va, X_te = X_tr_raw, X_va_raw, X_te_raw

        result = train_fn(X_tr, y_tr, X_va, y_va, seed=s, **train_kwargs)
        if "predict" not in result:
            raise ValueError(
                "train_fn must return a dict with a 'predict' callable."
            )
        y_pred_te = result["predict"](X_te)

        m = evaluate(y_te, y_pred_te, split_name="test", include_arrays=True)
        m["seed"] = s
        m["train_size"] = len(df_tr)
        m["valid_size"] = len(df_va)
        m["test_size"] = len(df_te)
        if smiles_col is not None and smiles_col in df_te.columns:
            m["test_smiles"] = df_te[smiles_col].tolist()
        else:
            m["test_smiles"] = None
        # Pass through everything from the trainer that isn't 'predict' or
        # 'model' (state-dict reference shouldn't be JSON-serialised here).
        m["train_extra"] = {
            k: v for k, v in result.items()
            if k not in ("predict", "model", "target_scaler")
        }
        per_seed.append(m)

        if verbose:
            print(
                f"Seed {s:>2}: test RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  "
                f"R²={m['R2']:+.3f}  (n_test={m['test_size']})"
            )

    agg = aggregate_seed_runs(per_seed)
    if verbose:
        print(
            f"\nAggregate over {len(seeds)} seeds:\n"
            f"  RMSE = {agg['RMSE_mean']:.3f} ± {agg['RMSE_std']:.3f}\n"
            f"  MAE  = {agg['MAE_mean']:.3f} ± {agg['MAE_std']:.3f}\n"
            f"  R²   = {agg['R2_mean']:+.3f} ± {agg['R2_std']:.3f}"
        )

    return {
        "per_seed": per_seed,
        "aggregate": agg,
        "config": {
            "seeds": list(seeds),
            "train_kwargs": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in train_kwargs.items()
            },
        },
    }
