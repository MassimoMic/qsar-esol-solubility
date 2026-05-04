"""Metriche di valutazione e plot diagnostici per QSAR di regressione."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R² — il trio standard per QSAR di regressione."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Measured",
    save_path: str | None = None,
) -> None:
    """Plot 'parity': asse x = misurato, asse y = predetto.

    Se il modello fosse perfetto, tutti i punti starebbero sulla bisettrice.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    metrics = regression_metrics(y_true, y_pred)
    ax.set_xlabel("Measured log S")
    ax.set_ylabel("Predicted log S")
    ax.set_title(
        f"{title}\nRMSE = {metrics['rmse']:.3f}  |  "
        f"MAE = {metrics['mae']:.3f}  |  R² = {metrics['r2']:.3f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Residui vs valori predetti — utile per spottare bias sistematici."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, s=20)
    ax.axhline(0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("Predicted log S")
    ax.set_ylabel("Residual (measured − predicted)")
    ax.set_title("Residual plot")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
