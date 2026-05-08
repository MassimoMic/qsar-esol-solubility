"""
Result persistence and cross-phase comparison.

Two responsibilities:
    1. Read/write per-phase JSON summaries in `reports/` so downstream
       notebooks can compute paired deltas without re-running the upstream
       phases.
    2. Compute paired deltas between phases (same seed → same scaffold split
       → variance of the delta is much smaller than variance of either side).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def save_summary_json(
    path: Path | str,
    phase: str,
    per_seed: List[Dict],
    aggregate: Dict,
    best_params: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """Persist a phase summary to JSON.

    Schema:
        {
            'phase': str,                 # e.g. 'phase4_mlp'
            'per_seed': [
                {'seed': int, 'RMSE': float, 'MAE': float, 'R2': float, ...},
                ...
            ],
            'aggregate': {
                'RMSE_mean': ..., 'RMSE_std': ..., ...
            },
            'best_params': {...} | None,
            'extra': {...} | None,        # arbitrary metadata
        }

    Args:
        path: Output JSON path; parent directory will be created.
        phase: Short identifier matching naming convention 'phaseN_<name>'.
        per_seed: List of per-seed metric dicts. y_true/y_pred arrays, if
                  present, are stripped before serialization.
        aggregate: Output of metrics.aggregate_seed_runs.
        best_params: Hyperparameters used for the run, if any.
        extra: Free-form metadata (timing, dataset version, ...).

    Returns:
        The Path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Strip arrays — they bloat JSON and are easily regenerated.
    cleaned_per_seed = []
    for r in per_seed:
        cleaned = {k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
        # Make sure all values are JSON-serialisable.
        for k, v in cleaned.items():
            if isinstance(v, (np.floating, np.integer)):
                cleaned[k] = v.item()
            elif isinstance(v, np.ndarray):
                cleaned[k] = v.tolist()
        cleaned_per_seed.append(cleaned)

    payload = {
        "phase": phase,
        "per_seed": cleaned_per_seed,
        "aggregate": {k: float(v) if isinstance(v, (np.floating,)) else v
                      for k, v in aggregate.items()},
        "best_params": best_params,
        "extra": extra or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_summary_json(path: Path | str) -> Dict:
    """Load a phase summary previously written by save_summary_json."""
    with open(path) as f:
        return json.load(f)


def paired_delta(
    phase_a_per_seed: List[Dict],
    phase_b_per_seed: List[Dict],
    metric: str = "RMSE",
) -> Dict:
    """Paired (per-seed) delta between two phases.

    Computes phase_b - phase_a per seed, requiring identical seed sets.
    The std of the delta is generally smaller than the std of either
    side because the per-seed scaffold split (the dominant noise source
    on small datasets) cancels out — same seed = same split.

    Args:
        phase_a_per_seed: list of per-seed dicts from baseline phase.
        phase_b_per_seed: list of per-seed dicts from new phase.
        metric: 'RMSE', 'MAE', or 'R2'.

    Returns:
        {
            'metric': str,
            'per_seed_delta': [{'seed': s, 'delta': float}, ...],
            'mean_delta': float,
            'std_delta': float,
            'wins': int,           # how many seeds Phase B improves
            'n_seeds': int,
        }
        Sign convention: negative is better for RMSE/MAE, positive is
        better for R². 'wins' counts seeds where Phase B is better in
        the appropriate direction.
    """
    a_by_seed = {r["seed"]: r[metric] for r in phase_a_per_seed}
    b_by_seed = {r["seed"]: r[metric] for r in phase_b_per_seed}
    common_seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if not common_seeds:
        raise ValueError(
            "No common seeds between the two phases. Cannot compute "
            "paired delta."
        )

    deltas = []
    wins = 0
    for s in common_seeds:
        d = b_by_seed[s] - a_by_seed[s]
        deltas.append({"seed": s, "delta": float(d)})
        # For RMSE/MAE: lower is better → negative delta is a win.
        # For R²: higher is better → positive delta is a win.
        if metric in ("RMSE", "MAE"):
            if d < 0:
                wins += 1
        else:  # R2
            if d > 0:
                wins += 1

    arr = np.array([d["delta"] for d in deltas])
    return {
        "metric": metric,
        "per_seed_delta": deltas,
        "mean_delta": float(arr.mean()),
        "std_delta": float(arr.std()),
        "wins": wins,
        "n_seeds": len(common_seeds),
    }
