**Paired delta summary (5-seed scaffold, same split per seed):**

| Comparison | ΔRMSE | ΔR² | RMSE wins | R² wins | Note |
|---|---|---|---|---|---|
| P1 → P2 | -0.529 ± 0.158 | +0.361 ± 0.095 | 5/5 | 5/5 | add RDKit descriptors (same RF) |
| P2 → P3 | -0.200 ± 0.036 | +0.102 ± 0.025 | 5/5 | 5/5 | tune XGBoost (same features) |
| P1 → P3 | -0.729 ± 0.125 | +0.462 ± 0.074 | 5/5 | 5/5 | tuned XGB + desc vs untuned RF + Morgan |
| P1 → P4 | -0.155 ± 0.055 | +0.122 ± 0.046 | 5/5 | 5/5 | tuned MLP vs untuned RF (Morgan-only both) |
| P2 → P4 | +0.374 ± 0.140 | -0.239 ± 0.078 | 0/5 | 0/5 | MLP Morgan-only vs RF Morgan+desc |
| P3 → P4 | +0.574 ± 0.105 | -0.341 ± 0.055 | 0/5 | 0/5 | MLP Morgan-only vs XGB Morgan+desc |
