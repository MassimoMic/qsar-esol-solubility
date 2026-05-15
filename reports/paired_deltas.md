**Paired delta summary (5-seed scaffold, same split per seed):**

| Comparison | ΔRMSE | ΔR² | RMSE wins | R² wins | Note |
|---|---|---|---|---|---|
| P1 → P2 | -0.529 ± 0.158 | +0.361 ± 0.095 | 5/5 | 5/5 | add RDKit descriptors (same RF) |
| P2 → P3 | -0.200 ± 0.036 | +0.102 ± 0.025 | 5/5 | 5/5 | tune XGBoost (same features) |
| P1 → P3 | -0.729 ± 0.125 | +0.462 ± 0.074 | 5/5 | 5/5 | tuned XGB + desc vs untuned RF + Morgan |
| P1 → P4 | -0.155 ± 0.055 | +0.122 ± 0.046 | 5/5 | 5/5 | tuned MLP vs untuned RF (Morgan-only both) |
| P2 → P4 | +0.374 ± 0.140 | -0.239 ± 0.078 | 0/5 | 0/5 | MLP Morgan-only vs RF Morgan+desc |
| P3 → P4 | +0.574 ± 0.105 | -0.341 ± 0.055 | 0/5 | 0/5 | MLP Morgan-only vs XGB Morgan+desc |
| P4 → P5 | -0.287 ± 0.098 | +0.190 ± 0.056 | 5/5 | 5/5 | ChemProp D-MPNN vs MLP (both no global desc) |
| P3 → P5 | +0.287 ± 0.042 | -0.151 ± 0.030 | 0/5 | 0/5 | ChemProp graph vs XGBoost Morgan+desc (best tabular) |
| P2 → P5 | +0.087 ± 0.070 | -0.049 ± 0.041 | 1/5 | 1/5 | ChemProp graph vs RF Morgan+desc |
| P1 → P5 | -0.442 ± 0.110 | +0.312 ± 0.066 | 5/5 | 5/5 | ChemProp vs untuned RF baseline |
