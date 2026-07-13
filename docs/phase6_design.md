# Phase 6 — Additive MLP+GNN on AqSolDB — `src/` design

Status: draft, pre-notebook. Goal: decide module boundaries and interfaces
before writing `09_aqsoldb_additive_gnn.ipynb`, following the project's
existing pattern (design doc → src/ → notebook).

Reference implementation: Bhattacharya & Roy, "An Additive MLP–GNN Framework
for Characterizing Chemical and Structural Contributions to Aqueous
Solubility," arXiv:2607.02212 (Jul 2026).

---

## 0. What's reused verbatim, what's new

| Module | Status | Notes |
|---|---|---|
| `splits.py` | **reused, unchanged** | `scaffold_split_balanced` / `scaffold_kfold` operate on any DataFrame with a `mol` column. AqSolDB just needs to produce that column. |
| `metrics.py` | **reused, unchanged** | `evaluate` / `aggregate_seed_runs` are metric-only, dataset-agnostic. |
| `reporting.py` | **reused, unchanged** | `save_summary_json` / `paired_delta` don't care about model architecture. |
| `training.py` | **reused for `evaluate_5seed_scaffold`**, extended for the model | The 5-seed evaluator already takes any `train_fn(X_tr, y_tr, X_va, y_va, seed=..., **kwargs) -> {'predict': callable}`. Problem: it assumes `featurize_fn` returns a flat `np.ndarray`. The two-branch model needs *two* aligned inputs (descriptor vector + graph). See §3 for the adapter. |
| `featurization.py` | **extended** | Add a 15-descriptor subset matching the paper's chemical branch (see §2). Morgan/RDKit-217 functions untouched. |
| `aqsoldb_curation.py` | **new** | AqSolDB-specific loading, light curation, canonical SMILES, `mol` column. Deliberately *not* named `curation.py` — keeps ESOL's implicit curation (in notebook 01) separate from AqSolDB's. |
| `graph_data.py` | **new** | SMILES → PyTorch Geometric `Data` object. Atom/bond featurization. |
| `additive_model.py` | **new** | `ChemicalBranch` (MLP), `StructuralBranch` (GCN or MPNN), `AdditiveSolubilityModel` (combiner with optional interaction term τ), `train_additive` (Layer-2-equivalent of `training.py`'s `train_mlp`). |

Nothing in `splits.py`, `metrics.py`, `reporting.py` needs to change — this
is the payoff of Lesson #10 (persist canonical JSON) and the model-agnostic
Layer 4 design from Phase 4 onward.

---

## 1. `aqsoldb_curation.py` — new

You said dirty data is fine, so this stays intentionally light — no
sugar-coating the curation, just enough to not corrupt the split.

```python
def load_aqsoldb(path: str) -> pd.DataFrame:
    """Load raw AqSolDB CSV. Returns columns: smiles, logS (renamed from
    whatever AqSolDB calls its target — check on load, don't assume)."""

def canonicalize_and_flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical SMILES via RDKit. Adds a `dup_group` column (int id per
    canonical-SMILES group, -1 if singleton) — does NOT drop or average
    duplicates automatically. Unlike ESOL's silent averaging, AqSolDB
    duplicates get flagged and left for a decision cell in the notebook,
    since with 9 merged source datasets some "duplicates" may carry
    genuinely different measurement conditions worth inspecting first."""

def add_mol_column(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """RDKit MolFromSmiles, drops rows where parsing fails (log count),
    adds `mol` column — same convention as ESOL's df so splits.py works
    unchanged."""
```

Decision left open for the notebook: whether to average duplicate-group
targets (ESOL precedent) or keep the first occurrence and log the rest.
Given the "dirty is fine" stance, I'd lean toward the cheaper option
(first-occurrence + logged count) unless the duplicate rate turns out to
be large enough to bias the split — that's an empirical check in notebook
09's first cell, not a decision to make blind here.

---

## 2. `featurization.py` — extension

Add one function, additive to the existing file:

```python
CHEM_BRANCH_DESCRIPTORS = [
    "MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors",
    "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumValenceElectrons", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "RingCount", "TPSA", "LabuteASA",
]  # the paper's 15 — deliberately excludes BalabanJ / BertzCT since those
   # are graph-topology-derived and would leak into the structural branch

def featurize_chem_branch(df, mol_col="mol") -> np.ndarray:
    """Subset of DESC_LIST restricted to CHEM_BRANCH_DESCRIPTORS, in that
    order. Reuses the existing DESC_LIST lookup — no new RDKit calls."""
```

This is a *strict subset* of what `featurize_rdkit_descriptors` already
computes — implemented as a column-select, not a new descriptor
computation, so it stays consistent with the 217-descriptor Phase 2/3
featurization if you ever want to cross-check.

Standardization (`StandardScaler`-equivalent) is fit-on-train-only, same
discipline as `fit_median_imputer` — added as a tiny paired
`fit_standard_scaler` / `apply_standard_scaler` in the same module, mirroring
the existing imputer pattern rather than reaching for sklearn's version
(keeps the whole file dependency-light, matches the file's existing style).

---

## 3. `graph_data.py` — new

```python
def mol_to_pyg_data(mol) -> torch_geometric.data.Data:
    """Node features: atomic number, degree, formal charge, hybridization
    (one-hot), aromaticity (bool). Edge features (both directions):
    bond type (one-hot), conjugation (bool), ring membership (bool).
    Matches the paper's feature set closely enough for comparability,
    while staying a strict subset of what ChemProp already computes
    internally in Phase 5 — so a future cross-reference is possible."""

def featurize_graph_batch(df, mol_col="mol") -> list[Data]:
    """One Data object per row, order-aligned with the DataFrame."""
```

No new dependency beyond `torch_geometric`, already used nowhere yet in
this repo but a natural addition (Project 1's ADMET GNN work will need it
too — this is shared infrastructure, not one-off).

---

## 4. `additive_model.py` — new, the core piece

### 4.1 Model classes

```python
class ChemicalBranch(nn.Module):
    """3-hidden-layer MLP, width h=4 (paper default; exposed as a
    constructor arg since ESOL/AqSolDB might want different width given
    the 9x data difference — treat h as an Optuna-tunable, not a
    hardcoded constant, unlike the paper)."""
    def forward(self, x) -> torch.Tensor:  # scalar per molecule
        ...

class StructuralBranch(nn.Module):
    """Either GCN or MPNN encoder (torch_geometric), 4 message-passing
    layers, embedding dim d (tunable), global max+mean pooling, linear
    head to scalar. `encoder_type: Literal['gcn', 'mpnn']` constructor arg
    so both configurations from the paper are reachable from one class."""
    def forward(self, batch) -> torch.Tensor:  # scalar per molecule
        ...

class AdditiveSolubilityModel(nn.Module):
    """y_hat = kappa + g(x) + f(G) + tau * g(x) * f(G)
    tau is a learnable scalar Parameter, or fixed at 0.0 when
    use_interaction=False (the no-interaction ablation the paper found
    consistently better — worth reproducing as a first check on AqSolDB
    before assuming it holds here too)."""
    def forward(self, x, batch) -> torch.Tensor:
        ...
    def branch_outputs(self, x, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (g(x), f(G)) separately — needed for the decomposition
        analysis (§5), not just the combined prediction."""
```

### 4.2 Training function — the `training.py` Layer-2 equivalent

```python
def train_additive(
    X_tr, G_tr, y_tr, X_va, G_va, y_va,
    encoder_type="mpnn", chem_hidden=4, graph_dim=8, use_interaction=False,
    lr=1e-3, max_epochs=5000, patience=..., batch_size=10, seed=42,
    device="cpu", verbose=False,
) -> dict:
    """Mirrors train_mlp's contract: early stopping on valid RMSE, best
    checkpoint restored, target NOT standardized (paper trains on raw
    logS directly — deviates from this repo's Phase 4 convention of
    internal target standardization; worth an explicit note in the
    notebook on why, and possibly an ablation, since Phase 4's BatchNorm
    finding suggests standardization matters a lot for MLP convergence
    on Morgan-sized inputs — the chem branch here is much narrower, 15
    features, so the effect may differ).
    Returns {'predict': callable, 'branch_outputs': callable, ...}
    matching evaluate_5seed_scaffold's contract.
    """
```

### 4.3 The featurize_fn / train_fn mismatch — the one real adapter needed

`evaluate_5seed_scaffold` calls `featurize_fn(df) -> np.ndarray` once per
split and passes the result straight to `train_fn`. The additive model
needs two aligned objects (descriptor matrix + list of graph `Data`), not
one array. Two options:

- **(A) Bundle**: `featurize_fn` returns a small wrapper object
  `AdditiveFeatures(X=array, graphs=list[Data])` that supports `__len__`
  and slicing, so `evaluate_5seed_scaffold`'s existing indexing
  (`X_tr[idx]`-style operations happen upstream via DataFrame row
  selection, not on the array itself) keeps working untouched — the
  imputer step (`impute=True` default) would need `impute=False` since
  the bundle isn't a raw NaN-bearing array; standardization happens
  inside `featurize_chem_branch` instead.
- **(B) Fork**: write `evaluate_5seed_scaffold_additive` as a thin
  variant in `additive_model.py` that duplicates the outer scaffold-split
  loop but calls both featurizers.

**Recommendation: (A).** It's a few lines (one dataclass with `__len__`/
`__getitem__`), keeps one single source of truth for the outer-loop
protocol (scaffold split → 5 seeds → paired delta → JSON), and avoids a
second copy of logic that would need to stay in sync with `training.py`
forever. This is the same reasoning that already motivated making Layer 4
model-agnostic for Phase 5 — worth applying it here too instead of
special-casing.

---

## 5. Decomposition / interpretability analysis — `notebooks/09` only, not `src/`

The paper's best-linear-projection (MBLP) and GNNExplainer analyses are
one-off diagnostics, not reusable infrastructure — same treatment as the
Phase 4 worst-10 SMARTS audit (notebook 07, not `src/`). Keep them in the
notebook. One exception worth promoting to `src/`: the **structural
variance share** ρ = std(f)/(std(f)+std(g)) and the **run-to-run
correlation** check (10 seeds, 45 pairs) are cheap, reusable stability
diagnostics that could matter for Project 1 too — candidate for a small
`interpretability.py` if it proves useful, but not blocking for a first
pass.

---

## 6. Open questions before writing the notebook

1. **Duplicate handling in AqSolDB** — average vs. first-occurrence
   (§1). Empirical check first cell of 09, not decided here.
2. **Target standardization** — paper trains on raw logS; this repo's
   Phase 4 precedent standardizes internally. Worth an early ablation
   given how decisive BatchNorm/standardization turned out to be in
   Phase 4's FANOVA (73% variance on a single binary knob) — the
   additive model's chem branch is narrower, so the effect might be
   smaller, but shouldn't be assumed away.
3. **GCN vs MPNN for the structural branch** — paper found MPNN
   marginally better on AqSolDB (1.06 vs 1.07 MAE), consistent with
   Phase 5's ChemProp choice. Default to MPNN, keep GCN reachable via the
   `encoder_type` arg for a one-line ablation.
4. **Compute budget** — AqSolDB (9,982 molecules) is ~9x ESOL. The
   paper trains 5000 epochs, batch size 10 — that's ~5M gradient steps
   per run before early stopping. Worth a quick epoch-time probe on
   Colab before committing to the full 5-seed + Optuna protocol; may
   need to reduce max_epochs or increase batch_size relative to the
   paper's defaults, similar to how Phase 4 reduced Optuna's per-trial
   budget relative to the final 5-seed evaluation.
