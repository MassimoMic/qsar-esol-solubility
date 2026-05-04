\## Reproducing



1\. Clone the repository

2\. Install dependencies: `pip install -r requirements.txt`

3\. Open `notebooks/01\_eda.ipynb` and run all cells — the notebook

&#x20;  downloads ESOL via DeepChem, performs deduplication, and saves

&#x20;  the processed dataset to `data/processed/esol\_dedup.csv`.

4\. Subsequent notebooks consume the processed dataset.



\## Notebooks



\- `01\_eda.ipynb` — Data loading, validation, deduplication, exploratory analysis.

\- `02\_baseline\_rf.ipynb` — \*(in progress)\* Random Forest baseline.



\## Method notes



\- Target: experimental logS (mol/L), unnormalized. The default DeepChem

&#x20; `NormalizationTransformer` is bypassed via `transformers=\[]`.

\- Deduplication on RDKit canonical SMILES revealed 11 duplicate groups

&#x20; hidden in the raw SMILES; values are averaged. One borderline case

&#x20; (a hexitol, ΔlogS ≈ 1.03) likely reflects stereoisomers indistinguishable

&#x20; without stereochemistry annotation; flagged for revisit on richer datasets.

\- Splitting: scaffold split (Bemis–Murcko), recomputed on the deduplicated

&#x20; dataset to avoid train/test leakage on identical molecules.



\## References



\- Delaney, J. S. \*J. Chem. Inf. Comput. Sci.\* 2004, 44, 1000–1005.

\- Wu, Z. et al. \*Chem. Sci.\* 2018, 9, 513–530 (MoleculeNet).

