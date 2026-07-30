[← README](../README.md)

# Roadmap

The five ESOL phases are complete. The evaluation harness in `src/`
(scaffold splitting, scaffold-disjoint inner CV, multi-seed evaluator, paired
delta reporting, JSON persistence) generalizes to the following.

*These are intentions, not commitments. Work actually outstanding on the ESOL
project itself is listed under [Open questions](../README.md#open-questions).*

---

## ADMET multi-task on Therapeutics Data Commons

Multiple ADMET endpoints (Caco-2, AMES, BBB, hERG, CYP2C9 and others) shared as
multi-task in a single GNN. Likely a custom architecture in PyTorch Geometric
(GIN with edge features), since ChemProp v2's multi-task interface is more rigid
than exploratory work on heterogeneous label noise requires.

**Baseline to beat:** Notwell & Wood (2023) — CatBoost over ECFP + Avalon + ErG
plus 200 descriptors, on the same 22 TDC benchmarks. That is the correct
*higher-bar* baseline; a multi-task GNN should be compared against it head to
head rather than against a fingerprint-only straw man.

This is also the natural setting in which to generalize the ESOL applicability-
domain analysis: the same stratified-residual protocol applied across 22
endpoints would show whether chemotype-localised failure modes are endpoint-
specific or shared.

## Virtual screening on ChEMBL

GNNs on a specific target, likely a kinase. Larger dataset (5k–50k molecules),
where the ratio between the value of graph representation and the value of global
descriptors should tilt toward the graph — testing the *converse* of the ESOL
Phase 5 finding.

Compute will be the binding constraint; see
[Lessons learned #13](lessons-learned.md).

## 3D / solvation-aware modelling of the polyol-aromatic case

The Phase 5 null result identified polyol-aromatic hybrids as responding to
neither model class nor 2D representation. Testing the 3D/solvation hypothesis
directly would mean explicit conformer ensembles (SchNet, PaiNN, equivariant
GNNs) or solubility-specific physics such as SMD solvation energies as auxiliary
features. This is the narrowest and most publishable of the follow-ons, because
the hypothesis is already stated and falsifiable.

## Quantum-classical benchmark

H₂, LiH and H₂O via VQE against DFT and SchNet — connected to parallel work in
quantum machine learning rather than to the QSAR line above.
