# Feature audit — 2026-09-17

This audit separates product completeness from scientific validity. A feature can
be technically complete while its model output remains research-only or blocked
from promotion by the evidence gate.

## 1. Molecule analysis

### What is already solid

- RDKit parsing and deterministic identity normalization.
- Explicit stereochemistry resolution before chiral prediction and 3D work.
- 2D depiction plus ETKDGv3 conformer ensemble with convergence/round-trip checks.
- Production prediction provenance, nearest-training similarity and reliability state.
- Shadow-model comparison is additive and cannot silently replace production.
- Academic/reference evidence is kept separate from prediction.

### Main blocker

The production odor model is still constrained by label quality. The reviewed
catalog release does not yet provide eligible assessed-negative support for all
113 targets, so production retraining/calibration remains blocked by the data
gate. Better architecture cannot manufacture missing ground truth.

### Decision

Do not replace the Judge simply because a newer graph/foundation model is
available. New encoders belong in shadow/benchmark mode until they pass the same
locked and prospective gates.

## 2. Candidate design

### What is already solid

- Chemistry-valid candidates are scored before expensive external/reference/3D work.
- Stereo variants are bounded and unresolved cases fail to review rather than being guessed.
- Reference checks fail closed when configured evidence is ambiguous or unavailable.
- Target-fit relaxation is explicit rather than presented as a successful strict match.
- Retrosynthesis, academic evidence, chemistry and odor prediction remain separate signals.

### Main blockers

1. The quality ceiling is the Judge: target-aligned generation cannot be more
   trustworthy than the target model used to score it.
2. The legacy character-level generator is still a fallback. Conditional SELFIES
   is architecturally better, but it should remain registry-gated until its own
   validity/diversity/prospective criteria pass.
3. Current chemistry bounds and structural alerts are useful deterministic
   triage, not a fragrance safety or volatility model. They need empirical audit
   against a representative known fragrance-material set before being treated as
   a discovery prior.

### Decision

Do not use Boltz to re-rank generated candidates yet. Receptor binding is a new
evidence dimension and must first prove incremental value against experimental
olfactory-receptor data.

## 3. Sensory data intake

### What is already solid

- Manual and CSV/XLSX validation paths.
- `PRESENT`, `ABSENT` and `UNASSESSED` remain distinct.
- Context fields retain concentration, solvent, temperature, replicate and provenance.
- Commits create immutable dataset snapshots and corrections are append-only.

### Missing product layer

The scientific protocol defines stronger panel requirements than the current UI
orchestration exposes. The next useful product work is not more CRUD; it is a
panel workflow that can enforce/randomize blinded samples, track assessor/session
eligibility, collect required replicates and calculate release-gate statistics
(Krippendorff alpha and ICC) without manual stitching.

This matters more to Judge promotion than adding another inference provider.

## 4. Receptor intelligence / Boltz

### Applicable now

Boltz-2.1 can accept a protein sequence and ligand SMILES, predict the molecular
complex, and return ligand-protein binding and structure-confidence metrics. That
makes it suitable for a **predicted receptor-binding evidence** experiment.

The new integration intentionally provides:

- sequence and SMILES validation;
- deterministic request idempotency;
- cost estimation before execution;
- explicit external-data consent;
- explicit live-compute consent;
- a local spending ceiling;
- normalized provider/model/version provenance;
- hashes rather than raw receptor sequence in normalized results;
- scientific limitations in every evidence record;
- no production/UI/ranking coupling.

### Not justified yet

- Calling a binding score `OR activation`.
- Mapping binding confidence directly to odor descriptors/intensity.
- Using Boltz output as a candidate hard gate.
- Generating novel fragrance molecules around an OR target before the receptor
  benchmark and downstream odor/volatility/safety objectives exist.

### Validation gate

Benchmark Boltz on reviewed receptor/ligand pairs with experimental ground truth.
Report AUPRC/AUROC, precision@K/recall@K per receptor, grouped bootstrap CIs,
scaffold/connectivity-held-out performance, stereochemical subsets, provider
confidence metrics and failure rate.

Only if Boltz shows useful incremental signal should the web application gain a
Receptor Evidence panel. It should still remain separate from the Judge initially.

## 5. Features explicitly not worth building yet

The navigation currently advertises future Projects, Library and Batch Runs.
Those are plausible workspace features, but they should not be implemented just
because the placeholders exist. Build them only when repeated R&D workflows show
a concrete need for persistent project grouping, reusable material collections,
or queued/batch jobs.

For a solo-operated research product, adding job orchestration, permissions and
persistent project state prematurely creates operational surface area without
improving the evidence bottleneck.

## Recommended order

### P0 — evidence quality

1. Acquire/review assessed negatives and intensity observations.
2. Complete panel orchestration and release-gate analytics.
3. Re-run calibration and locked/prospective promotion gates.

### P1 — model/receptor experiments

1. Run the Boltz OR benchmark with test mode, then a tightly capped live sample.
2. Benchmark a 3D molecular encoder / receptor-aware model against the current
   fingerprint and graph baselines under leakage-resistant splits.
3. Only then expose receptor evidence in the analysis UI.

### P2 — broader olfactory modeling

1. Mixture perceptual representations.
2. Odor intensity / threshold models with physically meaningful concentration
   and volatility context.
3. Multi-objective molecule design combining odor, receptor evidence,
   physicochemical/volatility constraints, synthesis, safety and uncertainty.

### Deferred product infrastructure

Projects, Library and Batch Runs remain deferred until usage data demonstrates
that session-only workflows are the bottleneck.
