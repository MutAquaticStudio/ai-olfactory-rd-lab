# Scientific evaluation and sensory protocol

This protocol defines evidence gates for Scent Molecule Studio. It is informed
by ISO 13299 sensory-profile principles but is not an ISO certification.

## Evidence levels

1. **Weak catalog evidence** — Leffingwell and GoodScents descriptor mentions.
   A mention can be `PRESENT`; omission remains `UNASSESSED`.
2. **Published panel evidence** — Dravnieks and Keller measurements retained in
   their original source domains, scales, subjects and replicates.
3. **Internal expert screen** — confirms sample identity, lexicon, solvent,
   concentration and suitability before panel exposure.
4. **Internal blinded panel** — at least eight eligible pseudonymous assessors,
   randomized sample codes and two independent sessions.
5. **Prospective validation** — generated candidates, matched controls and
   reference controls evaluated blind; at least 30 samples in a batch.

## Required observation fields

- Raw and standardized molecular identity, full/connectivity InChIKey and stereo
  state.
- Study, session, blinded sample code, pseudonymous assessor and replicate.
- Concentration with unit, solvent, temperature, preparation time and controls.
- Descriptor ontology version, tri-state presence, conditional 0–10 intensity,
  confidence and notes.
- Source name/version/license, raw identifier, batch hash and correction link.

## Panel release gate

- At least eight assessors each provide at least two replicates.
- Presence agreement is quantified by nominal Krippendorff alpha.
- Conditional intensity agreement is quantified by ICC(2,k).
- Default training thresholds are alpha ≥ 0.5 and ICC ≥ 0.5. Values and excluded
  targets remain in provenance; they are masked from training, not deleted.
- Descriptor support below ten positives in locked test is not promoted using an
  isolated per-label metric.

## Model evaluation

- Connectivity/scaffold groups never cross train, validation or locked test.
- Grouped five-fold CV with three seeds is restricted to development data.
- Calibration and thresholds use validation only. Locked test is evaluated once
  for a promotion decision; private blinded-panel data is never used for tuning.
- Bootstrap confidence intervals sample molecule groups rather than rows.
- Report results by source, nearest-training similarity bin and stereo challenge
  subset, including failed and missing cases.

### Architecture and promotion

The production API consumes the `MoleculePredictor` contract and currently
adapts the frozen Morgan MLP baseline. Chemprop and the optional DeepChem
chiral graph adapter run only in the Python 3.11–3.12 training environment.
Each run records the dataset hash, immutable split hash, Git commit, seed,
calibration artifact, metrics, and weight checksums under `artifacts/`. A model
is promoted only through an explicit quality-gate decision; registry updates
are atomic and retain the previous entry for rollback.

## Creator evaluation

For each target profile, sample 1,000 structures and report RDKit validity,
canonical uniqueness, chemistry PASS, scaffold novelty, internal diversity,
property/SA distributions and out-of-domain rate. Compare conditional target
enrichment with an unconditional control using grouped bootstrap confidence
intervals. No model is promoted when Judge score improves by reducing diversity
or pushing candidates out of domain.

Prospective promotion requires the blind-panel target presence/intensity effect
against matched controls to have a 95% confidence interval whose lower bound is
above zero.

## Privacy and retention

Assessor identity is pseudonymous. Private raw files, SQLite/WAL files and model
development artifacts stay outside Git and are not sent to external services.
PubChem receives candidate Isomeric SMILES only after explicit session consent;
sensory observations are never transmitted.
