# Judge data release — evidence before retraining

The current release implements the data/audit stages of the Judge program. It
does not satisfy the gate for retraining or production promotion. Judge v1 and
the existing shadow artifact remain the current models.

## Results, 2026-09-05

| Stage | Result |
|---|---|
| Reviewed catalog snapshot | 4,729 source records and 4,729 molecular identities |
| Connectivity audit | 4,227 connectivity groups; 1,854 identities have unresolved stereo |
| Catalog observations | 21,746 PRESENT; zero assessed ABSENT |
| Maturity | All 113 labels lack eligible negative support |
| Audit partitions | Train 2,818; calibration 483; validation 717; locked test 711 |
| Leakage | Zero connectivity/scaffold/group overlap; largest group has 768 molecules |
| Public panel intake | Keller 2016: 1,430,000 source measurement rows preserved |
| Public panel data gate | REVIEW_REQUIRED; no training targets released |
| Retraining / new learning curves | Not started because the data gate is unmet |

Butina groups do not guarantee a maximum similarity between partitions. The
report includes pair counts at Tanimoto ≥0.60 and nearest-neighbor statistics.
Some distinct identities have identical 2,048-bit fingerprints; the maximum
cross-partition similarity in this audit is 1.0. These statistics are not used
to redraw the split after inspecting model performance.

## Reproduce the catalog release

Use the training environment (Python 3.11 or 3.12):

```bash
.venv-training/bin/python prepare_judge_data.py \
  --reviewed-artifact artifacts/data/clean-master-113-reviewed-v1 \
  --data '/private/path/clean_master_olfactory_db.csv' \
  --dataset-version clean-master-evidence-audit-v1 \
  --report-dir artifacts/data/clean-master-evidence-audit-v1
```

The default data root is `~/.scent-molecule-studio`, overridable with
`SCENT_STUDIO_DATA_DIR` or `--data-root`. Evidence uses the existing
`studio.sqlite3`; no second database is created. Snapshots appear in Data
intake's dataset version listing. Source and measurement records never require
invented assessor IDs, temperature, concentration or study sessions.

`--resume` verifies the existing snapshot and its source before producing a
new report directory. Existing snapshots and audit directories are never
overwritten. To audit a release without loading production model resources:

```bash
.venv-training/bin/python build_split_manifest.py \
  --evidence-snapshot /private/data/snapshots/release/manifest.json \
  --output artifacts/benchmarks/release-audit.json
```

This writes an audit directory named `release-audit`, containing the split,
coverage CSV/JSON, partition coverage, source membership, collection queue,
report and checksums. Audit manifests cannot be passed to training.

## Public source decisions

- **Keller and Vosshall (2016):** the article's rights notice applies CC0 to
  article data unless stated otherwise. The pinned Pyrfume data archive is
  mirrored locally with checksums and normalized in bounded Parquet batches.
  [Original study and rights notice](https://link.springer.com/article/10.1186/s12868-016-0287-2).
- **Dravnieks (1985):** remains disabled. No reusable data license was confirmed;
  the ASTM product page states a restriction on AI use. A fixture-tested
  aggregate adapter is present, but no real archive data was downloaded or
  imported. [ASTM source](https://store.astm.org/ds61-eb.html).

```bash
.venv-training/bin/python ingest_pyrfume_sources.py --help
.venv-training/bin/python import_public_panel.py \
  --archive-dir /private/data/raw/pyrfume/keller_2016/8054ea98ed675005ec10e67359902f500e4911b0 \
  --dataset-version keller-2016-panel-audit-v1
```

Keller's processed table contains 1,028,739 missing values, 263,684 numeric
ratings and 9,728 explicit zeros across all measurement types. It also includes
questions about familiarity, intensity and detection; those counts are not
counts of odor-label negatives. Five source terms match the fixed vocabulary
literally. All measurements retain `UNASSESSED` training state pending protocol
review, even when the raw rating is zero. Descriptor ratings are not silently
renamed as conditional intensity.

`MUSKY` is sent to review: the study explicitly discusses its different meaning
for the lay panel and perfumery experts. Source observations with repeated
stimulus/subject keys remain separate rows. The processed export lacks explicit
session/replicate fields, so those fields remain null. Recovery requires the
original study records, not a synthetic repetition counter.

## Keller original-trial recovery (2026-09-06)

The optional original-workbook adapter restores trial provenance from the
published supplementary XLSX. Place the original workbook in the existing
private archive directory; its filename and SHA-256 must match
`data/source_registry.json`. Importing makes no network request and does not
approve protocol semantics or training labels.

```bash
.venv-training/bin/python import_public_panel.py \
  --archive-dir /private/data/raw/pyrfume/keller_2016/8054ea98ed675005ec10e67359902f500e4911b0 \
  --original-workbook 12868_2016_287_MOESM1_ESM.xlsx \
  --dataset-version keller-2016-trials-audit-v3
```

Each response is checked against the original value, descriptor, subject,
source identifier and dilution before publication. Extra/truncated rows,
duplicate subject/VIAL identities, incompatible metadata or invalid checksums
abort the import without registering a snapshot. The previous CSV-only import
remains available. Existing snapshot versions are never overwritten.

The new Parquet schema retains source Excel row/sheet, `VIAL #`, stable trial
ID, raw CID/CAS, odor name, catalog number, detection response and the literal
source replicate marker. `session_id` and chronological `replicate_number`
remain null: neither is inferred from workbook order. No demographic fields
are copied. The workbook checksum and source URL are recorded in the manifest.

Blank values are classified as unrecorded descriptor values or skipped after
no detection; **both remain `UNASSESSED`**. Detection/recognition/free text are
not numeric ratings, even when the text happens to be a number. Reports count
descriptor zeros separately from other scale zeros. The reviewed workbook has
zero explicit descriptor zeros; the 9,728 zeros belong to overall intensity,
pleasantness and familiarity, not negative odor labels.

`structure_review_queue.csv` collects unresolved stereo and missing identities;
raw CAS strings in the source CID column are preserved and flagged, not
silently replaced with PubChem IDs. Multiple warnings are retained together,
including MUSKY's semantic conflict. Mapping and evidence review remain
pending. The release is `AUDIT_ONLY`, not a training snapshot.

## Keller CAS-only correction (2026-09-08)

The user-authorized correction for **isobutyl acetate** changes the effective
`source_cas` from `109-19-0` to `110-19-0`. The original value remains in
`raw_source_cas`; affected rows carry `cas_correction_id`. The original XLSX,
CSV and prior snapshots are not edited. The corrected CAS is supported by the
[manufacturer record](https://www.sigmaaldrich.com/US/en/product/sigald/537470).

`data/keller_isobutyl_cas_correction_v1.json` pins the parent manifest and
workbook SHA-256, exact source name/identifier and expected scope: 2,860
measurement rows across 110 source trials. Import fails if these checks do
not match. A new immutable snapshot includes `cas_correction.json` and an audit
event linking it to its parent.

```bash
.venv-training/bin/python import_public_panel.py \
  --archive-dir /private/data/raw/pyrfume/keller_2016/8054ea98ed675005ec10e67359902f500e4911b0 \
  --original-workbook 12868_2016_287_MOESM1_ESM.xlsx \
  --cas-correction data/keller_isobutyl_cas_correction_v1.json \
  --dataset-version keller-2016-trials-cas-corrected-v4
```

This correction does **not** repair the raw CID field, assign a SMILES,
approve the physical sample identity, approve sensory evidence or release
training targets. The child remains `AUDIT_ONLY` and `training_eligible=false`.
The production model and its weights are unchanged.

## Keller locked harmonization and identity dispositions (2026-09-08)

`keller-2016-harmonized-audit-v5` derives from the CAS-corrected v4 snapshot.
It preserves all 1,430,000 measurements and applies the user-authorized,
versioned `data/keller_descriptor_policy_v1.json`. This is source harmonization,
not a claim that lay-panel ratings and perfumery catalog labels are equivalent.

- Literal: SWEET → sweet; GARLIC → garlic; SOUR → sour; BURNT → burnt; WARM → warm.
- Broad category: FRUIT → fruity; FISH → fishy; SPICES → spicy; WOOD → woody;
  GRASS → grassy; FLOWER → floral. No propagation to narrower descriptors.
- Source-only: EDIBLE, BAKERY, COLD, ACID, MUSKY, SWEATY, AMMONIA/URINOUS,
  DECAYED and CHEMICAL. In particular, MUSKY is not mapped to perfumery musk.

The 11 mapped descriptors cover 605,000 response slots and 72,860 positive
trial observations. These are not independent molecules or released training
targets. Blank/zero values remain `UNASSESSED`; raw applicability stays on its
source scale. No molecular consensus, assessed negatives or conditional
intensity values are inferred.

All 149 identities from the original unresolved-identity queue have an explicit
disposition (this is not a re-verification of every molecule in Keller):

| Decision | Identities |
|---|---:|
| Accept database reference structure only | 64 |
| Hold for stereo/material evidence | 68 |
| Hold for conflicting identifiers | 12 |
| Hold for ambiguous CAS matches | 4 |
| CAS correction applied; CID/sample identity still unresolved | 1 |

Accepted database structures are stored in `identity_decisions.json` and CSV,
**not substituted for the sample's SMILES**. The 85 unresolved items remain in
`identity_hold_queue.csv`. Resolution may require source corrections, material
composition or lot documentation; user approval alone cannot establish those
facts. Every decision links to cached evidence with hashes. The unchanged
historical cache remains private and is referenced by the new manifest.

```bash
.venv-training/bin/python finalize_keller_review.py \
  --parent-manifest /private/data/snapshots/keller-2016-trials-cas-corrected-v4/manifest.json \
  --identity-evidence-dir /private/data/reviews/keller-identity-2026-09-06 \
  --dataset-version keller-2016-harmonized-audit-v5
```

The command makes no network requests. It validates parent/evidence checksums,
exact queue coverage, affected row counts, taxonomy membership and accepted
reference InChIKeys/stereo with RDKit. The new snapshot is registered in the
existing Data Foundation database with a `KELLER_HARMONIZATION_FINALIZED` event.
Original descriptors/mapping states are retained in `*_before_review` columns;
raw responses, context, sample structures and training masks are unchanged.

**Current gate:** harmonization locked; sample identity and assessment evidence
not universally approved. Snapshot status remains `AUDIT_ONLY`; no retraining
or production promotion is authorized by this release.

## Pretraining readiness after harmonization (2026-09-08)

`audit_keller_readiness.py` audits the harmonized Keller snapshot alongside the
catalog snapshot without pooling their evidence domains or modifying either
release. It produces an immutable report directory with checksummed inputs,
`coverage_113.csv`/JSON, `condition_review.csv`, `collection_priorities.csv`,
`data_gate.json` and a Vietnamese `REPORT_VI.md`.

```bash
.venv-training/bin/python audit_keller_readiness.py \
  --panel-manifest /private/data/snapshots/keller-2016-harmonized-audit-v5/manifest.json \
  --catalog-manifest /private/data/snapshots/clean-master-evidence-audit-v1/manifest.json \
  --output /private/data/reviews/pretrain-readiness-v1
```

The current audit covers all 113 labels and 10,560 source-identity/condition/
descriptor groups. It confirms 72,860 mapped positive trial observations, but
**zero released positive/negative targets eligible for calibration**. Weak
catalog positives and raw panel applicability observations are reported
separately; neither is silently upgraded to reviewed molecular consensus.

Source identity and connectivity counts are distinct from response/assessor
counts. Concentration, solvent and study context remain separate. Assessor IDs
are not exported in reports. Source repeat markers are counted but never
interpreted as independently verified sessions. Presence agreement and intensity
ICC are `NOT_EVALUABLE` (null), not zero, for this unassessed audit release.

Global support gaps to 50 positive/50 negative are collection diagnostics,
not promises of calibration eligibility: the per-label Platt requirement must
be met inside the calibration partition after chemical grouping. Partition
support remains unevaluable until a reviewed training snapshot exists. No new
split, training run, learning curve or production promotion occurs in this audit.

## Next gate

1. Resolve the held identities with source evidence and review measurement
   conditions/consensus. The Keller descriptor harmonization is already locked;
   do not reopen it implicitly or treat it as assessment approval.
2. Release a new training snapshot with assessed negatives and valid intensity;
   keep catalog, individual panel and aggregate domains distinct.
3. Freeze a new benchmark split. CV uses train + validation only and retains
   the original chemical groups. Final calibration uses calibration only.
4. Run fresh baselines and graph models; produce new curves from those runs.
5. Require AP, bootstrap, ECE, Brier, intensity and prospective-panel gates
   before changing the registry pointer.

Current learning curves in the README remain historical weak-label benchmark
results. Vocabulary approval, source permission, evidence review and model
promotion are separate recorded decisions. Academic evidence is not imported
into training by these commands.
