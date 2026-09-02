# Accuracy-first architecture

This release uses a hybrid boundary rather than replacing the application with
DeepChem.

```text
React → FastAPI → application services → olfactory domain
                                  ├─ RDKit/stereo/3D/screening
                                  ├─ reference providers and taxonomy
                                  ├─ MoleculePredictor contract
                                  └─ SQLite/Parquet/FAISS adapters
training environment only → Chemprop and optional DeepChem graph Judge
```

## Model boundary

`olfactory.prediction.MoleculePredictor` is the only model interface used by
the API. `PredictionBatch` carries 113 presence probabilities, optional
intensity, ensemble uncertainty, nearest-training similarity, reliability, and
model/data/calibration versions. `LegacyMorganPredictor` adapts the unchanged
production weights. A graph model must implement the same contract before it
can be loaded in shadow mode.

`EnsemblePredictor` combines independently seeded members by mean probability
and reports their standard deviation as uncertainty; it does not silently
combine incompatible calibration bundles.

## Data and split integrity

The append-only data foundation keeps raw SMILES, isomeric and connectivity
identities, stereo state, source/license, and standardization logs. Unknown
sensory descriptors remain `UNASSESSED`; they are never silently converted to
negative labels. `build_split_manifest.py` writes one immutable SHA-256-bound
70/15/15 split. Connectivity groups are never shared between partitions;
Murcko scaffolds group cyclic molecules and Butina Morgan similarity groups
acyclic molecules. Development folds and three seeds are recorded in the same
manifest.

## DeepChem candidate

`olfactory.training.deepchem_judge` is optional and is not imported by the web
runtime. It uses DeepChem's chiral `MolGraphConvFeaturizer` and a small
edge-aware PyTorch message-passing encoder with masked weighted BCE and Huber
intensity heads. Each run writes `config.json`, `split.json`,
`calibration.json`, `metrics.json`, `weights.pth`, and a checksummed
`manifest.json` below `artifacts/judge/<run_id>/`. It is always `CANDIDATE`;
promotion is an explicit quality-gate operation.

```bash
python3.11 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install -r requirements-deepchem.txt
python build_split_manifest.py --legacy-baseline
python train_deepchem_judge.py --legacy-baseline \
  --split-manifest artifacts/benchmarks/split_manifest.json
```

No artifact overwrites the v1 weights. The registry pointer is the rollback
mechanism; a candidate is not promoted unless macro AP, bootstrap delta, micro
AP retention, calibration, intensity MAE, and prospective sensory gates all
pass on the locked benchmark.

## Academic RAG isolation

The local FAISS RAG package (`olfactory/rag/`) is context retrieval only.
`academic_rag_pipeline.py` remains a CLI-compatible wrapper. Academic
abstracts/full text never enter a training target without a separately
provenanced sensory assessment and snapshot.
