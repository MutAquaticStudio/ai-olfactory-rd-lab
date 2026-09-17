# Boltz receptor-binding research gate

## Decision

Boltz is integrated as an **external research-evidence provider**, not as a
production odor model and not as a candidate-ranking signal.

The first question is empirical: **does Boltz-2.1 receptor-ligand binding output
carry useful signal against experimentally characterized olfactory-receptor
pairs?** Until that is demonstrated, the feature remains outside the FastAPI web
runtime and outside `MoleculePredictor`.

This boundary preserves the existing evidence model:

- odor prediction remains model output;
- chemistry screening remains deterministic configured rules;
- catalog/reference verification remains identity evidence;
- sensory records remain experimental evidence;
- Boltz is a fifth, explicitly labelled **predicted receptor-binding** channel.

A high Boltz `binding_confidence` does **not** establish receptor activation,
agonism, antagonism, odor quality, intensity, threshold, safety, or sensory
validation. `optimization_score` is retained as a provider ranking score and is
not converted to Kd, EC50, or another physical assay value.

## External-data boundary

Both cost estimation and prediction submit the receptor amino-acid sequence and
ligand SMILES to `api.boltz.bio`. The CLI therefore requires
`--consent-external-boltz` before making either request.

Sensory observations, assessor data, private panel records, formula data, and
other local R&D data are not part of the request.

Use a workspace-scoped **test** key first. Boltz test mode is appropriate for
checking authentication, payloads, job lifecycle, normalization, and failure
handling; its returned metrics are synthetic and must not be used for scientific
evaluation.

## Configuration

```bash
export BOLTZ_API_KEY="sk_bc_ws_test_..."
export BOLTZ_BASE_URL="https://api.boltz.bio"
export BOLTZ_REQUEST_TIMEOUT_SECONDS="30"
```

Do not store an API key in `.env.example`, Git, benchmark CSV files, JSONL
outputs, notebooks, or screenshots.

## Benchmark input

Start from `examples/boltz_or_benchmark_template.csv`.

Required columns:

| Column | Meaning |
| --- | --- |
| `receptor_id` | Stable receptor identifier, e.g. a reviewed human OR symbol/accession |
| `receptor_sequence` | Reviewed one-letter amino-acid sequence |
| `compound_id` | Stable local/source compound identifier |
| `smiles` | Ligand SMILES; normalized to canonical isomeric SMILES before submission |

Optional ground-truth columns:

| Column | Meaning |
| --- | --- |
| `experimental_state` | Source-defined state such as `ACTIVE` / `INACTIVE` after protocol review |
| `experimental_value` | Numeric assay value when comparable within that source/protocol |
| `experimental_unit` | Unit for the experimental value |
| `source_id` | DOI, dataset row, assay ID, or other provenance pointer |

Do not merge heterogeneous assay definitions into one binary target without a
reviewed mapping. Binding, activation, response amplitude, and threshold assays
are different measurements.

## Cost-first workflow

Cost estimation is the default and still requires external-data consent:

```bash
python benchmark_boltz_or.py \
  --input path/to/or_pairs.csv \
  --output artifacts/benchmarks/boltz_or_estimates.jsonl \
  --consent-external-boltz
```

Run a synthetic test-key lifecycle:

```bash
python benchmark_boltz_or.py \
  --input path/to/or_pairs.csv \
  --output artifacts/benchmarks/boltz_or_test.jsonl \
  --consent-external-boltz \
  --execute \
  --limit 5 \
  --max-total-cost-usd 0
```

Test-mode estimates may be synthetic. If the provider reports a non-zero test
estimate, raise the local ceiling deliberately rather than removing the gate.

A live key is blocked from compute unless `--allow-live` is also supplied:

```bash
python benchmark_boltz_or.py \
  --input path/to/or_pairs.csv \
  --output artifacts/benchmarks/boltz_or_live.jsonl \
  --consent-external-boltz \
  --execute \
  --allow-live \
  --max-total-cost-usd 5.00
```

The CLI estimates the complete selected batch before submitting any prediction.
If the total estimate exceeds the local ceiling, no prediction is submitted.
Identical normalized receptor/ligand requests use deterministic idempotency keys
to reduce duplicate billing on retries.

## Normalized evidence schema

Provider responses are mapped into a stable research record:

```text
PREDICTED_RECEPTOR_BINDING
├─ provider / resource ID
├─ model / model version / live mode
├─ binding_confidence
├─ optimization_score
├─ structure_confidence
├─ pTM / ipTM / ligand ipTM
├─ complex pLDDT / interface pLDDT
├─ predicted-distance-error metrics
├─ expiring structure artifact URLs
├─ SHA-256 fingerprints of submitted sequence and SMILES
└─ explicit scientific limitations
```

Raw receptor sequences are not copied into normalized result records. Boltz's
expiring artifact URLs are evidence pointers, not durable local storage.

## Scientific benchmark gate

Do not judge usefulness from a handful of attractive complexes. Build a
receptor/ligand benchmark with reviewed experimental provenance and include both
positive and negative/low-response examples where the source protocol supports
them.

Minimum reporting should include:

1. AUROC and, more importantly for sparse positives, AUPRC for
   `binding_confidence` against the reviewed binary assay target.
2. Precision@K / recall@K per receptor for screening use cases.
3. Bootstrap confidence intervals grouped by ligand connectivity, not raw rows.
4. Performance by receptor, ligand scaffold/connectivity, stereochemical status,
   and nearest-known-ligand similarity.
5. Failure/missing rate and Boltz structure/interface confidence distributions.
6. If a continuous activation measurement is scientifically comparable within a
   source, report rank correlation separately; do not reinterpret
   `optimization_score` as the assay value.

At least one held-out evaluation must prevent the same ligand connectivity from
appearing on both sides. A random pair split can leak nearly identical ligands
and overstate practical screening performance.

### Promotion decision

A web/API feature may be added only after the benchmark establishes that the
binding signal is useful for a concrete workflow. Even after promotion it should
initially remain a separate research-evidence panel and must not modify odor
probabilities or candidate target-fit ranking.

Promotion into ranking requires a second decision with prospective validation;
that decision is intentionally outside this integration.

## Future library screening

Boltz also exposes a small-molecule library-screen API. Do not switch the current
candidate pipeline to it yet.

The library-screen defaults include Boltz SMARTS structural filtering designed
around drug-discovery concerns. Fragrance molecules have a different objective
function, so measure the filter's effect on known odorants before adopting it as
a fragrance chemistry gate. Do not silently disable provider safety filters
either: any alternative filter policy must be explicit, versioned, benchmarked,
and reviewed separately from the existing local chemistry screen.

If the single-pair benchmark passes, library screening is the next sensible
experiment because it can rank many odorants against a receptor while preserving
Boltz as an independent evidence channel.
