"""Audit artifacts for an evidence release; never starts model training."""
from __future__ import annotations

from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..data_foundation.evidence import (
    audit_target_table, content_hash, evidence_coverage, file_hash, load_evidence_snapshot,
)
from .benchmark import build_benchmark_manifest, save_immutable_manifest, assert_no_leakage


def split_audit(table, split: dict, molecules: pd.DataFrame) -> dict:
    assert_no_leakage(split)
    with rdBase.BlockLogs():
        mols = [Chem.MolFromSmiles(smiles) for smiles in table.smiles]
        keys = [Chem.MolToInchiKey(mol).split("-")[0] for mol in mols]
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) for mol in mols]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fingerprints = [generator.GetFingerprint(mol) for mol in mols]
    partitions = {name: split[field] for name, field in (
        ("train", "train_indices"), ("calibration", "calibration_indices"),
        ("validation", "validation_indices"), ("locked_test", "test_indices"))}
    summary, membership, overlap, similarity = {}, [], {}, {}
    ratios = dict(zip(partitions, (0.60, 0.10, 0.15, 0.15)))
    key_rows = molecules.groupby("inchikey").record_id.agg(list).to_dict()
    for name, indices in partitions.items():
        summary[name] = {
            "rows": len(indices), "groups": len({split["group_ids"][i] for i in indices}),
            "target_rows": ratios[name] * len(mols), "actual_ratio": len(indices) / len(mols),
            "deviation_rows": len(indices) - ratios[name] * len(mols),
            "label_support": {label: {"present": int((table.presence[indices, j] == 1).sum()),
                                      "absent": int((table.presence[indices, j] == 0).sum()),
                                      "unassessed": int(np.isnan(table.presence[indices, j]).sum())}
                              for j, label in enumerate(table.label_names)},
        }
        for index in indices:
            full_key = Chem.MolToInchiKey(mols[index])
            membership.append({"index": index, "isomeric_smiles": table.smiles[index],
                               "inchikey": full_key, "connectivity_key": keys[index],
                               "scaffold": scaffolds[index], "group_id": split["group_ids"][index],
                               "partition": name, "source_records": key_rows[full_key]})
    for left, right in itertools.combinations(partitions, 2):
        li, ri = partitions[left], partitions[right]
        key = left + ":" + right
        overlap[key] = {
            "connectivity": len({keys[i] for i in li} & {keys[i] for i in ri}),
            "scaffold": len(({scaffolds[i] for i in li} & {scaffolds[i] for i in ri}) - {""}),
            "group": len({split["group_ids"][i] for i in li} & {split["group_ids"][i] for i in ri}),
        }
        if any(overlap[key].values()):
            raise ValueError("Chemical leakage in split audit")
        nearest, pair_count = [], 0
        right_fps = [fingerprints[i] for i in ri]
        for index in li:
            values = DataStructs.BulkTanimotoSimilarity(fingerprints[index], right_fps)
            pair_count += sum(value >= 0.60 for value in values)
            if values:
                nearest.append(max(values))
        similarity[key] = {"pairs_ge_0_60": pair_count, "max_tanimoto": max(nearest) if nearest else None,
                           "left_nearest_median": float(np.median(nearest)) if nearest else None}
    counts = Counter(split["group_ids"])
    return {"partitions": summary, "membership": sorted(membership, key=lambda item: item["index"]),
            "connectivity_groups": len(set(keys)), "murcko_scaffolds": len(set(scaffolds) - {""}),
            "acyclic_clusters": len({g for g in counts if g.startswith("acyclic:")}),
            "largest_group": max(counts.values()), "overlap": overlap, "cross_partition_similarity": similarity,
            "rdkit_version": rdBase.rdkitVersion,
            "grouping_config": {"morgan_radius": 2, "bits": 2048, "chirality": True,
                                "butina_distance_cutoff": 0.4, "connectivity_representatives": True},
            "note": "Butina cluster separation is not an all-pairs similarity cutoff."}


def build_release_audit(snapshot: Path, output: Path, seed: int = 42) -> dict:
    manifest, molecules, observations = load_evidence_snapshot(snapshot)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    coverage = evidence_coverage(manifest, molecules, observations)
    table = audit_target_table(manifest, molecules, observations)
    split = build_benchmark_manifest(table, dataset_version=manifest["dataset_version"], seed=seed)
    split.update({"purpose": "AUDIT_ONLY", "training_eligible": False,
                  "snapshot_manifest_sha256": manifest["manifest_sha256"],
                  "snapshot_files": manifest["files"]})
    split.pop("manifest_sha256")
    # Benchmark manifests use ASCII-normalized JSON for their existing checksum contract.
    import hashlib
    split["manifest_sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    leakage = split_audit(table, split, molecules)
    partition_coverage = {}
    for partition, details in leakage["partitions"].items():
        identities = {row["inchikey"] for row in leakage["membership"] if row["partition"] == partition}
        partition_coverage[partition] = evidence_coverage(
            manifest, molecules[molecules.inchikey.isin(identities)], observations[observations.inchikey.isin(identities)])
    output.mkdir(parents=True)
    save_immutable_manifest(output / "split_manifest.json", split)
    for name, value in (("coverage.json", coverage), ("split_report.json", leakage),
                        ("partition_coverage.json", partition_coverage)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    pd.DataFrame(coverage["labels"]).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(leakage["membership"]).to_csv(output / "membership.csv", index=False)
    queue = pd.DataFrame({"descriptor": coverage["collection_queue"], "reviewer": "",
                          "evidence_link": "", "presence_state": "UNASSESSED",
                          "measurement_context": "", "intensity_scale": "", "review_decision": "PENDING"})
    queue.to_csv(output / "collection_queue.csv", index=False)
    lines = ["# Judge data release audit", "", f"Dataset: `{manifest['dataset_version']}`", "",
             f"Source rows: {len(molecules)}; unique molecules: {coverage['unique_molecules']}; "
             f"connectivity groups: {coverage['connectivity_groups']}.", "",
             "Status: AUDIT_ONLY. Training and production promotion are not authorized by this release.", "",
             "| Partition | Molecules | Groups | Target rows | Deviation |", "|---|---:|---:|---:|---:|"]
    for name, row in leakage["partitions"].items():
        lines.append(f"| {name} | {row['rows']} | {row['groups']} | {row['target_rows']:.1f} | {row['deviation_rows']:+.1f} |")
    lines += ["", "Connectivity, scaffold and group overlap: zero.", "",
              "Cross-partition similarity remains measurable; see split_report.json.", "",
              "## Data gate", "", *[f"- {reason}" for reason in coverage["blocked_reasons"]], "",
              "Next: source-specific panel import and human evidence review. No new learning curve was generated."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    result = {"snapshot_manifest_sha256": manifest["manifest_sha256"],
              "files": {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()},
              "training_eligible": False}
    (output / "audit_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
