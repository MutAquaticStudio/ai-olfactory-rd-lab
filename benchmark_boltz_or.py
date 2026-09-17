#!/usr/bin/env python3
"""Benchmark Boltz receptor-ligand binding evidence without touching production ranking.

The default operation estimates cost only. Real compute requires --execute and a
BOLTZ_API_KEY. A deterministic idempotency key is used per normalized receptor /
ligand pair so a retry does not accidentally create duplicate compute.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping

from olfactory.receptor_binding import (
    BoltzReceptorBindingProvider,
    ReceptorBindingError,
    build_boltz_provider_from_env,
    normalize_isomeric_smiles,
    normalize_receptor_sequence,
)


REQUIRED_COLUMNS = ("receptor_id", "receptor_sequence", "compound_id", "smiles")
OPTIONAL_COLUMNS = ("experimental_state", "experimental_value", "experimental_unit", "source_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate or execute research-only Boltz olfactory-receptor binding predictions."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV with receptor/ligand pairs")
    parser.add_argument("--output", required=True, type=Path, help="JSONL output path")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit paid/live or synthetic/test compute after estimating all rows.",
    )
    parser.add_argument(
        "--max-total-cost-usd",
        type=Decimal,
        default=Decimal("1.00"),
        help="Hard local cost ceiling for --execute. Default: 1.00 USD.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-wait-seconds", type=float, default=900.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a small smoke benchmark.",
    )
    return parser.parse_args()


def load_rows(path: Path, *, limit: int | None = None) -> List[Dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Input file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError("Missing required CSV columns: " + ", ".join(missing))
        rows: List[Dict[str, str]] = []
        for index, raw in enumerate(reader, start=2):
            row = {key: str(value or "").strip() for key, value in raw.items() if key is not None}
            if not any(row.values()):
                continue
            row["_source_row"] = str(index)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Input CSV contains no benchmark rows.")
    return rows


def validate_rows(rows: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        source_row = row.get("_source_row", "?")
        receptor_id = row.get("receptor_id", "").strip()
        compound_id = row.get("compound_id", "").strip()
        if not receptor_id or not compound_id:
            raise ValueError(f"Row {source_row}: receptor_id and compound_id are required.")
        sequence = normalize_receptor_sequence(row.get("receptor_sequence", ""))
        smiles = normalize_isomeric_smiles(row.get("smiles", ""))
        dedupe_key = (receptor_id, sequence, smiles)
        if dedupe_key in seen:
            raise ValueError(
                f"Row {source_row}: duplicate normalized receptor/ligand pair for {receptor_id} / {compound_id}."
            )
        seen.add(dedupe_key)
        item = {key: row.get(key, "") for key in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
        item["receptor_sequence"] = sequence
        item["smiles"] = smiles
        item["source_row"] = source_row
        normalized.append(item)
    return normalized


def parse_cost(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Boltz returned an invalid cost estimate: {value!r}") from error


def estimate_all(
    provider: BoltzReceptorBindingProvider,
    rows: Iterable[Mapping[str, str]],
) -> tuple[List[Dict[str, object]], Decimal]:
    estimates: List[Dict[str, object]] = []
    total = Decimal("0")
    for row in rows:
        estimate = provider.estimate_cost(row["receptor_sequence"], row["smiles"])
        cost = parse_cost(estimate["estimated_cost_usd"])
        total += cost
        estimates.append(
            {
                "receptor_id": row["receptor_id"],
                "compound_id": row["compound_id"],
                "source_row": row["source_row"],
                "estimated_cost_usd": str(cost),
                "idempotency_key": estimate["idempotency_key"],
                "mode": estimate["mode"],
                "breakdown": estimate.get("breakdown"),
                "disclaimer": estimate.get("disclaimer"),
            }
        )
    return estimates, total


def run_all(
    provider: BoltzReceptorBindingProvider,
    rows: Iterable[Mapping[str, str]],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for row in rows:
        started = provider.start(row["receptor_sequence"], row["smiles"])
        resource_id = str(started.get("provider_resource_id") or "")
        if not resource_id:
            raise ReceptorBindingError(
                "BOLTZ_INVALID_RESPONSE",
                "Boltz start response did not include a resource ID.",
            )
        if str(started.get("status") or "").lower() in {"succeeded", "failed"}:
            evidence = started
        else:
            evidence = provider.wait(
                resource_id,
                poll_seconds=ARGS.poll_seconds,
                max_wait_seconds=ARGS.max_wait_seconds,
            )
        results.append(
            {
                "receptor_id": row["receptor_id"],
                "compound_id": row["compound_id"],
                "smiles": row["smiles"],
                "experimental_state": row.get("experimental_state") or None,
                "experimental_value": row.get("experimental_value") or None,
                "experimental_unit": row.get("experimental_unit") or None,
                "source_id": row.get("source_id") or None,
                "source_row": row["source_row"],
                "evidence": evidence,
            }
        )
    return results


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    try:
        rows = validate_rows(load_rows(ARGS.input, limit=ARGS.limit))
        provider = build_boltz_provider_from_env()
        if provider is None:
            raise ValueError(
                "BOLTZ_API_KEY is not configured. Use a workspace test key first; do not commit the key."
            )
        estimates, total_cost = estimate_all(provider, rows)
        print(
            json.dumps(
                {
                    "rows": len(rows),
                    "provider": provider.metadata,
                    "estimated_total_cost_usd": str(total_cost),
                    "execute": ARGS.execute,
                },
                indent=2,
            )
        )
        if not ARGS.execute:
            write_jsonl(ARGS.output, ({"estimate": item} for item in estimates))
            print(f"Cost estimates written to {ARGS.output}")
            return 0
        if ARGS.max_total_cost_usd < 0:
            raise ValueError("--max-total-cost-usd must be non-negative.")
        if total_cost > ARGS.max_total_cost_usd:
            raise ValueError(
                f"Estimated total {total_cost} USD exceeds local ceiling "
                f"{ARGS.max_total_cost_usd} USD; no compute was submitted."
            )
        results = run_all(provider, rows)
        write_jsonl(ARGS.output, results)
        print(f"Research evidence written to {ARGS.output}")
        return 0
    except (ValueError, ReceptorBindingError) as error:
        code = getattr(error, "code", "BENCHMARK_INPUT_ERROR")
        print(f"{code}: {error}", file=sys.stderr)
        return 2


ARGS = parse_args()

if __name__ == "__main__":
    raise SystemExit(main())
