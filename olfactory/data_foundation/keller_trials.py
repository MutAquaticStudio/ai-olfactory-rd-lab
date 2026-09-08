"""Recover source trial provenance without imputing ratings or stereochemistry."""
from __future__ import annotations

from contextlib import closing
import json
import math
import re

import pandas as pd
from openpyxl import load_workbook

from .evidence import content_hash


DESCRIPTORS = (
    "EDIBLE", "BAKERY", "SWEET", "FRUIT", "FISH", "GARLIC", "SPICES", "COLD",
    "SOUR", "BURNT", "ACID", "WARM", "MUSKY", "SWEATY", "AMMONIA/URINOUS",
    "DECAYED", "WOOD", "GRASS", "FLOWER", "CHEMICAL",
)
OTHER_MEASURES = {
    "CAN OR CAN'T SMELL": "DETECTION",
    "KNOW OR DON'T KNOW THE SMELL": "RECOGNITION",
    "THE ODOR IS:": "FREE_TEXT",
    "HOW STRONG IS THE SMELL?": "OVERALL_INTENSITY",
    "HOW PLEASANT IS THE SMELL?": "PLEASANTNESS",
    "HOW FAMILIAR IS THE SMELL?": "FAMILIARITY",
}
MEASURES = (*OTHER_MEASURES, *DESCRIPTORS)
IDENTIFIERS = ("C.A.S.", "Catalogue #*", "CID", "Odor", "Odor dilution",
               "Subject # (this study)", "VIAL #")


def _text(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return format(value, ".15g")
    return str(value)


def _original_measurements(path, checksum):
    """Read only needed columns; no demographics or inferred session order."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if "data" not in workbook.sheetnames:
            raise ValueError("Keller workbook schema mismatch: missing data sheet")
        rows = workbook["data"].iter_rows(values_only=True)
        next(rows, None)
        next(rows, None)
        headers = [str(value).strip() for value in next(rows, ())]
        if len(headers) != len(set(headers)) or not set((*IDENTIFIERS, *MEASURES)) <= set(headers):
            raise ValueError("Keller workbook schema mismatch: required columns")
        seen, vials = set(), {}
        for excel_row, cells in enumerate(rows, start=4):
            if all(value is None for value in cells):
                continue
            record = dict(zip(headers, cells))
            values = {key: _text(record.get(key)) for key in (*IDENTIFIERS, *MEASURES)}
            if any(not values[key] for key in ("C.A.S.", "CID", "Odor", "Odor dilution",
                                                "Subject # (this study)", "VIAL #")):
                raise ValueError(f"Keller trial identity mismatch at Excel row {excel_row}")
            subject, vial = values["Subject # (this study)"], values["VIAL #"]
            if (subject, vial) in seen:
                raise ValueError(f"Keller duplicate trial identity mismatch at Excel row {excel_row}")
            seen.add((subject, vial))
            ratio = values["Odor dilution"].replace(",", "")
            identity = (values["CID"], values["C.A.S."], ratio)
            if vial in vials and vials[vial] != identity:
                raise ValueError(f"Keller vial identity mismatch at Excel row {excel_row}")
            vials[vial] = identity
            metadata = {
                "trial_id": content_hash({"study": "keller_2016", "workbook_sha256": checksum,
                                          "subject": subject, "vial": vial}),
                "source_excel_row": str(excel_row), "source_sheet": "data", "vial_id": vial,
                "trial_identity_status": "RECOVERED_FROM_ORIGINAL_WORKBOOK",
                "raw_source_identifier": values["CID"], "source_cas": values["C.A.S."],
                "source_odor_name": values["Odor"], "source_catalog_number": values["Catalogue #*"],
                # A marker identifies a source-designated repeat, not visit/chronological order.
                "source_replicate_marker": "true" if "(replicate)" in values["Odor"].lower() else "false",
                "detection_response": values["CAN OR CAN'T SMELL"],
            }
            for term in MEASURES:
                yield metadata, subject, ratio, term, values[term]
    finally:
        workbook.close()


def recover_keller_trials(frames, path, checksum):
    """Fail closed unless every processed response aligns with the pinned original.

    Ordinal alignment is only accepted after checking value, term, subject,
    source identifier and concentration on every row. Truncation is an error.
    """
    with closing(_original_measurements(path, checksum)) as original, closing(frames):
        for frame in frames:
            recovered = []
            for row in frame.to_dict("records"):
                source = next(original, None)
                if source is None:
                    raise ValueError("Keller response mismatch: processed table has extra rows")
                metadata, subject, ratio, term, value = source
                context = json.loads(row["context"])
                if (row["source_term"] != term or _text(row["raw_value"]) != value
                        or _text(row["assessor_id"]) != subject
                        or _text(row["source_identifier"]) != metadata["raw_source_identifier"]
                        or context.get("ratio") != ratio):
                    raise ValueError(f"Keller response mismatch at processed row {row['source_row']}")
                numerator, denominator = (float(part) for part in ratio.split("/"))
                concentration = context.get("concentration")
                if denominator <= 0 or not isinstance(concentration, (int, float)) or not math.isclose(
                        concentration, numerator / denominator, rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(f"Keller concentration mismatch at processed row {row['source_row']}")
                row.update(metadata)
                kind = "ODOR_DESCRIPTOR" if term in DESCRIPTORS else OTHER_MEASURES[term]
                row["measurement_kind"] = kind
                row["blank_reason"] = None
                if value is None:
                    row["blank_reason"] = (
                        "SKIPPED_NO_DETECTION" if metadata["detection_response"] == "I can't smell anything"
                        else "UNRECORDED_DESCRIPTOR_VALUE" if kind == "ODOR_DESCRIPTOR"
                        else "UNRECORDED_VALUE"
                    )
                warnings = [warning for warning in row["warning"].split(";") if warning]
                if re.fullmatch(r"\d{2,7}-\d{2}-\d", metadata["raw_source_identifier"]):
                    row["identifier_type"] = "CAS_IN_SOURCE_CID_FIELD"
                    warnings.append("SOURCE_CID_IS_CAS_REVIEW_REQUIRED")
                if row["stereo_state"] == "UNRESOLVED":
                    warnings.append("STEREO_IDENTITY_REVIEW_REQUIRED")
                if metadata["detection_response"] not in ("I smell something", "I can't smell anything"):
                    warnings.append("DETECTION_RESPONSE_REVIEW_REQUIRED")
                if kind in ("DETECTION", "RECOGNITION", "FREE_TEXT"):
                    # A subject's free-text "1" is not a numerical sensory rating.
                    row["raw_numeric_value"] = None
                    row["scale"] = None
                elif value is not None and metadata["detection_response"] == "I can't smell anything":
                    warnings.append("RATING_WITHOUT_DETECTION_REVIEW_REQUIRED")
                if term == "MUSKY":
                    row["descriptor"] = None
                    row["mapping_state"] = "REVIEW_REQUIRED"
                row["warning"] = ";".join(dict.fromkeys(warnings))
                recovered.append(row)
            yield pd.DataFrame(recovered)
        if next(original, None) is not None:
            raise ValueError("Keller response mismatch: processed table is truncated")
