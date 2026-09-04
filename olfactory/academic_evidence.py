"""Academic structure/odor evidence extraction and exact-identity matching.

This module deliberately treats text extraction as *candidate generation*.  A
mention found in a paper is never silently promoted to a verified structure or
to a training label.  Raw values, normalized values, provenance and review
state are retained so a chemist can audit every decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from rdkit import Chem, rdBase

from .features import canonical_isomeric_smiles


ACADEMIC_EVIDENCE_SCHEMA_VERSION = 1
ACADEMIC_EVIDENCE_MANIFEST_VERSION = 1


class EvidenceStatus(str, Enum):
    """Result of comparing a query structure with academic mentions."""

    EXACT_MATCH = "EXACT_MATCH"
    MENTION_ONLY = "MENTION_ONLY"
    NO_EXACT_EVIDENCE = "NO_EXACT_EVIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ReviewState(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class StructureMentionKind(str, Enum):
    SMILES = "SMILES"
    INCHI = "INCHI"
    CAS = "CAS"
    NAME = "NAME"


class MatchLevel(str, Enum):
    EXACT_STEREO = "EXACT_STEREO"
    EXACT_CONNECTIVITY = "EXACT_CONNECTIVITY"
    EXACT_CAS = "EXACT_CAS"
    NAME_ONLY = "NAME_ONLY"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


@dataclass(frozen=True)
class AcademicDocument:
    """Immutable identity/provenance for one indexed paper."""

    paper_id: str
    title: str
    link: str
    source: str
    doi: str
    published_date: str
    content_type: str
    text_sha256: str
    source_type: str = "UNKNOWN"
    license_status: str = "UNVERIFIED"
    open_access: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_sha256"] = self.text_sha256
        return payload


@dataclass(frozen=True)
class NormalizedStructure:
    """Structure identity without any unlogged chemical edits."""

    raw_value: str
    input_kind: str
    isomeric_smiles: Optional[str]
    connectivity_smiles: Optional[str]
    inchikey: Optional[str]
    connectivity_key: Optional[str]
    stereo_state: str
    rdkit_valid: bool
    standardization_log: Tuple[str, ...] = ()
    conflict_flags: Tuple[str, ...] = ()
    review_required: bool = True
    error_code: Optional[str] = None

    @property
    def canonical_isomeric_smiles(self) -> Optional[str]:
        """Alias used by the evidence API and dataset contracts."""
        return self.isomeric_smiles

    @property
    def full_inchikey(self) -> Optional[str]:
        return self.inchikey

    @property
    def connectivity_inchikey_block(self) -> Optional[str]:
        return self.connectivity_key

    @property
    def exact_identity_ready(self) -> bool:
        return bool(
            self.rdkit_valid
            and self.isomeric_smiles
            and self.inchikey
            and not self.review_required
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["standardization_log"] = list(self.standardization_log)
        payload["conflict_flags"] = list(self.conflict_flags)
        payload["canonical_isomeric_smiles"] = self.isomeric_smiles
        payload["full_inchikey"] = self.inchikey
        payload["connectivity_inchikey_block"] = self.connectivity_key
        payload["exact_identity_ready"] = self.exact_identity_ready
        return payload


@dataclass(frozen=True)
class StructureMention:
    """A candidate identifier found in a paper, retaining citation context."""

    mention_id: str
    paper_id: str
    raw_value: str
    kind: str
    title: str = ""
    link: str = ""
    source: str = ""
    doi: str = ""
    content_type: str = "full_text"
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    evidence_excerpt: str = ""
    confidence: float = 0.0
    warnings: Tuple[str, ...] = ()
    normalized: Optional[NormalizedStructure] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        payload["normalized"] = self.normalized.to_dict() if self.normalized else None
        return payload


@dataclass(frozen=True)
class AcademicEvidence:
    """One auditable structure mention and its review status."""

    evidence_id: str
    document: AcademicDocument
    mention: StructureMention
    status: str = EvidenceStatus.MENTION_ONLY.value
    match_level: Optional[str] = None
    odor_descriptors: Tuple[str, ...] = ()
    presence_state: str = "UNASSESSED"
    intensity: Optional[float] = None
    source_type: str = "UNKNOWN"
    review_state: str = ReviewState.UNREVIEWED.value
    conflict_flags: Tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document": self.document.to_dict(),
            "mention": self.mention.to_dict(),
            "status": _enum_value(self.status),
            "match_level": _enum_value(self.match_level) if self.match_level else None,
            "odor_descriptors": list(self.odor_descriptors),
            "presence_state": self.presence_state,
            "intensity": self.intensity,
            "source_type": self.source_type,
            "review_state": self.review_state,
            "conflict_flags": list(self.conflict_flags),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AcademicEvidenceSummary:
    """Response contract for exact-identity academic evidence lookup."""

    query_isomeric_smiles: str
    status: str
    normalized_structure: Optional[NormalizedStructure]
    matches: Tuple[AcademicEvidence, ...] = ()
    conflicts: Tuple[str, ...] = ()
    provenance: Tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_isomeric_smiles": self.query_isomeric_smiles,
            "status": _enum_value(self.status),
            "normalized_structure": (
                self.normalized_structure.to_dict()
                if self.normalized_structure
                else None
            ),
            "matches": [item.to_dict() for item in self.matches],
            "conflicts": list(self.conflicts),
            "provenance": list(self.provenance),
        }


def _stereo_state(molecule: Chem.Mol) -> str:
    potential = Chem.FindPotentialStereo(molecule)
    unresolved = any(str(item.specified).lower().endswith("unspecified") for item in potential)
    defined = any(str(item.specified).lower().endswith("specified") for item in potential)
    if unresolved:
        return "UNRESOLVED"
    if defined:
        return "DEFINED"
    return "ACHIRAL"


def _looks_like_smiles(value: str) -> bool:
    text = value.strip()
    if len(text) < 2 or len(text) > 4096 or any(char.isspace() for char in text):
        return False
    if not re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+", text):
        return False
    if ":" in text:
        return False
    if not re.search(r"(?:Cl|Br|[BCNOPSFIbcnops])", text):
        return False
    # Plain years/IDs should not become candidate molecules.
    if not re.search(r"[A-Za-z]", text):
        return False
    return bool(re.search(r"[0-9\[\]\(\)=#$\\/%.:]", text)) or len(text) <= 4


def _valid_cas(value: str) -> bool:
    match = re.fullmatch(r"(\d{2,7})-(\d{2})-(\d)", value.strip())
    if not match:
        return False
    digits = match.group(1) + match.group(2)
    check = int(match.group(3))
    total = sum(int(digit) * (index + 1) for index, digit in enumerate(reversed(digits)))
    return total % 10 == check


def _infer_kind(raw_value: str) -> StructureMentionKind:
    value = raw_value.strip()
    if value.startswith("InChI=1"):
        return StructureMentionKind.INCHI
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", value):
        return StructureMentionKind.CAS
    if _looks_like_smiles(value):
        return StructureMentionKind.SMILES
    return StructureMentionKind.NAME


def _invalid_structure(
    raw_value: str,
    kind: StructureMentionKind,
    error_code: str,
    flags: Sequence[str] = (),
    warnings: Sequence[str] = (),
) -> NormalizedStructure:
    return NormalizedStructure(
        raw_value=raw_value,
        input_kind=kind.value,
        isomeric_smiles=None,
        connectivity_smiles=None,
        inchikey=None,
        connectivity_key=None,
        stereo_state="UNKNOWN",
        rdkit_valid=False,
        standardization_log=tuple(warnings),
        conflict_flags=tuple(flags),
        review_required=True,
        error_code=error_code,
    )


def normalize_structure(
    raw_value: str,
    *,
    kind: str | StructureMentionKind | None = None,
) -> NormalizedStructure:
    """Normalize a structure without silently changing its chemistry.

    CAS and name mentions remain review-only because an external resolver is
    required before they can be mapped to one exact stereochemical structure.
    Salts/fragments and unresolved stereo are preserved and explicitly flagged.
    """

    raw = _clean(raw_value)
    if not raw:
        return _invalid_structure(raw, StructureMentionKind.NAME, "EMPTY_IDENTIFIER")
    mention_kind = (
        kind
        if isinstance(kind, StructureMentionKind)
        else StructureMentionKind(str(kind).upper())
        if kind is not None
        else _infer_kind(raw)
    )
    if mention_kind is StructureMentionKind.CAS:
        if not _valid_cas(raw):
            return _invalid_structure(raw, mention_kind, "INVALID_CAS", ("INVALID_CAS",))
        return _invalid_structure(
            raw,
            mention_kind,
            "CAS_REQUIRES_STRUCTURE",
            ("CAS_REQUIRES_STRUCTURE_MAPPING",),
        )
    if mention_kind is StructureMentionKind.NAME:
        return _invalid_structure(raw, mention_kind, "NAME_ONLY", ("NAME_ONLY",))

    try:
        with rdBase.BlockLogs():
            molecule = (
                Chem.MolFromInchi(raw, sanitize=True)
                if mention_kind is StructureMentionKind.INCHI
                else Chem.MolFromSmiles(raw, sanitize=True)
            )
        if molecule is None:
            return _invalid_structure(raw, mention_kind, "RDKit_PARSE_ERROR")
        Chem.SanitizeMol(molecule)
        isomeric = canonical_isomeric_smiles(molecule)
        connectivity = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=False,
        )
        round_trip = Chem.MolFromSmiles(isomeric, sanitize=True)
        if round_trip is None or canonical_isomeric_smiles(round_trip) != isomeric:
            return _invalid_structure(
                raw,
                mention_kind,
                "CANONICAL_ROUND_TRIP_MISMATCH",
                ("CANONICAL_ROUND_TRIP_MISMATCH",),
            )
        inchikey = Chem.MolToInchiKey(molecule) or None
        if not inchikey:
            return _invalid_structure(raw, mention_kind, "INCHIKEY_UNAVAILABLE")
    except Exception:
        return _invalid_structure(raw, mention_kind, "RDKit_STANDARDIZATION_ERROR")

    conflicts: list[str] = []
    fragments = len(Chem.GetMolFrags(molecule))
    if fragments != 1:
        conflicts.append("MULTIPLE_FRAGMENTS_PRESERVED")
    if any(atom.GetNumRadicalElectrons() > 0 for atom in molecule.GetAtoms()):
        conflicts.append("RADICAL_PRESERVED")
    stereo_state = _stereo_state(molecule)
    if stereo_state == "UNRESOLVED":
        conflicts.append("UNRESOLVED_STEREO")
    review_required = bool(conflicts)
    log = ["CANONICALIZED_ISOMERIC_SMILES", "RAW_STRUCTURE_PRESERVED"]
    return NormalizedStructure(
        raw_value=raw,
        input_kind=mention_kind.value,
        isomeric_smiles=isomeric,
        connectivity_smiles=connectivity,
        inchikey=inchikey,
        connectivity_key=inchikey.split("-")[0],
        stereo_state=stereo_state,
        rdkit_valid=True,
        standardization_log=tuple(log),
        conflict_flags=tuple(conflicts),
        review_required=review_required,
        error_code=None,
    )


_EXPLICIT_SMILES_RE = re.compile(
    r"(?i)\b(?:isomeric\s+)?smiles?\s*[:=]\s*([^\s,;|<>]+)"
)
_INCHI_RE = re.compile(r"\bInChI=1S?/[A-Za-z0-9@+\-()\[\]=/#,.;\\]+")
_CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_NAME_RE = re.compile(r"(?i)\b(?:compound|odorant|chemical)\s+name\s*[:=]\s*([A-Za-z][A-Za-z0-9()'\- ]{1,79})")
_TOKEN_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+")


def _strip_identifier(value: str) -> str:
    cleaned = value.strip().strip("\"'`“”‘’<>{}")
    return re.sub(r"(?<=[A-Za-z0-9\]\)])[,.;:]+$", "", cleaned)


def _excerpt(text: str, start: int, end: int, radius: int = 140) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())


def _mention(
    raw_value: str,
    kind: StructureMentionKind,
    text: str,
    start: int,
    end: int,
    *,
    paper_id: str,
    title: str,
    link: str,
    source: str,
    doi: str,
    content_type: str,
    confidence: float,
    warnings: Sequence[str] = (),
) -> StructureMention:
    raw = _strip_identifier(raw_value)
    mention_id = hashlib.sha256(
        f"{paper_id}|{content_type}|{kind.value}|{raw}|{start}".encode("utf-8")
    ).hexdigest()[:32]
    normalized = normalize_structure(raw, kind=kind)
    merged_warnings = tuple(dict.fromkeys([*warnings, *normalized.conflict_flags]))
    return StructureMention(
        mention_id=mention_id,
        paper_id=paper_id,
        raw_value=raw,
        kind=kind.value,
        title=title,
        link=link,
        source=source,
        doi=doi,
        content_type=content_type,
        span_start=start,
        span_end=end,
        evidence_excerpt=_excerpt(text, start, end),
        confidence=confidence,
        warnings=merged_warnings,
        normalized=normalized,
    )


def extract_structure_mentions(
    text: str,
    *,
    paper_id: str = "",
    title: str = "",
    link: str = "",
    source: str = "",
    doi: str = "",
    content_type: str = "full_text",
) -> Tuple[StructureMention, ...]:
    """Extract explicit structure identifiers as reviewable candidates.

    The extractor intentionally favors precision of provenance over aggressive
    named-entity guessing.  It recognizes labelled SMILES, InChI, CAS and
    chemistry-like tokens; names are handled only when supplied explicitly to
    :func:`normalize_structure`.
    """

    if not text:
        return ()
    found: list[StructureMention] = []
    seen: set[tuple[str, str]] = set()

    def add(match: re.Match[str], kind: StructureMentionKind, confidence: float) -> None:
        raw = _strip_identifier(match.group(1) if match.lastindex else match.group(0))
        if not raw:
            return
        key = (kind.value, raw)
        if key in seen:
            return
        seen.add(key)
        start = match.start(1) if match.lastindex else match.start()
        end = match.end(1) if match.lastindex else match.end()
        found.append(
            _mention(
                raw,
                kind,
                text,
                start,
                end,
                paper_id=paper_id,
                title=title,
                link=link,
                source=source,
                doi=doi,
                content_type=content_type,
                confidence=confidence,
            )
        )

    for match in _EXPLICIT_SMILES_RE.finditer(text):
        add(match, StructureMentionKind.SMILES, 0.95)
    for match in _INCHI_RE.finditer(text):
        add(match, StructureMentionKind.INCHI, 0.95)
    for match in _CAS_RE.finditer(text):
        add(match, StructureMentionKind.CAS, 0.85 if _valid_cas(match.group(0)) else 0.35)
    for match in _NAME_RE.finditer(text):
        add(match, StructureMentionKind.NAME, 0.40)

    # Unlabelled SMILES occur frequently in tables and supplementary prose.
    for match in _TOKEN_RE.finditer(text):
        raw = _strip_identifier(match.group(0))
        if not _looks_like_smiles(raw):
            continue
        if raw.lower().startswith("inchi="):
            continue
        normalized = normalize_structure(raw, kind=StructureMentionKind.SMILES)
        if not normalized.rdkit_valid:
            continue
        key = (StructureMentionKind.SMILES.value, raw)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            _mention(
                raw,
                StructureMentionKind.SMILES,
                text,
                match.start(),
                match.end(),
                paper_id=paper_id,
                title=title,
                link=link,
                source=source,
                doi=doi,
                content_type=content_type,
                confidence=0.55,
                warnings=("UNLABELLED_TOKEN",),
            )
        )
    return tuple(sorted(found, key=lambda item: (item.span_start or 0, item.mention_id)))


def evidence_records_from_document(
    document: AcademicDocument,
    text: str,
    *,
    source_type: Optional[str] = None,
) -> Tuple[AcademicEvidence, ...]:
    """Build unreviewed evidence candidates for one document."""

    mentions = extract_structure_mentions(
        text,
        paper_id=document.paper_id,
        title=document.title,
        link=document.link,
        source=document.source,
        doi=document.doi,
        content_type=document.content_type,
    )
    records: list[AcademicEvidence] = []
    resolved_source_type = source_type or document.source_type
    for mention in mentions:
        normalized = mention.normalized
        conflicts = list(mention.warnings)
        if normalized and normalized.conflict_flags:
            conflicts.extend(normalized.conflict_flags)
        status = (
            EvidenceStatus.REVIEW_REQUIRED.value
            if not normalized or normalized.review_required or not normalized.rdkit_valid
            else EvidenceStatus.MENTION_ONLY.value
        )
        evidence_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"academic-evidence:{document.paper_id}:{mention.mention_id}")
        )
        records.append(
            AcademicEvidence(
                evidence_id=evidence_id,
                document=document,
                mention=mention,
                status=status,
                source_type=resolved_source_type,
                review_state=ReviewState.UNREVIEWED.value,
                conflict_flags=tuple(dict.fromkeys(conflicts)),
            )
        )
    return tuple(records)


def annotate_chunk_provenance(
    records: Sequence[AcademicEvidence],
    chunks: Iterable[Any],
) -> Tuple[AcademicEvidence, ...]:
    """Attach the first matching chunk index to extracted mentions.

    The PDF extractor owns page boundaries while the RAG splitter owns chunk
    boundaries.  Keeping this small bridge explicit avoids inventing page
    numbers and still gives reviewers a retrievable citation span.
    """
    chunk_texts: list[str] = []
    for chunk in chunks:
        text = getattr(chunk, "page_content", None)
        if text is None and isinstance(chunk, Mapping):
            text = chunk.get("page_content")
        chunk_texts.append(_clean(text))
    annotated: list[AcademicEvidence] = []
    for record in records:
        mention = record.mention
        if mention.chunk_index is None and mention.raw_value:
            for index, chunk_text in enumerate(chunk_texts):
                if mention.raw_value in chunk_text:
                    mention = replace(mention, chunk_index=index)
                    break
        annotated.append(replace(record, mention=mention))
    return tuple(annotated)


def _record_from_dict(payload: Mapping[str, Any]) -> AcademicEvidence:
    document_payload = payload.get("document") or {}
    mention_payload = payload.get("mention") or {}
    normalized_payload = mention_payload.get("normalized")
    normalized = None
    if normalized_payload:
        normalized = NormalizedStructure(
            raw_value=_clean(normalized_payload.get("raw_value")),
            input_kind=_clean(normalized_payload.get("input_kind")),
            isomeric_smiles=normalized_payload.get("isomeric_smiles"),
            connectivity_smiles=normalized_payload.get("connectivity_smiles"),
            inchikey=normalized_payload.get("inchikey"),
            connectivity_key=normalized_payload.get("connectivity_key"),
            stereo_state=_clean(normalized_payload.get("stereo_state")) or "UNKNOWN",
            rdkit_valid=bool(normalized_payload.get("rdkit_valid", False)),
            standardization_log=tuple(normalized_payload.get("standardization_log") or ()),
            conflict_flags=tuple(normalized_payload.get("conflict_flags") or ()),
            review_required=bool(normalized_payload.get("review_required", True)),
            error_code=normalized_payload.get("error_code"),
        )
    mention = StructureMention(
        mention_id=_clean(mention_payload.get("mention_id")),
        paper_id=_clean(mention_payload.get("paper_id")),
        raw_value=_clean(mention_payload.get("raw_value")),
        kind=_clean(mention_payload.get("kind")) or StructureMentionKind.NAME.value,
        title=_clean(mention_payload.get("title")),
        link=_clean(mention_payload.get("link")),
        source=_clean(mention_payload.get("source")),
        doi=_clean(mention_payload.get("doi")),
        content_type=_clean(mention_payload.get("content_type")) or "full_text",
        page=mention_payload.get("page"),
        chunk_index=mention_payload.get("chunk_index"),
        span_start=mention_payload.get("span_start"),
        span_end=mention_payload.get("span_end"),
        evidence_excerpt=_clean(mention_payload.get("evidence_excerpt")),
        confidence=float(mention_payload.get("confidence", 0.0)),
        warnings=tuple(mention_payload.get("warnings") or ()),
        normalized=normalized,
    )
    document = AcademicDocument(
        paper_id=_clean(document_payload.get("paper_id")),
        title=_clean(document_payload.get("title")),
        link=_clean(document_payload.get("link")),
        source=_clean(document_payload.get("source")),
        doi=_clean(document_payload.get("doi")),
        published_date=_clean(document_payload.get("published_date")),
        content_type=_clean(document_payload.get("content_type")) or "full_text",
        text_sha256=_clean(
            document_payload.get("text_sha256")
            or document_payload.get("content_sha256")
        ),
        source_type=_clean(document_payload.get("source_type")) or "UNKNOWN",
        license_status=_clean(document_payload.get("license_status")) or "UNVERIFIED",
        open_access=bool(document_payload.get("open_access", False)),
    )
    return AcademicEvidence(
        evidence_id=_clean(payload.get("evidence_id")),
        document=document,
        mention=mention,
        status=_clean(payload.get("status")) or EvidenceStatus.MENTION_ONLY.value,
        match_level=payload.get("match_level"),
        odor_descriptors=tuple(payload.get("odor_descriptors") or ()),
        presence_state=_clean(payload.get("presence_state")) or "UNASSESSED",
        intensity=payload.get("intensity"),
        source_type=_clean(payload.get("source_type")) or document.source_type,
        review_state=_clean(payload.get("review_state")) or ReviewState.UNREVIEWED.value,
        conflict_flags=tuple(payload.get("conflict_flags") or ()),
        created_at=_clean(payload.get("created_at")) or utc_now(),
    )


class AcademicEvidenceStore:
    """Atomic, local JSONL store for derived evidence candidates."""

    def __init__(self, path: Path | str | None = None):
        configured = os.environ.get("SCENT_STUDIO_ACADEMIC_EVIDENCE_PATH")
        self.path = Path(path or configured or (Path.home() / ".scent-molecule-studio" / "academic_evidence.jsonl")).expanduser()
        self.manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")

    def load(self) -> Tuple[AcademicEvidence, ...]:
        if not self.path.is_file():
            return ()
        records: list[AcademicEvidence] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(_record_from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError):
                # A malformed derived row is ignored; the manifest checksum is
                # still visible to the caller and the source can be rebuilt.
                continue
        return tuple(records)

    def _write(self, records: Sequence[AcademicEvidence], *, dataset_sha256: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        lines = "".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for item in records)
        try:
            temporary.write_text(lines, encoding="utf-8")
            os.replace(temporary, self.path)
            manifest = {
                "schema_version": ACADEMIC_EVIDENCE_MANIFEST_VERSION,
                "evidence_schema_version": ACADEMIC_EVIDENCE_SCHEMA_VERSION,
                "created_at": utc_now(),
                "record_count": len(records),
                "dataset_sha256": dataset_sha256,
                "records_sha256": sha256_text(lines),
            }
            manifest_tmp = self.manifest_path.with_name(f".{self.manifest_path.name}.{uuid.uuid4().hex}.tmp")
            manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(manifest_tmp, self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def assert_compatible(self, dataset_sha256: str) -> None:
        """Fail closed when a store was built from another catalog snapshot."""
        previous_hash = self._manifest_dataset_sha256()
        if not dataset_sha256 or not previous_hash:
            return
        if previous_hash != dataset_sha256:
            raise ValueError(
                "Academic evidence dataset hash mismatch; rebuild the local evidence store."
            )

    def _manifest_dataset_sha256(self) -> str:
        if not self.manifest_path.is_file():
            return ""
        try:
            existing_manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "Academic evidence manifest is unreadable; rebuild the local evidence store."
            ) from error
        return _clean(existing_manifest.get("dataset_sha256"))

    def upsert(
        self,
        records: Iterable[AcademicEvidence],
        *,
        dataset_sha256: str = "",
    ) -> int:
        records = tuple(records)
        if not dataset_sha256:
            dataset_sha256 = self._manifest_dataset_sha256()
        self.assert_compatible(dataset_sha256)
        current = {item.evidence_id: item for item in self.load()}
        added = 0
        for record in records:
            if record.evidence_id not in current:
                added += 1
            current[record.evidence_id] = record
        ordered = tuple(current[key] for key in sorted(current))
        if records or not self.path.exists():
            self._write(ordered, dataset_sha256=dataset_sha256)
        return added

    def set_review_state(
        self,
        evidence_id: str,
        review_state: str | ReviewState,
        *,
        dataset_sha256: str = "",
    ) -> AcademicEvidence:
        """Append a reviewer decision while preserving the original mention.

        The derived JSONL record is replaced atomically; the raw paper text and
        extraction fields are never edited.  This is deliberately a small local
        review seam, not an authorization system.
        """
        state = _enum_value(review_state).upper()
        if state not in {item.value for item in ReviewState}:
            raise ValueError(f"Unknown review state: {review_state}")
        current = self.get(evidence_id)
        if current is None:
            raise KeyError(evidence_id)
        updated = replace(current, review_state=state)
        self.upsert((updated,), dataset_sha256=dataset_sha256)
        return updated

    def get(self, evidence_id: str) -> Optional[AcademicEvidence]:
        return next((item for item in self.load() if item.evidence_id == evidence_id), None)

    def sources(self) -> list[dict[str, Any]]:
        by_paper: dict[str, dict[str, Any]] = {}
        for item in self.load():
            document = item.document
            by_paper.setdefault(
                document.paper_id,
                {
                    "paper_id": document.paper_id,
                    "title": document.title,
                    "link": document.link,
                    "source": document.source,
                    "doi": document.doi,
                    "content_type": document.content_type,
                    "text_sha256": document.text_sha256,
                    "source_type": document.source_type,
                    "license_status": document.license_status,
                    "evidence_count": 0,
                },
            )["evidence_count"] += 1
        return [by_paper[key] for key in sorted(by_paper)]

    def verify(
        self,
        isomeric_smiles: str,
        *,
        include_abstracts: bool = False,
    ) -> AcademicEvidenceSummary:
        query = normalize_structure(isomeric_smiles, kind=StructureMentionKind.SMILES)
        if not query.rdkit_valid or not query.isomeric_smiles:
            raise ValueError(query.error_code or "INVALID_STRUCTURE")
        matches: list[AcademicEvidence] = []
        conflicts: list[str] = []
        provenance: list[dict[str, Any]] = []
        for evidence in self.load():
            if evidence.document.content_type == "abstract" and not include_abstracts:
                continue
            if evidence.review_state == ReviewState.REJECTED.value:
                continue
            candidate = evidence.mention.normalized
            if candidate is None or not candidate.rdkit_valid:
                continue
            level: Optional[MatchLevel] = None
            status = EvidenceStatus.MENTION_ONLY.value
            if (
                candidate.inchikey
                and query.inchikey
                and candidate.exact_identity_ready
                and query.exact_identity_ready
                and candidate.inchikey == query.inchikey
                and candidate.stereo_state != "UNRESOLVED"
                and query.stereo_state != "UNRESOLVED"
            ):
                level = MatchLevel.EXACT_STEREO
                citation_ready = bool(
                    evidence.document.paper_id
                    and (evidence.document.link or evidence.document.doi)
                    and evidence.mention.span_start is not None
                    and evidence.mention.span_end is not None
                    and (
                        evidence.mention.page is not None
                        or evidence.mention.chunk_index is not None
                    )
                )
                license_ready = bool(
                    evidence.document.open_access
                    or evidence.document.license_status.upper()
                    in {"OA_CONFIRMED", "LICENSE_APPROVED", "PRIVATE"}
                )
                if not citation_ready:
                    conflicts.append("MISSING_CITATION_PROVENANCE")
                    status = EvidenceStatus.REVIEW_REQUIRED.value
                elif not license_ready:
                    conflicts.append("LICENSE_NOT_VERIFIED")
                    status = EvidenceStatus.REVIEW_REQUIRED.value
                elif (
                    evidence.document.content_type == "full_text"
                    and evidence.review_state == ReviewState.ACCEPTED.value
                ):
                    status = EvidenceStatus.EXACT_MATCH.value
                else:
                    status = EvidenceStatus.MENTION_ONLY.value
            elif candidate.connectivity_key and candidate.connectivity_key == query.connectivity_key:
                level = MatchLevel.EXACT_CONNECTIVITY
                status = EvidenceStatus.REVIEW_REQUIRED.value
                conflicts.append("CONNECTIVITY_MATCH_STEREO_REVIEW")
            if level is None:
                continue
            matches.append(
                AcademicEvidence(
                    evidence_id=evidence.evidence_id,
                    document=evidence.document,
                    mention=evidence.mention,
                    status=status,
                    match_level=level.value,
                    odor_descriptors=evidence.odor_descriptors,
                    presence_state=evidence.presence_state,
                    intensity=evidence.intensity,
                    source_type=evidence.source_type,
                    review_state=evidence.review_state,
                    conflict_flags=evidence.conflict_flags,
                    created_at=evidence.created_at,
                )
            )
            provenance.append(
                {
                    "paper_id": evidence.document.paper_id,
                    "title": evidence.document.title,
                    "link": evidence.document.link,
                    "doi": evidence.document.doi,
                    "content_type": evidence.document.content_type,
                    "page": evidence.mention.page,
                    "chunk_index": evidence.mention.chunk_index,
                    "text_sha256": evidence.document.text_sha256,
                }
            )
        if any(item.status == EvidenceStatus.EXACT_MATCH.value for item in matches):
            status = EvidenceStatus.EXACT_MATCH.value
        elif any(item.status == EvidenceStatus.REVIEW_REQUIRED.value for item in matches):
            status = EvidenceStatus.REVIEW_REQUIRED.value
        elif matches:
            status = EvidenceStatus.MENTION_ONLY.value
        else:
            status = EvidenceStatus.NO_EXACT_EVIDENCE.value
        return AcademicEvidenceSummary(
            query_isomeric_smiles=query.isomeric_smiles,
            status=status,
            normalized_structure=query,
            matches=tuple(matches),
            conflicts=tuple(dict.fromkeys(conflicts)),
            provenance=tuple(provenance),
        )


class AcademicEvidenceService:
    """Application boundary used by API and offline review tooling."""

    def __init__(self, store: AcademicEvidenceStore | None = None):
        self.store = store or AcademicEvidenceStore()

    def verify(self, isomeric_smiles: str, *, include_abstracts: bool = False) -> AcademicEvidenceSummary:
        return self.store.verify(isomeric_smiles, include_abstracts=include_abstracts)

    def get(self, evidence_id: str) -> Optional[AcademicEvidence]:
        return self.store.get(evidence_id)

    def set_review_state(
        self,
        evidence_id: str,
        review_state: str | ReviewState,
        *,
        dataset_sha256: str = "",
    ) -> AcademicEvidence:
        return self.store.set_review_state(
            evidence_id,
            review_state,
            dataset_sha256=dataset_sha256,
        )

    def sources(self) -> list[dict[str, Any]]:
        return self.store.sources()


def verify_academic_evidence(
    isomeric_smiles: str,
    *,
    store: AcademicEvidenceStore | None = None,
    include_abstracts: bool = False,
) -> AcademicEvidenceSummary:
    """Convenience function for offline callers and API integrations."""
    return AcademicEvidenceService(store).verify(
        isomeric_smiles,
        include_abstracts=include_abstracts,
    )


__all__ = [
    "ACADEMIC_EVIDENCE_SCHEMA_VERSION",
    "ACADEMIC_EVIDENCE_MANIFEST_VERSION",
    "AcademicDocument",
    "AcademicEvidence",
    "AcademicEvidenceService",
    "AcademicEvidenceStore",
    "AcademicEvidenceSummary",
    "EvidenceStatus",
    "MatchLevel",
    "NormalizedStructure",
    "ReviewState",
    "StructureMention",
    "StructureMentionKind",
    "evidence_records_from_document",
    "annotate_chunk_provenance",
    "extract_structure_mentions",
    "normalize_structure",
    "sha256_text",
    "verify_academic_evidence",
]
