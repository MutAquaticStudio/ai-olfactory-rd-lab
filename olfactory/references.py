"""Licensed reference-source checks for structural and catalog evidence.

This module deliberately does not scrape TGSC or ScenTree. Catalog providers are
enabled only by an operator-supplied, checksum-verified snapshot manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Sequence, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
from rdkit import Chem, rdBase

from .pubchem import NoveltyResult, NoveltyStatus, PubChemClient


PUBCHEM = "PUBCHEM"
TGSC = "TGSC"
SCENTREE = "SCENTREE"


class ReferenceStatus(str, Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNVERIFIED = "UNVERIFIED"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class MatchLevel(str, Enum):
    EXACT_STEREO = "EXACT_STEREO"
    EXACT_CONNECTIVITY = "EXACT_CONNECTIVITY"
    EXACT_CAS = "EXACT_CAS"
    NAME_ONLY = "NAME_ONLY"


class ReferenceGateStatus(str, Enum):
    PASS = "PASS"
    BLOCKED_MATCH = "BLOCKED_MATCH"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class ReferenceQuery:
    isomeric_smiles: str
    inchi_key: Optional[str]
    connectivity_key: Optional[str]
    cas: Optional[str] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class ReferenceProviderMetadata:
    provider: str
    display_name: str
    source_type: str
    enabled: bool
    external: bool
    query_types: Tuple[str, ...]
    dataset_version: Optional[str]
    license_status: str
    configuration_error: Optional[str] = None


@dataclass(frozen=True)
class ReferenceEvidence:
    provider: str
    status: ReferenceStatus
    match_level: Optional[MatchLevel]
    queried_identifier: Optional[str]
    record_ids: Tuple[str, ...] = ()
    record_urls: Tuple[str, ...] = ()
    checked_at: Optional[str] = None
    source_version: Optional[str] = None
    error_code: Optional[str] = None


@dataclass(frozen=True)
class ReferenceGate:
    status: ReferenceGateStatus
    blocking_providers: Tuple[str, ...] = ()
    reason_code: Optional[str] = None


@dataclass(frozen=True)
class ReferenceCheckBundle:
    evidences: Tuple[ReferenceEvidence, ...]
    gate: ReferenceGate

    def evidence_for(self, provider: str) -> Optional[ReferenceEvidence]:
        normalized = provider.upper()
        return next(
            (item for item in self.evidences if item.provider == normalized),
            None,
        )

    def pubchem_novelty(self, isomeric_smiles: str) -> NoveltyResult:
        """Compatibility projection for the v0.5 candidate payload."""
        evidence = self.evidence_for(PUBCHEM)
        if evidence is None or evidence.status in {
            ReferenceStatus.UNVERIFIED,
            ReferenceStatus.NOT_CONFIGURED,
            ReferenceStatus.AMBIGUOUS,
        }:
            return NoveltyResult(
                NoveltyStatus.UNVERIFIED,
                isomeric_smiles,
                error_code=evidence.error_code if evidence else "REFERENCE_NOT_RUN",
            )
        if evidence.status is ReferenceStatus.MATCH:
            cids = tuple(
                int(value)
                for value in evidence.record_ids
                if str(value).isdigit()
            )
            return NoveltyResult(NoveltyStatus.FOUND, isomeric_smiles, cids=cids)
        return NoveltyResult(NoveltyStatus.NOT_FOUND, isomeric_smiles)


class ReferenceProvider(Protocol):
    @property
    def metadata(self) -> ReferenceProviderMetadata:
        ...

    def check(self, query: ReferenceQuery, *, consent: bool) -> ReferenceEvidence:
        ...


def reference_query_from_smiles(isomeric_smiles: str) -> ReferenceQuery:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(isomeric_smiles, sanitize=True)
    if molecule is None:
        raise ValueError("Reference query requires a valid Isomeric SMILES")
    inchi_key: Optional[str]
    try:
        inchi_key = Chem.MolToInchiKey(molecule) or None
    except Exception:
        inchi_key = None
    connectivity_key = inchi_key.split("-")[0] if inchi_key else None
    return ReferenceQuery(
        isomeric_smiles=Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        ),
        inchi_key=inchi_key,
        connectivity_key=connectivity_key,
    )


class PubChemProvider:
    def __init__(self, client: PubChemClient) -> None:
        self.client = client
        self._metadata = ReferenceProviderMetadata(
            provider=PUBCHEM,
            display_name="PubChem/NCBI",
            source_type="STRUCTURAL_IDENTITY",
            enabled=True,
            external=True,
            query_types=("ISOMERIC_SMILES",),
            dataset_version=None,
            license_status="PUBLIC_API",
        )

    @property
    def metadata(self) -> ReferenceProviderMetadata:
        return self._metadata

    def check(self, query: ReferenceQuery, *, consent: bool) -> ReferenceEvidence:
        result = self.client.verify(query.isomeric_smiles, consent=consent)
        if result.status is NoveltyStatus.FOUND:
            return ReferenceEvidence(
                provider=PUBCHEM,
                status=ReferenceStatus.MATCH,
                match_level=MatchLevel.EXACT_STEREO,
                queried_identifier=query.isomeric_smiles,
                record_ids=tuple(str(cid) for cid in result.cids),
                source_version="PUG_REST_same_stereo_isotope",
            )
        if result.status is NoveltyStatus.NOT_FOUND:
            return ReferenceEvidence(
                provider=PUBCHEM,
                status=ReferenceStatus.NO_MATCH,
                match_level=None,
                queried_identifier=query.isomeric_smiles,
                source_version="PUG_REST_same_stereo_isotope",
            )
        return ReferenceEvidence(
            provider=PUBCHEM,
            status=ReferenceStatus.UNVERIFIED,
            match_level=None,
            queried_identifier=query.isomeric_smiles,
            source_version="PUG_REST_same_stereo_isotope",
            error_code=result.error_code,
        )


@dataclass(frozen=True)
class _SnapshotRecord:
    record_id: str
    record_url: Optional[str]
    inchi_key: Optional[str]
    connectivity_key: Optional[str]
    cas: Optional[str]
    normalized_name: Optional[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_optional(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return normalized or None


def _normalize_cas(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = re.sub(r"\s+", "", value)
    return normalized if re.fullmatch(r"\d{2,7}-\d{2}-\d", normalized) else None


def _normalize_record_url(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


class LicensedSnapshotProvider:
    """Exact identifier lookup over an operator-approved local snapshot."""

    def __init__(
        self,
        provider: str,
        display_name: str,
        manifest_path: Optional[Path],
    ) -> None:
        self.provider = provider.upper()
        self.display_name = display_name
        self._records: Tuple[_SnapshotRecord, ...] = ()
        self._indexes: Dict[str, Dict[str, Tuple[_SnapshotRecord, ...]]] = {}
        self._metadata = self._load(manifest_path)

    @property
    def metadata(self) -> ReferenceProviderMetadata:
        return self._metadata

    def _disabled(
        self,
        license_status: str,
        error: Optional[str] = None,
    ) -> ReferenceProviderMetadata:
        return ReferenceProviderMetadata(
            provider=self.provider,
            display_name=self.display_name,
            source_type="FRAGRANCE_CATALOG",
            enabled=False,
            external=False,
            query_types=("FULL_INCHIKEY", "CONNECTIVITY_INCHIKEY", "CAS", "NAME"),
            dataset_version=None,
            license_status=license_status,
            configuration_error=error,
        )

    def _load(self, manifest_path: Optional[Path]) -> ReferenceProviderMetadata:
        if manifest_path is None:
            return self._disabled("NOT_CONFIGURED")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(manifest.get("provider", "")).upper() != self.provider:
                raise ValueError("PROVIDER_MISMATCH")
            if manifest.get("license_approved") is not True:
                raise ValueError("LICENSE_NOT_APPROVED")
            valid_until = manifest.get("valid_until")
            if valid_until:
                expiry = datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                if expiry <= datetime.now(timezone.utc):
                    raise ValueError("SNAPSHOT_EXPIRED")
            version = str(manifest["dataset_version"]).strip()
            expected_sha = str(manifest["snapshot_sha256"]).strip().lower()
            snapshot_path = Path(str(manifest["snapshot_path"]))
            if not snapshot_path.is_absolute():
                snapshot_path = (manifest_path.parent / snapshot_path).resolve()
            if _sha256(snapshot_path) != expected_sha:
                raise ValueError("SNAPSHOT_CHECKSUM_MISMATCH")
            frame = self._read_snapshot(snapshot_path)
            records = self._records_from_frame(frame)
            if not records:
                raise ValueError("SNAPSHOT_HAS_NO_IDENTIFIERS")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            stable_codes = {
                "PROVIDER_MISMATCH",
                "LICENSE_NOT_APPROVED",
                "SNAPSHOT_EXPIRED",
                "SNAPSHOT_CHECKSUM_MISMATCH",
                "SNAPSHOT_HAS_NO_IDENTIFIERS",
                "UNSUPPORTED_SNAPSHOT_FORMAT",
            }
            candidate_code = str(error)
            code = (
                candidate_code
                if candidate_code in stable_codes
                else f"{type(error).__name__.upper()}_CONFIGURATION_ERROR"
            )
            return self._disabled("INVALID_CONFIGURATION", code)

        self._records = tuple(records)
        self._indexes = {
            "inchi_key": self._build_index("inchi_key"),
            "connectivity_key": self._build_index("connectivity_key"),
            "cas": self._build_index("cas"),
            "normalized_name": self._build_index("normalized_name"),
        }
        return ReferenceProviderMetadata(
            provider=self.provider,
            display_name=self.display_name,
            source_type="FRAGRANCE_CATALOG",
            enabled=True,
            external=False,
            query_types=("FULL_INCHIKEY", "CONNECTIVITY_INCHIKEY", "CAS", "NAME"),
            dataset_version=version,
            license_status="APPROVED",
        )

    @staticmethod
    def _read_snapshot(path: Path) -> pd.DataFrame:
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError("UNSUPPORTED_SNAPSHOT_FORMAT")

    @staticmethod
    def _records_from_frame(frame: pd.DataFrame) -> Sequence[_SnapshotRecord]:
        records = []
        with rdBase.BlockLogs():
            for row_index, row in frame.iterrows():
                inchi_key = _clean_optional(row.get("inchi_key"))
                smiles = _clean_optional(row.get("isomeric_smiles")) or _clean_optional(
                    row.get("smiles")
                )
                if not inchi_key and smiles:
                    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
                    if molecule is not None:
                        try:
                            inchi_key = Chem.MolToInchiKey(molecule) or None
                        except Exception:
                            inchi_key = None
                if inchi_key:
                    inchi_key = inchi_key.upper()
                connectivity_key = (
                    inchi_key.split("-")[0]
                    if inchi_key
                    else _clean_optional(row.get("connectivity_key"))
                )
                if connectivity_key:
                    connectivity_key = connectivity_key.upper()
                cas = _normalize_cas(_clean_optional(row.get("cas")))
                name = _normalize_name(_clean_optional(row.get("name")))
                if not any((inchi_key, connectivity_key, cas, name)):
                    continue
                record_id = _clean_optional(row.get("record_id")) or str(row_index)
                records.append(
                    _SnapshotRecord(
                        record_id=record_id,
                        record_url=_normalize_record_url(
                            _clean_optional(row.get("record_url"))
                        ),
                        inchi_key=inchi_key,
                        connectivity_key=connectivity_key,
                        cas=cas,
                        normalized_name=name,
                    )
                )
        return records

    def _build_index(self, attribute: str) -> Dict[str, Tuple[_SnapshotRecord, ...]]:
        values: Dict[str, list[_SnapshotRecord]] = {}
        for record in self._records:
            key = getattr(record, attribute)
            if key:
                values.setdefault(key, []).append(record)
        return {key: tuple(records) for key, records in values.items()}

    def _evidence(
        self,
        status: ReferenceStatus,
        match_level: Optional[MatchLevel],
        queried_identifier: Optional[str],
        records: Sequence[_SnapshotRecord] = (),
        error_code: Optional[str] = None,
    ) -> ReferenceEvidence:
        return ReferenceEvidence(
            provider=self.provider,
            status=status,
            match_level=match_level,
            queried_identifier=queried_identifier,
            record_ids=tuple(record.record_id for record in records),
            record_urls=tuple(
                record.record_url for record in records if record.record_url
            ),
            source_version=self.metadata.dataset_version,
            error_code=error_code,
        )

    def check(self, query: ReferenceQuery, *, consent: bool) -> ReferenceEvidence:
        del consent  # Local snapshots never transmit an identifier.
        if not self.metadata.enabled:
            return self._evidence(
                ReferenceStatus.UNVERIFIED
                if self.metadata.license_status == "INVALID_CONFIGURATION"
                else ReferenceStatus.NOT_CONFIGURED,
                None,
                None,
                error_code=self.metadata.configuration_error,
            )
        candidates = (
            ("inchi_key", query.inchi_key.upper() if query.inchi_key else None, MatchLevel.EXACT_STEREO),
            (
                "connectivity_key",
                query.connectivity_key.upper() if query.connectivity_key else None,
                MatchLevel.EXACT_CONNECTIVITY,
            ),
            ("cas", _normalize_cas(query.cas), MatchLevel.EXACT_CAS),
        )
        for index_name, identifier, match_level in candidates:
            if identifier and identifier in self._indexes[index_name]:
                return self._evidence(
                    ReferenceStatus.MATCH,
                    match_level,
                    identifier,
                    self._indexes[index_name][identifier],
                )
        normalized_name = _normalize_name(query.name)
        if normalized_name and normalized_name in self._indexes["normalized_name"]:
            return self._evidence(
                ReferenceStatus.AMBIGUOUS,
                MatchLevel.NAME_ONLY,
                query.name,
                self._indexes["normalized_name"][normalized_name],
                error_code="NAME_ONLY_MATCH",
            )
        queried_identifier = query.inchi_key or query.connectivity_key or query.cas or query.name
        return self._evidence(ReferenceStatus.NO_MATCH, None, queried_identifier)


class TGSCProvider(LicensedSnapshotProvider):
    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        super().__init__(TGSC, "The Good Scents Company", manifest_path)


class ScentreeProvider(LicensedSnapshotProvider):
    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        super().__init__(SCENTREE, "ScenTree", manifest_path)


class ReferenceVerifier:
    """Run configured reference providers concurrently and apply a fail-closed gate."""

    def __init__(
        self,
        providers: Iterable[ReferenceProvider],
        *,
        max_workers: int = 4,
    ) -> None:
        self.providers = tuple(providers)
        identifiers = [provider.metadata.provider for provider in self.providers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Reference provider identifiers must be unique")
        self.max_workers = max(1, min(max_workers, len(self.providers) or 1))

    @property
    def metadata(self) -> Tuple[ReferenceProviderMetadata, ...]:
        return tuple(provider.metadata for provider in self.providers)

    @property
    def required_external_providers(self) -> Tuple[str, ...]:
        return tuple(
            item.provider
            for item in self.metadata
            if item.enabled and item.external
        )

    @staticmethod
    def _safe_check(
        provider: ReferenceProvider,
        query: ReferenceQuery,
        consents: Set[str],
    ) -> ReferenceEvidence:
        metadata = provider.metadata
        if not metadata.enabled:
            result = provider.check(query, consent=False)
            return replace(
                result,
                checked_at=result.checked_at or datetime.now(timezone.utc).isoformat(),
            )
        if metadata.external and metadata.provider not in consents:
            return ReferenceEvidence(
                provider=metadata.provider,
                status=ReferenceStatus.UNVERIFIED,
                match_level=None,
                queried_identifier=None,
                source_version=metadata.dataset_version,
                error_code="CONSENT_REQUIRED",
                checked_at=datetime.now(timezone.utc).isoformat(),
            )
        try:
            result = provider.check(
                query,
                consent=not metadata.external or metadata.provider in consents,
            )
            return replace(
                result,
                checked_at=result.checked_at or datetime.now(timezone.utc).isoformat(),
            )
        except Exception as error:
            return ReferenceEvidence(
                provider=metadata.provider,
                status=ReferenceStatus.UNVERIFIED,
                match_level=None,
                queried_identifier=None,
                source_version=metadata.dataset_version,
                error_code=f"PROVIDER_ERROR_{type(error).__name__}",
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

    @staticmethod
    def evaluate(evidences: Sequence[ReferenceEvidence]) -> ReferenceGate:
        matches = tuple(
            item.provider
            for item in evidences
            if item.status is ReferenceStatus.MATCH
        )
        if matches:
            return ReferenceGate(
                ReferenceGateStatus.BLOCKED_MATCH,
                matches,
                "KNOWN_REFERENCE_MATCH",
            )
        review = tuple(
            item.provider
            for item in evidences
            if item.status in {ReferenceStatus.AMBIGUOUS, ReferenceStatus.UNVERIFIED}
        )
        if review:
            return ReferenceGate(
                ReferenceGateStatus.REVIEW_REQUIRED,
                review,
                "REFERENCE_UNVERIFIED",
            )
        return ReferenceGate(ReferenceGateStatus.PASS)

    def verify(
        self,
        query: ReferenceQuery,
        *,
        consents: Iterable[str] = (),
    ) -> ReferenceCheckBundle:
        consent_set = {str(provider).upper() for provider in consents}
        results: list[Optional[ReferenceEvidence]] = [None] * len(self.providers)
        active = []
        for index, provider in enumerate(self.providers):
            if provider.metadata.enabled:
                active.append((index, provider))
            else:
                results[index] = self._safe_check(provider, query, consent_set)
        if active:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(active))
            ) as executor:
                futures = {
                    index: executor.submit(
                        self._safe_check,
                        provider,
                        query,
                        consent_set,
                    )
                    for index, provider in active
                }
                for index, future in futures.items():
                    results[index] = future.result()
        if any(item is None for item in results):
            raise RuntimeError("Reference verifier produced an incomplete result")
        evidences = tuple(item for item in results if item is not None)
        return ReferenceCheckBundle(evidences, self.evaluate(evidences))


def build_reference_verifier(
    pubchem_client: PubChemClient,
    *,
    tgsc_manifest: Optional[Path] = None,
    scentree_manifest: Optional[Path] = None,
) -> ReferenceVerifier:
    return ReferenceVerifier(
        (
            PubChemProvider(pubchem_client),
            TGSCProvider(tgsc_manifest),
            ScentreeProvider(scentree_manifest),
        )
    )
