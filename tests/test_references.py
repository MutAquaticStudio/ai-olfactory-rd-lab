import hashlib
import json
import threading
from pathlib import Path

from rdkit import Chem

from olfactory.pubchem import NoveltyResult, NoveltyStatus
from olfactory.references import (
    MatchLevel,
    PubChemProvider,
    ReferenceEvidence,
    ReferenceGateStatus,
    ReferenceProviderMetadata,
    ReferenceQuery,
    ReferenceStatus,
    ReferenceVerifier,
    ScentreeProvider,
    TGSCProvider,
    reference_query_from_smiles,
)


class StaticPubChem:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def verify(self, smiles, *, consent):
        self.calls.append((smiles, consent))
        return self.result


def write_snapshot(tmp_path: Path, provider: str, rows: str, **overrides):
    snapshot = tmp_path / f"{provider.lower()}.csv"
    snapshot.write_text(rows, encoding="utf-8")
    manifest = {
        "provider": provider,
        "dataset_version": "licensed-2026-09",
        "license_approved": True,
        "snapshot_path": snapshot.name,
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    manifest.update(overrides)
    manifest_path = tmp_path / f"{provider.lower()}-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_pubchem_provider_maps_identity_result_without_changing_semantics():
    query = reference_query_from_smiles("CCO")
    client = StaticPubChem(NoveltyResult(NoveltyStatus.FOUND, "CCO", cids=(702,)))
    evidence = PubChemProvider(client).check(query, consent=True)

    assert evidence.status is ReferenceStatus.MATCH
    assert evidence.match_level is MatchLevel.EXACT_STEREO
    assert evidence.record_ids == ("702",)
    assert client.calls == [("CCO", True)]


def test_unconfigured_catalog_provider_never_performs_a_lookup():
    provider = TGSCProvider()
    evidence = provider.check(reference_query_from_smiles("CCO"), consent=False)

    assert not provider.metadata.enabled
    assert provider.metadata.license_status == "NOT_CONFIGURED"
    assert evidence.status is ReferenceStatus.NOT_CONFIGURED


def test_snapshot_requires_license_approval_and_matching_sha256(tmp_path):
    query = reference_query_from_smiles("CCO")
    rows = f"record_id,inchi_key,name\nethanol,{query.inchi_key},Ethanol\n"
    unlicensed = write_snapshot(tmp_path, "TGSC", rows, license_approved=False)
    invalid_sha = write_snapshot(
        tmp_path,
        "SCENTREE",
        rows,
        snapshot_sha256="0" * 64,
    )

    tgsc = TGSCProvider(unlicensed)
    scentree = ScentreeProvider(invalid_sha)

    assert not tgsc.metadata.enabled
    assert tgsc.metadata.configuration_error == "LICENSE_NOT_APPROVED"
    assert not scentree.metadata.enabled
    assert scentree.metadata.configuration_error == "SNAPSHOT_CHECKSUM_MISMATCH"


def test_expired_configured_snapshot_fails_closed_into_review(tmp_path):
    query = reference_query_from_smiles("CCO")
    rows = f"record_id,inchi_key,name\nethanol,{query.inchi_key},Ethanol\n"
    manifest = write_snapshot(
        tmp_path,
        "TGSC",
        rows,
        valid_until="2020-01-01T00:00:00Z",
    )
    provider = TGSCProvider(manifest)

    bundle = ReferenceVerifier([provider]).verify(query)

    assert provider.metadata.configuration_error == "SNAPSHOT_EXPIRED"
    assert bundle.evidences[0].status is ReferenceStatus.UNVERIFIED
    assert bundle.gate.status is ReferenceGateStatus.REVIEW_REQUIRED


def test_snapshot_matching_prefers_stereo_then_connectivity_and_flags_name_only(tmp_path):
    query = reference_query_from_smiles("C[C@H](O)C(=O)O")
    other_stereo = reference_query_from_smiles("C[C@@H](O)C(=O)O")
    rows = (
        "record_id,record_url,inchi_key,cas,name\n"
        f"lactic-r,https://catalog.invalid/lactic-r,{query.inchi_key},50-21-5,Lactic acid\n"
    )
    provider = TGSCProvider(write_snapshot(tmp_path, "TGSC", rows))

    exact = provider.check(query, consent=False)
    connectivity = provider.check(other_stereo, consent=False)
    cas = provider.check(
        ReferenceQuery("CC", None, None, cas="50-21-5"),
        consent=False,
    )
    name = provider.check(
        ReferenceQuery("CC", None, None, name="Lactic acid"),
        consent=False,
    )

    assert exact.status is ReferenceStatus.MATCH
    assert exact.match_level is MatchLevel.EXACT_STEREO
    assert exact.record_urls == ("https://catalog.invalid/lactic-r",)
    assert connectivity.match_level is MatchLevel.EXACT_CONNECTIVITY
    assert cas.match_level is MatchLevel.EXACT_CAS
    assert name.status is ReferenceStatus.AMBIGUOUS
    assert name.match_level is MatchLevel.NAME_ONLY


class BarrierProvider:
    def __init__(self, provider, barrier):
        self.barrier = barrier
        self._metadata = ReferenceProviderMetadata(
            provider=provider,
            display_name=provider,
            source_type="FRAGRANCE_CATALOG",
            enabled=True,
            external=False,
            query_types=("FULL_INCHIKEY",),
            dataset_version="test",
            license_status="APPROVED",
        )

    @property
    def metadata(self):
        return self._metadata

    def check(self, query, *, consent):
        self.barrier.wait(timeout=2)
        return ReferenceEvidence(
            self.metadata.provider,
            ReferenceStatus.NO_MATCH,
            None,
            query.inchi_key,
        )


def test_enabled_providers_run_concurrently_and_all_no_match_passes():
    barrier = threading.Barrier(2)
    verifier = ReferenceVerifier(
        [BarrierProvider("ONE", barrier), BarrierProvider("TWO", barrier)]
    )

    bundle = verifier.verify(reference_query_from_smiles("CCO"))

    assert bundle.gate.status is ReferenceGateStatus.PASS
    assert [item.status for item in bundle.evidences] == [
        ReferenceStatus.NO_MATCH,
        ReferenceStatus.NO_MATCH,
    ]


class RecordingExternalProvider:
    def __init__(self):
        self.calls = []
        self._metadata = ReferenceProviderMetadata(
            provider="EXTERNAL",
            display_name="External",
            source_type="FRAGRANCE_CATALOG",
            enabled=True,
            external=True,
            query_types=("FULL_INCHIKEY",),
            dataset_version="test",
            license_status="APPROVED",
        )

    @property
    def metadata(self):
        return self._metadata

    def check(self, query, *, consent):
        self.calls.append((query, consent))
        return ReferenceEvidence(
            "EXTERNAL",
            ReferenceStatus.NO_MATCH,
            None,
            query.inchi_key,
        )


def test_external_provider_is_not_called_without_explicit_consent():
    provider = RecordingExternalProvider()
    bundle = ReferenceVerifier([provider]).verify(reference_query_from_smiles("CCO"))

    assert provider.calls == []
    assert bundle.evidences[0].status is ReferenceStatus.UNVERIFIED
    assert bundle.evidences[0].error_code == "CONSENT_REQUIRED"
    assert bundle.gate.status is ReferenceGateStatus.REVIEW_REQUIRED


def test_match_blocks_and_unverified_requires_review():
    match = ReferenceEvidence("TGSC", ReferenceStatus.MATCH, MatchLevel.EXACT_CAS, "1-11-1")
    unavailable = ReferenceEvidence(
        "SCENTREE",
        ReferenceStatus.UNVERIFIED,
        None,
        None,
        error_code="TIMEOUT",
    )

    assert ReferenceVerifier.evaluate([match]).status is ReferenceGateStatus.BLOCKED_MATCH
    assert ReferenceVerifier.evaluate([unavailable]).status is ReferenceGateStatus.REVIEW_REQUIRED
