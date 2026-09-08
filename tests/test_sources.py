import json

import pytest

from olfactory.data_foundation.sources import load_source_registry, mirror_pyrfume_archive


def test_source_registry_pins_commits_and_separates_quality_tiers():
    registry = load_source_registry()
    assert len(registry["pyrfume_data_commit"]) == 40
    assert registry["archives"]["leffingwell"]["quality_tier"] == "WEAK_LABEL_CATALOG"
    assert registry["archives"]["keller_2016"]["quality_tier"] == "QUANTITATIVE_PANEL_REPLICATED"
    assert registry["archives"]["dravnieks_1985"]["license_status"] == "REVIEW_REQUIRED"
    assert registry["archives"]["keller_2016"]["license_status"] == "APPROVED"
    assert registry["archives"]["keller_2016"]["license_evidence_url"].startswith("https://link.springer.com/")


def test_mirroring_is_blocked_before_license_review(tmp_path):
    with pytest.raises(PermissionError, match="LICENSE_REVIEW_REQUIRED"):
        mirror_pyrfume_archive("leffingwell", tmp_path, license_approval_ticket=None)
