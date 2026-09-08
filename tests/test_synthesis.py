from pathlib import Path

from olfactory.synthesis import (
    AiZynthFinderService,
    UnavailableRetrosynthesisService,
)


def test_unconfigured_route_search_is_explicit_and_not_a_synthesis_claim():
    result = UnavailableRetrosynthesisService().search("CCO")

    assert result.status == "NOT_CONFIGURED"
    assert result.route_found is None


def test_configured_route_search_preserves_evidence_and_limits(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("test: true", encoding="utf-8")
    calls = []

    def runner(smiles: str, path: Path, limit: int):
        calls.append((smiles, path, limit))
        return {"route_found": True, "route_steps": 4, "precursor_coverage": 0.75}

    result = AiZynthFinderService(config, runner=runner).search("CCO")

    assert calls == [("CCO", config.resolve(), 300)]
    assert result.status == "ROUTE_FOUND"
    assert result.route_steps == 4
    assert result.precursor_coverage == 0.75
    assert "not a synthesis guarantee" in result.warnings[0]


def test_route_search_failure_returns_reviewable_status_without_exception(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("test: true", encoding="utf-8")

    def runner(*_):
        raise RuntimeError("private message")

    result = AiZynthFinderService(config, runner=runner).search("CCO")

    assert result.status == "SEARCH_ERROR"
    assert result.warnings == ("RuntimeError",)
