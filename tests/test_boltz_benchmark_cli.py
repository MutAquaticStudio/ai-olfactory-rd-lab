from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import benchmark_boltz_or as cli


VALID_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


def write_input(path: Path) -> None:
    path.write_text(
        "receptor_id,receptor_sequence,compound_id,smiles,experimental_state,experimental_value,experimental_unit,source_id\n"
        f"OR_TEST,{VALID_SEQUENCE},cmp-1,CCO,ACTIVE,1.0,normalized,source-1\n",
        encoding="utf-8",
    )


class FakeProvider:
    def __init__(self, *, mode="test", estimate="0.10"):
        self.config = SimpleNamespace(mode=mode)
        self.metadata = {
            "provider": "BOLTZ",
            "mode": mode,
            "external": True,
            "research_only": True,
        }
        self.estimate = estimate
        self.estimate_calls = []
        self.start_calls = []

    def estimate_cost(self, sequence, smiles):
        self.estimate_calls.append((sequence, smiles))
        return {
            "estimated_cost_usd": self.estimate,
            "idempotency_key": "nox-or-binding-v1-test",
            "mode": self.config.mode,
            "breakdown": {"num_units": 1},
            "disclaimer": "test",
        }

    def start(self, sequence, smiles):
        self.start_calls.append((sequence, smiles))
        return {
            "provider_resource_id": "pred-test",
            "status": "succeeded",
            "evidence_type": "PREDICTED_RECEPTOR_BINDING",
        }

    def wait(self, resource_id, *, poll_seconds, max_wait_seconds):
        raise AssertionError("Immediate fake result should not poll")


def test_cli_refuses_any_external_call_without_explicit_consent(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.jsonl"
    write_input(input_path)

    def provider_should_not_be_built():
        raise AssertionError("No provider should be built before external consent")

    monkeypatch.setattr(cli, "build_boltz_provider_from_env", provider_should_not_be_built)
    result = cli.main(["--input", str(input_path), "--output", str(output_path)])

    assert result == 2
    assert not output_path.exists()


def test_cli_blocks_live_compute_before_estimation_without_allow_live(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.jsonl"
    write_input(input_path)
    provider = FakeProvider(mode="live")
    monkeypatch.setattr(cli, "build_boltz_provider_from_env", lambda: provider)

    result = cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--consent-external-boltz",
            "--execute",
        ]
    )

    assert result == 2
    assert provider.estimate_calls == []
    assert provider.start_calls == []


def test_cli_cost_ceiling_prevents_prediction_submission(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.jsonl"
    write_input(input_path)
    provider = FakeProvider(mode="test", estimate="2.00")
    monkeypatch.setattr(cli, "build_boltz_provider_from_env", lambda: provider)

    result = cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--consent-external-boltz",
            "--execute",
            "--max-total-cost-usd",
            "1.00",
        ]
    )

    assert result == 2
    assert len(provider.estimate_calls) == 1
    assert provider.start_calls == []
    assert not output_path.exists()


def test_cli_estimate_only_writes_cost_record_without_prediction(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.jsonl"
    write_input(input_path)
    provider = FakeProvider(mode="test", estimate="0.10")
    monkeypatch.setattr(cli, "build_boltz_provider_from_env", lambda: provider)

    result = cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--consent-external-boltz",
        ]
    )

    assert result == 0
    assert len(provider.estimate_calls) == 1
    assert provider.start_calls == []
    text = output_path.read_text(encoding="utf-8")
    assert '"estimated_cost_usd": "0.10"' in text
    assert '"receptor_id": "OR_TEST"' in text


def test_cli_test_execution_writes_research_evidence(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.jsonl"
    write_input(input_path)
    provider = FakeProvider(mode="test", estimate="0")
    monkeypatch.setattr(cli, "build_boltz_provider_from_env", lambda: provider)

    result = cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--consent-external-boltz",
            "--execute",
            "--max-total-cost-usd",
            "0",
        ]
    )

    assert result == 0
    assert len(provider.estimate_calls) == 1
    assert len(provider.start_calls) == 1
    text = output_path.read_text(encoding="utf-8")
    assert '"evidence_type": "PREDICTED_RECEPTOR_BINDING"' in text
    assert '"experimental_state": "ACTIVE"' in text
