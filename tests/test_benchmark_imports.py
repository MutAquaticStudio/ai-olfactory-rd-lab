"""Benchmark helper imports must not require private model resources in CI."""
import os
from pathlib import Path
import subprocess
import sys


def test_benchmark_imports_without_private_bundle(tmp_path):
    environment = dict(os.environ, SCENT_STUDIO_RESOURCE_DIR=str(tmp_path / 'missing-bundle'))
    result = subprocess.run(
        [sys.executable, '-c', 'import benchmark_judge_v2; import benchmark_baselines'],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
