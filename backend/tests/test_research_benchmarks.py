import json
import os
from pathlib import Path

from backend.scripts.run_research_benchmarks import run_all_benchmarks


def test_run_all_benchmarks_writes_results(tmp_path):
    output_path = tmp_path / "benchmark_results.json"

    report = run_all_benchmarks(output_path=str(output_path))

    assert output_path.exists()
    assert report["phi_privacy_metrics"]["phi_recall_percentage"] >= 0
    assert report["clinical_safety_metrics"]["critical_false_negative_rate"] >= 0
    assert report["target_status"]["phi_recall_target_met"] is True
