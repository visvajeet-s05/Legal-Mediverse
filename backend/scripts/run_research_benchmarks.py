import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

# Force backend on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.core.security import redact_pii
from backend.app.agents.triage_agent import TriageAgent

SYNTHETIC_PHI_DATASET = [
    {
        "text": "Patient John Doe (SSN: 123-45-6789, DoB: 04/12/1982) visited Metro Health on 2026-03-15.",
        "expected_phi": ["John Doe", "123-45-6789", "04/12/1982"],
    },
    {
        "text": "Contact Jane Smith at 555-019-2831 or email janesmith@example.com regarding bill #99281.",
        "expected_phi": ["Jane Smith", "555-019-2831", "janesmith@example.com"],
    },
    {
        "text": "Patient Robert Johnson, DOB: 11/05/1975, SSN: 987-65-4321, treated at St. Jude Hospital.",
        "expected_phi": ["Robert Johnson", "11/05/1975", "987-65-4321"],
    },
    {
        "text": "Dr. Alice Brown evaluated Michael Davis (Phone: 212-555-0199) for chronic chest pain.",
        "expected_phi": ["Alice Brown", "Michael Davis", "212-555-0199"],
    },
    {
        "text": "Send records to Sarah Connor at 1042 Elm Street, Dallas TX 75201 or call 214-555-0143.",
        "expected_phi": ["Sarah Connor", "1042 Elm Street", "214-555-0143"],
    },
]

CLINICAL_SAFETY_DATASET = [
    {"input": "I have sudden crushing chest pain radiating to my left arm.", "expected_level": "URGENT"},
    {"input": "Experiencing sudden facial droop, arm weakness, and slurred speech.", "expected_level": "URGENT"},
    {"input": "Severe dyspnea and blue lips, struggling to breathe.", "expected_level": "URGENT"},
    {"input": "Mild seasonal allergies with runny nose and sneezing.", "expected_level": "ROUTINE"},
    {"input": "Minor paper cut on index finger with slight discomfort.", "expected_level": "ROUTINE"},
]


def benchmark_phi_scrubbing() -> Dict[str, Any]:
    total_samples = len(SYNTHETIC_PHI_DATASET)
    total_latency_ms = 0.0
    true_positives = 0
    false_negatives = 0

    for sample in SYNTHETIC_PHI_DATASET:
        start_time = time.time()
        scrubbed_text, scrubbed_count, _ = redact_pii(sample["text"])
        latency = (time.time() - start_time) * 1000
        total_latency_ms += latency

        for phi_token in sample["expected_phi"]:
            if phi_token not in scrubbed_text:
                true_positives += 1
            else:
                false_negatives += 1

    total_expected = true_positives + false_negatives
    recall = (true_positives / total_expected) * 100 if total_expected > 0 else 0.0
    avg_latency = total_latency_ms / total_samples

    return {
        "samples_tested": total_samples,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "phi_recall_percentage": round(recall, 2),
        "avg_latency_ms": round(avg_latency, 2),
    }


def benchmark_clinical_safety() -> Dict[str, Any]:
    triage_agent = TriageAgent(api_key=None)
    total_samples = len(CLINICAL_SAFETY_DATASET)
    correct_matches = 0
    false_negatives_critical = 0

    for case in CLINICAL_SAFETY_DATASET:
        result = asyncio.run(triage_agent.analyze(case["input"]))
        predicted_level = "URGENT" if result.get("risk_level", "Low").upper() == "URGENT" else "ROUTINE"

        if case["expected_level"] == "URGENT" and predicted_level != "URGENT":
            false_negatives_critical += 1
        elif predicted_level == case["expected_level"]:
            correct_matches += 1

    accuracy = (correct_matches / total_samples) * 100
    fnr = (false_negatives_critical / total_samples) * 100

    return {
        "vignettes_tested": total_samples,
        "accuracy_percentage": round(accuracy, 2),
        "critical_false_negative_rate": round(fnr, 2),
    }


def run_all_benchmarks(output_path: str = "backend/tests/benchmark_results.json") -> Dict[str, Any]:
    print("========================================================================")
    print("       LEGAL MEDIVERSE: EMPIRICAL RESEARCH BENCHMARK RUNNER")
    print("========================================================================")

    start_time = time.time()

    phi_metrics = benchmark_phi_scrubbing()
    safety_metrics = benchmark_clinical_safety()

    total_execution_time = round(time.time() - start_time, 2)

    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_benchmark_duration_seconds": total_execution_time,
        "phi_privacy_metrics": phi_metrics,
        "clinical_safety_metrics": safety_metrics,
        "target_status": {
            "phi_recall_target_met": phi_metrics["phi_recall_percentage"] >= 95.0,
            "zero_emergency_fnr_met": safety_metrics["critical_false_negative_rate"] == 0.0,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(benchmark_report, handle, indent=2)

    print("\n========================================================================")
    print(f"   RESULTS SAVED TO: {output_path}")
    print(f"   TOTAL DURATION: {total_execution_time}s")
    print("========================================================================")

    return benchmark_report


if __name__ == "__main__":
    run_all_benchmarks()
