import csv
import json
import time
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.workflow import ask

DATASET_PATH = ROOT / "tests" / "evaluation_questions.json"
LOG_DIR = ROOT / "logs"
CSV_PATH = LOG_DIR / "evaluation_runs.csv"
SUMMARY_PATH = LOG_DIR / "evaluation_summary.json"

EVALUATED_STATUSES = {
    "grounded",
    "partially_grounded",
    "unsupported",
}

CSV_FIELDS = [
    "timestamp",
    "test_id",
    "category",
    "question",
    "source_used",
    "hallucination_status",
    "hallucination_score",
    "unsupported_claims",
    "evaluation_explanation",
    "latency_seconds",
    "error",
]


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        test_cases = json.load(file)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for test_case in test_cases:
        test_id = test_case["id"]
        question = test_case["question"]
        category = test_case.get("category", "general")

        print(f"\nRunning {test_id}: {question}")

        start_time = time.perf_counter()
        error = ""

        try:
            result = ask(question)

            row = {
                "timestamp": timestamp,
                "test_id": test_id,
                "category": category,
                "question": question,
                "source_used": result.get("source_used", ""),
                "hallucination_status": result.get(
                    "hallucination_status", "missing"
                ),
                "hallucination_score": result.get(
                    "hallucination_score", -1.0
                ),
                "unsupported_claims": json.dumps(
                    result.get("unsupported_claims", []),
                    ensure_ascii=False,
                ),
                "evaluation_explanation": result.get(
                    "evaluation_explanation", ""
                ),
                "latency_seconds": 0.0,
                "error": "",
            }

        except Exception as exc:
            error = repr(exc)

            row = {
                "timestamp": timestamp,
                "test_id": test_id,
                "category": category,
                "question": question,
                "source_used": "",
                "hallucination_status": "run_error",
                "hallucination_score": -1.0,
                "unsupported_claims": "[]",
                "evaluation_explanation": "",
                "latency_seconds": 0.0,
                "error": error,
            }

            print(f"ERROR: {error}")

        row["latency_seconds"] = round(
            time.perf_counter() - start_time, 3
        )

        results.append(row)

        print(
            f"Source: {row['source_used'] or 'unknown'} | "
            f"Status: {row['hallucination_status']} | "
            f"Score: {row['hallucination_score']} | "
            f"Latency: {row['latency_seconds']}s"
        )

    # Append results so multiple runs can be compared.
    file_exists = CSV_PATH.exists()

    with CSV_PATH.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=CSV_FIELDS
        )

        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerows(results)

    # Calculate metrics for this run only.
    statuses = [
        row["hallucination_status"]
        for row in results
    ]

    evaluated = [
        row for row in results
        if row["hallucination_status"] in EVALUATED_STATUSES
        and float(row["hallucination_score"]) >= 0
    ]

    evaluated_count = len(evaluated)
    total_count = len(results)

    grounded_count = sum(
        row["hallucination_status"] == "grounded"
        for row in evaluated
    )

    partial_count = sum(
        row["hallucination_status"] == "partially_grounded"
        for row in evaluated
    )

    unsupported_count = sum(
        row["hallucination_status"] == "unsupported"
        for row in evaluated
    )

    evaluation_error_count = sum(
        status in {"evaluation_error", "run_error", "missing"}
        for status in statuses
    )

    scores = [
        float(row["hallucination_score"])
        for row in evaluated
    ]

    summary = {
        "timestamp": timestamp,
        "total_questions": total_count,
        "evaluated_questions": evaluated_count,
        "grounded_count": grounded_count,
        "partially_grounded_count": partial_count,
        "unsupported_count": unsupported_count,
        "evaluation_error_count": evaluation_error_count,
        "grounded_rate": (
            round(grounded_count / evaluated_count, 4)
            if evaluated_count else None
        ),
        "unsupported_rate": (
            round(unsupported_count / evaluated_count, 4)
            if evaluated_count else None
        ),
        "evaluation_error_rate": (
            round(evaluation_error_count / total_count, 4)
            if total_count else 0.0
        ),
        "average_groundedness_score": (
            round(sum(scores) / len(scores), 4)
            if scores else None
        ),
        "average_latency_seconds": (
            round(
                sum(row["latency_seconds"] for row in results)
                / total_count,
                3,
            )
            if total_count else 0.0
        ),
        "status_counts": dict(Counter(statuses)),
    }

    with SUMMARY_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n========== EVALUATION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {CSV_PATH}")
    print(f"Latest summary:  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
