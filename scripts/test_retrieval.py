
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.vectorstore import get_retriever


HANDBOOK = "company_it_handbook.md"

TEST_CASES = [
    {
        "id": "R001",
        "question": "How do I connect to the company VPN?",
        "expected_sources": [HANDBOOK],
    },
    {
        "id": "R002",
        "question": "What should I do if my VPN connection fails?",
        "expected_sources": [HANDBOOK],
    },
    {
        "id": "R003",
        "question": "How do I complete multi-factor authentication?",
        "expected_sources": [HANDBOOK],
    },
]


def main():
    retriever = get_retriever()
    total_hits = 0
    total_relevant = 0
    total_retrieved = 0

    for case in TEST_CASES:
        print("\n" + "=" * 70)
        print(f"{case['id']}: {case['question']}")

        docs = retriever.invoke(case["question"])
        expected = set(case["expected_sources"])

        retrieved_sources = [
            Path(doc.metadata.get("source", "unknown")).name
            for doc in docs
        ]

        print("Retrieved chunks:", len(docs))

        for rank, (doc, source) in enumerate(
            zip(docs, retrieved_sources), start=1
        ):
            relevant = source in expected

            if relevant:
                total_relevant += 1

            preview = " ".join(doc.page_content.split())[:150]

            print(
                f"\nRank {rank} | "
                f"Expected source match: {relevant}"
            )
            print("Source:", source)
            print("Preview:", preview)

        hit = any(source in expected for source in retrieved_sources)
        total_hits += int(hit)
        total_retrieved += len(retrieved_sources)

        print("\nHit@4:", int(hit))

        if retrieved_sources:
            source_precision = sum(
                source in expected for source in retrieved_sources
            ) / len(retrieved_sources)
            print("Source-level Precision@4:", round(source_precision, 3))
        else:
            print("Source-level Precision@4: 0.0")

    print("\n" + "=" * 70)
    print("OVERALL RETRIEVAL SUMMARY")
    print("Questions:", len(TEST_CASES))
    print("Hit@4:", round(total_hits / len(TEST_CASES), 3))

    if total_retrieved:
        print(
            "Source-level Precision@4:",
            round(total_relevant / total_retrieved, 3),
        )
    else:
        print("Source-level Precision@4: 0.0")


if __name__ == "__main__":
    main()
