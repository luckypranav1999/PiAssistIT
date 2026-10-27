
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.workflow import llm
from app.rag.evaluation import evaluate_groundedness


def main():
    question = "What is the company's VPN password policy?"

    # Deliberately limited evidence: it says nothing about password expiry.
    evidence = """
    Company IT handbook:
    Employees must connect to the company VPN using their
    company credentials and complete multi-factor authentication.
    """

    # Deliberately fabricated claim, absent from the evidence.
    answer = (
        "The company requires every employee to change their VPN "
        "password every 30 days. Failure to do so results in "
        "automatic account suspension."
    )

    print("Running deliberate unsupported-claim test...")

    result = evaluate_groundedness(
        model=llm(),
        question=question,
        answer=answer,
        evidence=evidence,
    )

    print("\n========== EVALUATOR CHALLENGE ==========")
    print("Verdict:", result.verdict)
    print("Score:", result.score)
    print("Unsupported claims:", result.unsupported_claims)
    print("Explanation:", result.explanation)

    if result.verdict == "unsupported":
        print("\nPASS: Evaluator identified the unsupported answer.")
    elif result.verdict == "partially_grounded":
        print(
            "\nPARTIAL: Evaluator detected a grounding problem, "
            "but did not classify the entire answer as unsupported."
        )
    else:
        print(
            "\nFAIL: Evaluator marked a clearly unsupported answer "
            "as grounded. Review the evaluation prompt and model output."
        )


if __name__ == "__main__":
    main()
