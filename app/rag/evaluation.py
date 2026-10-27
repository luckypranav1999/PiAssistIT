
from typing import Literal

from pydantic import BaseModel, Field


class HallucinationEvaluation(BaseModel):
    verdict: Literal[
        "grounded",
        "partially_grounded",
        "unsupported",
    ] = Field(
        description=(
            "grounded if all material claims are supported; "
            "partially_grounded if some claims are unsupported; "
            "unsupported if the answer is largely unsupported."
        )
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimated proportion of material answer claims "
            "supported by the supplied evidence. 1.0 means all "
            "material claims are supported; 0.0 means none are."
        ),
    )
    unsupported_claims: list[str] = Field(
        description="Specific material claims not supported by the evidence."
    )
    explanation: str = Field(
        description="Brief reason for the groundedness verdict."
    )



def evaluate_groundedness(
    model,
    question: str,
    answer: str,
    evidence: str,
) -> HallucinationEvaluation:
    evaluator = model.with_structured_output(
        HallucinationEvaluation,
        method="json_mode",
    )

    return evaluator.invoke(f"""
You are a strict groundedness evaluator for an enterprise IT support assistant.

Evaluate whether the answer is supported by the supplied evidence.

IMPORTANT: Return your evaluation as valid JSON.
The JSON object must contain these fields:
- verdict: "grounded", "partially_grounded", or "unsupported"
- score: a number between 0.0 and 1.0
- unsupported_claims: a list of unsupported claims
- explanation: a concise explanation

Scoring:
- 1.0: All factual claims are supported by the evidence.
- 0.5: Some claims are supported, but some are unsupported or unverifiable.
- 0.0: The answer is largely unsupported by the evidence.

Do not treat plausible statements as supported unless the evidence supports them.
Evaluate factual claims, not writing style.
Do not penalize reasonable paraphrasing of the evidence.

Question:
{question}

Answer to evaluate:
{answer}

Evidence:
{evidence}

Return only the JSON evaluation object.
""")
