
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever
from app.rag.evaluation import evaluate_groundedness

logger = logging.getLogger("piassist")
settings = get_settings()

_llm = None
_web_search = None


# --------------------------------------------------
# 1. LLM INITIALIZATION
# --------------------------------------------------

def llm():
    global _llm

    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")

        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    return _llm


# --------------------------------------------------
# 2. WEB SEARCH INITIALIZATION
# --------------------------------------------------

def web_search_tool():
    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


# --------------------------------------------------
# 3. TRACE HELPER
# --------------------------------------------------

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


# --------------------------------------------------
# 4. ROUTE THE USER QUESTION
# --------------------------------------------------

def route_question(state: AgentState):
    router = llm().with_structured_output(
        RouteDecision,
        method="json_mode",
    )

    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.

Use "kb" for questions about:
- Company IT policies
- VPN and password resets
- MFA and account access
- Laptop setup and software access
- Security, email, devices, and troubleshooting

Use "direct" only for greetings, thanks, or casual chat
that does not require company knowledge.

Question: {state["question"]}

Return valid JSON like:
{{"route": "kb"}}
""")

    return {
        "source_used": decision.route,
        "trace": add_trace(
            state,
            f"Router → {decision.route.upper()}",
        ),
    }


def route_after_router(
    state: AgentState,
) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"

    return "direct_answer"


# --------------------------------------------------
# 5. RETRIEVE DOCUMENTS FROM PRIVATE KNOWLEDGE BASE
# --------------------------------------------------

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks",
        ),
    }


# --------------------------------------------------
# 6. GRADE PRIVATE KNOWLEDGE BASE EVIDENCE
# --------------------------------------------------

def grade_kb(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"{doc.page_content}"
        for doc in state["kb_docs"]
    )

    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.

Question:
{state["question"]}

Private company knowledge base evidence:
{context}

Return "good" only if the evidence is sufficient to answer
the question confidently and specifically.

Otherwise return "weak".

Return valid JSON:
{{"grade": "good"}}
or
{{"grade": "weak"}}
""")

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"

    return "search_web"


# --------------------------------------------------
# 7. SEARCH THE WEB IF PRIVATE EVIDENCE IS INSUFFICIENT
# --------------------------------------------------

def search_web(state: AgentState):
    result = web_search_tool().invoke(
        {"query": state["current_query"]}
    )

    lines = []
    citations = []

    if isinstance(result, dict):
        if result.get("answer"):
            lines.append(
                "Search answer: " + result["answer"]
            )

        for item in result.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            citations.append({
                "title": title or url,
                "url": url,
                "type": "web",
            })
    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search",
        ),
    }


# --------------------------------------------------
# 8. GRADE WEB EVIDENCE
# --------------------------------------------------

def grade_web(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    grade = grader.invoke(f"""
You grade web evidence for an enterprise IT support assistant.

Question:
{state["question"]}

Web evidence:
{state["web_results"]}

Return "good" if the evidence is sufficient and directly
relevant to the question. Otherwise return "weak".

Return valid JSON:
{{"grade": "good"}}
or
{{"grade": "weak"}}
""")

    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()}",
        ),
    }


def after_web(
    state: AgentState,
) -> Literal[
    "generate_from_web",
    "rewrite_query",
    "insufficient",
]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


# --------------------------------------------------
# 9. REWRITE QUERY IF EVIDENCE IS STILL INSUFFICIENT
# --------------------------------------------------

def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
Rewrite this IT support question for better private
knowledge retrieval and vendor web search.

Preserve the original intent.
Add useful technical keywords.
Do not answer the question.
Return only the rewritten query.

Question:
{state["question"]}
""").content.strip()

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}",
        ),
    }


# --------------------------------------------------
# 10. GENERATE ANSWER USING PRIVATE KNOWLEDGE BASE
# --------------------------------------------------

def generate_from_kb(state: AgentState):
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
        f"{doc.page_content}"
        for doc in state["kb_docs"]
    )

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

Answer ONLY from the private company knowledge base below.

Be concise, actionable, and safe.
If steps are present, present them clearly.
Do not invent policy details.
If the evidence does not support a detail, do not make it up.

Mention that the answer is based on the company's
private knowledge base.

Question:
{state["question"]}

Private knowledge base:
{context}
""").content

    citations = []
    seen = set()

    for doc in state["kb_docs"]:
        source = doc.metadata.get("source", "Private KB")

        if source not in seen:
            seen.add(source)

            citations.append({
                "title": source.split("/")[-1],
                "url": "",
                "type": "private_kb",
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


# --------------------------------------------------
# 11. GENERATE ANSWER USING WEB EVIDENCE
# --------------------------------------------------

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

The private company knowledge base was insufficient.

Answer ONLY from the web evidence provided below.
Clearly say this is external web information and may need
IT validation before changing company-managed systems.

Do not invent details unsupported by the evidence.
If the evidence is insufficient, say so.

Question:
{state["question"]}

Web evidence:
{state["web_results"]}
""").content

    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH",
        ),
    }


# --------------------------------------------------
# 12. HALLUCINATION / GROUNDEDNESS EVALUATION
# --------------------------------------------------

def evaluate_answer(state: AgentState):
    """
    Evaluate whether the generated answer is supported
    by the evidence used to produce it.

    This node runs after private-KB or web answer generation.
    """

    source = state.get("source_used", "")

    # Select the evidence used to generate the answer.
    if source == "private_kb":
        evidence = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"{doc.page_content}"
            for doc in state.get("kb_docs", [])
        )

    elif source == "web_search":
        evidence = state.get("web_results", "")

    else:
        return {
            "hallucination_status": "skipped",
            "hallucination_score": -1.0,
            "unsupported_claims": [],
            "evaluation_explanation": (
                "Groundedness evaluation skipped because "
                "this answer did not use private KB or web evidence."
            ),
            "trace": add_trace(
                state,
                "Groundedness evaluation → SKIPPED",
            ),
        }

    # Do not evaluate an empty answer or empty evidence.
    if not state.get("answer") or not evidence.strip():
        return {
            "hallucination_status": "skipped",
            "hallucination_score": -1.0,
            "unsupported_claims": [],
            "evaluation_explanation": (
                "Evaluation skipped because the answer or evidence was empty."
            ),
            "trace": add_trace(
                state,
                "Groundedness evaluation → SKIPPED (missing answer/evidence)",
            ),
        }

    try:
        result = evaluate_groundedness(
            model=llm(),
            question=state["question"],
            answer=state["answer"],
            evidence=evidence,
        )

        logger.info(
           "Groundedness evaluation completed",
           extra={
            "custom_dimensions": {
            "hallucination_status": result.verdict,
            "hallucination_score": result.score,
            "unsupported_claims_count": len(result.unsupported_claims),
            "unsupported_claims": result.unsupported_claims,
            "evaluation_explanation": result.explanation,
            "source_used": source,
        }
    },
)
        print(
            f"APP INSIGHTS TEST: verdict={result.verdict}, "
            f"score={result.score}"
        )


        return {
            "hallucination_status": result.verdict,
            "hallucination_score": result.score,
            "unsupported_claims": result.unsupported_claims,
            "evaluation_explanation": result.explanation,
            "trace": add_trace(
                state,
                (
                    "Groundedness evaluation → "
                    f"{result.verdict.upper()} "
                    f"(score: {result.score:.2f})"
                ),
            ),
        }


    except Exception as exc:
        import traceback

        print("\n===== GROUNDEDNESS EVALUATION ERROR =====")
        traceback.print_exc()
        print("Error details:", repr(exc))
        print("=========================================\n")

        return {
            "hallucination_status": "evaluation_error",
            "hallucination_score": -1.0,
            "unsupported_claims": [],
            "evaluation_explanation": (
                f"Groundedness evaluation failed: {exc!r}"
            ),
            "trace": add_trace(
                state,
                "Groundedness evaluation → ERROR",
            ),
        }



# --------------------------------------------------
# 13. DIRECT RESPONSE FOR GREETINGS / CASUAL CHAT
# --------------------------------------------------

def direct_answer(state: AgentState):
    answer = llm().invoke(
        f"Respond briefly and naturally to: {state['question']}"
    ).content

    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }


# --------------------------------------------------
# 14. FALLBACK WHEN RELIABLE EVIDENCE CANNOT BE FOUND
# --------------------------------------------------

def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the "
            "company knowledge base or external search to answer "
            "confidently. Please contact the IT help desk or "
            "provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence",
        ),
    }


# --------------------------------------------------
# 15. BUILD THE LANGGRAPH WORKFLOW
# --------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "evaluate_answer": evaluate_answer,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, function in nodes.items():
        graph.add_node(name, function)

    # Entry point
    graph.add_edge(START, "route_question")

    # Router
    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    # Private KB retrieval and evidence grading
    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    # Web fallback and evidence grading
    graph.add_edge("search_web", "grade_web")

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # Retry path
    graph.add_edge("rewrite_query", "retrieve_kb")

    # Evaluate generated answers before returning them
    graph.add_edge("generate_from_kb", "evaluate_answer")
    graph.add_edge("generate_from_web", "evaluate_answer")
    graph.add_edge("evaluate_answer", END)

    # Other paths
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)

    return graph.compile()


# Compile the graph only after all node functions are defined.
agent_graph = build_graph()


# --------------------------------------------------
# 16. PUBLIC ENTRY POINT
# --------------------------------------------------

def ask(question: str):
    initial_state: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],

        # Groundedness evaluation fields
        "hallucination_status": "not_evaluated",
        "hallucination_score": -1.0,
        "unsupported_claims": [],
        "evaluation_explanation": "",
    }

    return agent_graph.invoke(initial_state)
