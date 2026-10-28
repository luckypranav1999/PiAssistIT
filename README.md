# PiAssistIT — Enterprise IT Support Agentic RAG Copilot

PiAssistIT is an IT-support copilot that answers employee questions using a private knowledge base first, falls back to web search when the private evidence is weak, and records an observable decision trace. The project combines retrieval-augmented generation (RAG), a conditional LangGraph workflow, answer-groundedness evaluation, a FastAPI web application, and Azure telemetry.

> **Project status:** This README describes the current repository implementation and the Azure deployment/monitoring setup demonstrated for this project. The sample company and its IT policies are fictional. This is a portfolio/demo project, not a production-certified enterprise service.

## What it does

- Routes greetings and simple conversation to a direct-answer path.
- Retrieves relevant chunks from a Pinecone-backed private knowledge base for IT-support questions.
- Grades private evidence before deciding whether to answer from it or try web search.
- Uses Tavily when private evidence is weak, then grades the web evidence.
- Rewrites the query and retries retrieval when web evidence remains weak, subject to the configured retry limit.
- Generates an answer from the selected evidence and records citations/trace information where available.
- Evaluates groundedness after private-KB or web-based answer generation, recording a verdict, score, unsupported claims, and explanation.
- Exposes chat and document-ingestion APIs through FastAPI.
- Initializes Microsoft OpenTelemetry/Azure Monitor telemetry for Application Insights, and uses application logs and the audit database for troubleshooting.

## Architecture

```text
Employee / Browser
        |
        v
HTML, CSS, JavaScript UI
        |  POST /api/chat
        v
FastAPI application ----------------------> SQLite audit log
        |
        v
LangGraph workflow
        |
        +--> Direct answer (greetings/simple chat) ------------------> End
        |
        +--> Retrieve private knowledge base (Pinecone)
                 |
                 v
            Grade KB evidence
              /          \
           GOOD           WEAK
            |               |
            v               v
      Generate from KB   Tavily web search
            |               |
            |           Grade web evidence
            |             /        \
            |          GOOD         WEAK
            |            |             |
            |            v             v
            |       Generate web   Rewrite query and retry
            |            |          or insufficient evidence
            |            v
            +----> Groundedness evaluation
                         |
                         v
                   Response + trace

FastAPI / application logs and telemetry
        |
        v
Microsoft OpenTelemetry -> Azure Monitor / Application Insights
        |
        v
Workbook and configured alert rules
```

The diagram is conceptual: direct-answer and insufficient-evidence paths end without running the groundedness evaluator. The evaluator is wired after KB-based and web-based answer generation. The workflow is a **LangGraph-based agentic RAG workflow**; the repository does not establish a separate multi-agent architecture or an implemented MCP integration.

## Workflow details

1. **Route question:** An LLM-backed structured decision selects `kb` or `direct`.
2. **Retrieve:** For knowledge questions, retrieve up to the configured number of chunks from the private Pinecone namespace.
3. **Grade private evidence:** Structured grading determines whether the private evidence is good or weak.
4. **Choose a path:** Good evidence goes to KB answer generation. Weak evidence triggers Tavily web search.
5. **Grade web evidence:** Good web evidence is used to generate a web-based answer. Weak evidence can trigger query rewriting and a retry; after the retry limit, the workflow returns an insufficient-evidence response.
6. **Evaluate groundedness:** KB- and web-generated answers are evaluated against their supporting evidence. The result is added to the trace and telemetry/logging flow.
7. **Return response:** The API returns the answer, source, trace, citations, and related evaluation fields available from the workflow.

### Important evaluation limitation

Groundedness evaluation is **not currently run on every possible response**. It runs after `generate_from_kb` and `generate_from_web`, but not after the direct-answer or insufficient-evidence nodes. Treat the score as a measure for evaluated answers, not as a guarantee that every response is hallucination-free.

## Technology stack

| Layer | Technology | Role |
|---|---|---|
| Workflow orchestration | LangGraph | Stateful nodes, conditional routing, retries |
| LLM interface | LangChain `ChatOpenAI` with an OpenAI-compatible endpoint | Routing, evidence grading, query rewriting, answer generation, groundedness evaluation |
| Model endpoint | Azure AI Foundry-compatible endpoint when configured in environment | Hosts the configured chat and embedding deployments |
| Embeddings | `OpenAIEmbeddings` with configured embedding model | Converts document/query text into vectors |
| Vector database | Pinecone | Stores and retrieves private knowledge-base chunks |
| Web search | Tavily | Fallback source when private evidence is weak |
| API | FastAPI + Uvicorn | Serves the application and REST endpoints |
| Frontend | HTML, CSS, JavaScript, Jinja templates | Chat interface and document upload UI |
| Audit | SQLite | Stores application audit records |
| Telemetry | Microsoft OpenTelemetry / Azure Monitor / Application Insights | Application telemetry and monitoring |
| Hosting / CI | Azure App Service and GitHub Actions | Hosting and repository-driven deployment workflow |

**Model configuration note:** The code reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` from configuration. Set these to the correct Azure AI Foundry endpoint, key, and deployment/model name for your resource. The embedding client also uses the configured endpoint and `EMBEDDING_MODEL`. Merely creating a deployment in Foundry does not prove the running app is using it; verify the App Service configuration and live traces.

## Repository structure

```text
app/
├── api/routes.py          # Chat, health, and ingestion routes
├── core/config.py         # Environment-backed settings
├── core/logging.py        # Logging configuration
├── rag/state.py           # LangGraph state
├── rag/vectorstore.py     # Pinecone and embeddings
├── rag/workflow.py        # Workflow nodes and graph edges
├── services/audit.py      # SQLite audit support
├── services/ingestion.py  # Document loading and chunking
└── main.py                # FastAPI app and telemetry initialization

data/sample_kb/            # Sample IT knowledge documents
static/                    # CSS and JavaScript
 templates/                 # HTML templates
requirements.txt
Dockerfile
run.py
README.md
```

## Prerequisites

- Python 3.11+ (use the version supported by the current dependency set; the deployed App Service was configured for Python 3.12).
- A Pinecone account and index access.
- An OpenAI-compatible chat/embedding endpoint. For the Azure setup, configure the Azure AI Foundry endpoint and deployments.
- A Tavily API key for the web-search fallback.
- Azure Application Insights/Azure Monitor configuration if you want telemetry in Azure.

## Local setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/luckypranav1999/PiAssistIT.git
cd PiAssistIT
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the repository root. Use your own credentials and endpoint values; never commit this file.

```dotenv
# Azure AI Foundry / OpenAI-compatible endpoint
OPENAI_API_KEY=your_endpoint_key
OPENAI_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small

# Private knowledge base
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=fde-it-support-rag
PINECONE_NAMESPACE=company-it-kb

# Web-search fallback
TAVILY_API_KEY=your_tavily_api_key

# Protected ingestion endpoint
ADMIN_API_KEY=replace_with_a_long_random_secret

# Optional settings
APP_ENV=development
TOP_K=4
MAX_RETRIES=1
```

Use the **exact base URL and deployment/model identifiers shown by your Azure AI Foundry resource**. Azure endpoint URL formats can differ by deployment/API configuration. If you are using a different OpenAI-compatible provider, configure its endpoint and key accordingly. Do not add credentials to source code or screenshots.

The settings class also supports `AUDIT_DB_PATH` and `UPLOAD_DIR` for overriding the audit database and upload directory. Check `app/core/config.py` for the current defaults and supported settings.

### 4. Load the sample knowledge base

```bash
python ingest_sample_kb.py
```

This loads the sample documents, splits them into chunks, creates embeddings, and adds the chunks to the configured Pinecone namespace. Confirm your index dimension matches the selected embedding model before using an existing index.

### 5. Start the app

```bash
python run.py
```

Open `http://127.0.0.1:8000`. FastAPI's interactive API documentation is available at `http://127.0.0.1:8000/docs`.

If startup fails, check the terminal logs, required environment variables, dependency versions, network access, and provider endpoint format.

## Demo scenarios

### Private knowledge-base answer

Ask: **“How do I connect to the company VPN from home?”**

Expected behavior: route to KB → retrieve private chunks → grade evidence → answer from private KB → groundedness evaluation.

### Policy question

Ask: **“Can IT support ask me to share my MFA code?”**

Expected behavior: answer from the private handbook if relevant evidence is retrieved. The answer should reflect the actual content of the sample policy document.

### External information fallback

Ask: **“What is the latest Microsoft Teams outage guidance?”**

If the private evidence is weak, the workflow can search Tavily, grade the returned evidence, and generate an answer with an external-information caveat.

### Query rewrite / retry

Ask a vague troubleshooting question. If the evidence path remains weak, the graph can rewrite the query and retry, subject to `MAX_RETRIES`; otherwise it returns an insufficient-evidence response.

### Direct conversation

Ask: **“Hello!”**

Expected behavior: route to the direct-answer node without private retrieval or web search. This path currently bypasses groundedness evaluation.

## API endpoints

### `GET /api/health`

Health-check endpoint.

### `POST /api/chat`

Example request:

```json
{
  "question": "How do I reset my company password?"
}
```

The response includes fields such as `answer`, `source_used`, `trace`, `citations`, and evaluation details when produced by the workflow. Exact fields depend on the current API schema and execution path.

### `POST /api/ingest`

Uploads a supported document and adds its chunks to the vector store. The route requires the `X-Admin-Key` header to match the configured `ADMIN_API_KEY`. Check the API schema and `app/api/routes.py` for accepted upload types and payload details.

## Azure deployment and monitoring

The project has been deployed to Azure App Service, with Microsoft OpenTelemetry initialization in `app/main.py` intended to send application telemetry to Azure Monitor/Application Insights. The deployment's startup command has been configured as:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For your own deployment:

1. Create/configure the App Service with a supported Python runtime.
2. Configure the required environment variables in **App Service → Environment variables**. Do not put secrets in the repository.
3. Confirm the startup command and install dependencies from `requirements.txt` during deployment.
4. Confirm the app starts and the health endpoint responds.
5. Verify that requests and application logs arrive in the correct Application Insights resource.
6. Use the workbook/query setup in your Azure environment to inspect groundedness evaluation events and flagged responses.
7. Validate alert rules with known test cases before relying on them.

### Monitoring metrics shown in the demo

The Azure workbook has displayed counts/rates for grounded and partially grounded evaluations, plus flagged evaluations. These values describe the events collected at the time of the screenshot/demo; they are **not fixed project performance benchmarks** and will change as new questions are tested.

A hallucination/groundedness alert is useful as a signal for investigation, not proof that every unflagged answer is correct. Ensure the telemetry query, evaluation coverage, alert threshold, and notification action are configured and tested for your own environment.

## Security and limitations

- Keep `.env`, API keys, admin keys, and Azure connection details out of Git.
- Replace the default admin key with a strong secret and restrict access to the ingestion endpoint.
- The sample knowledge base is fictional; do not upload confidential company documents to a demo without approval and appropriate access controls.
- Web search results may be stale or incorrect; treat external answers as needing verification.
- Groundedness evaluation is only executed on the KB and web answer-generation branches in the current graph.
- SQLite is a lightweight audit option; assess persistence, access, retention, and concurrency requirements before production use.
- This project demonstrates agentic workflow patterns, not a separately verified multi-agent system. MCP is not documented as an implemented feature.
- Before sharing the deployed site publicly, verify the browser's security warning, TLS configuration, domain reputation, and App Service settings.

## Troubleshooting checklist

- **Missing configuration:** check that all required environment variables are present in local `.env` or App Service settings.
- **Model/API errors:** verify the endpoint URL, key, API compatibility, model/deployment name, and that the deployment is available to the resource.
- **Embedding/vector errors:** check the embedding model's vector dimension against the Pinecone index dimension and namespace.
- **No useful private results:** confirm ingestion succeeded, the namespace matches, and sample documents were indexed.
- **No web fallback:** inspect the trace and private-evidence grade; web search is conditional, not called for every question.
- **No groundedness event:** check whether the response used the direct-answer or insufficient-evidence path, which currently bypasses evaluation.
- **No Azure telemetry:** confirm the App Insights/Azure Monitor resource and instrumentation configuration, then inspect logs and telemetry ingestion.
- **Deployment failure:** inspect GitHub Actions deployment logs and App Service startup logs.

## Interview-ready explanation

“PiAssistIT is a FastAPI-based IT support copilot built around a LangGraph agentic RAG workflow. It routes simple conversation directly, retrieves internal knowledge from Pinecone for support questions, grades evidence, and uses Tavily as a fallback when private evidence is weak. It can rewrite a query and retry when evidence remains insufficient. Answers generated from the KB or web path are evaluated for groundedness, and the application emits telemetry for Azure Monitor/Application Insights. This makes the retrieval decisions and evaluation outcomes inspectable, while the current limitation is that groundedness evaluation does not cover every response path.”

## Repository

[GitHub — luckypranav1999/PiAssistIT](https://github.com/luckypranav1999/PiAssistIT)
