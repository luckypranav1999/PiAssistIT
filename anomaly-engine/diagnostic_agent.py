
import os
import json
import logging
import re
from datetime import datetime, timezone

from azure.identity import DefaultAzureCredential
from azure.data.tables import TableServiceClient, UpdateMode
from azure.core.exceptions import ResourceNotFoundError

from langchain_openai import ChatOpenAI
from metric_loader import load_local_settings


logger = logging.getLogger(__name__)

ANOMALY_TABLE = "PiAssistAnomalies"
DIAGNOSIS_TABLE = "PiAssistDiagnoses"


def get_table_clients():
    """Connect to anomaly and diagnosis tables using Azure identity."""

    load_local_settings()

    endpoint = os.getenv("TABLE_STORAGE_ENDPOINT")

    if not endpoint:
        raise ValueError("TABLE_STORAGE_ENDPOINT is missing.")

    credential = DefaultAzureCredential()

    service = TableServiceClient(
        endpoint=endpoint,
        credential=credential
    )

    anomaly_client = service.get_table_client(ANOMALY_TABLE)
    diagnosis_client = service.create_table_if_not_exists(
        table_name=DIAGNOSIS_TABLE
    )

    return anomaly_client, diagnosis_client


def get_ai_model():
    """Initialize the existing Azure AI model configuration."""

    load_local_settings()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("AZURE_AI_MODEL", "gpt-4.1-mini")

    if not api_key or not base_url:
        raise ValueError(
            "OPENAI_API_KEY or OPENAI_BASE_URL is missing."
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_retries=2,
        timeout=45
    )


def clean_json_response(content):
    """Extract JSON from the model response."""

    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    content = str(content).strip()

    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    )

    return json.loads(content)


def build_diagnosis(anomaly, llm):
    """Generate an evidence-based diagnosis for one anomaly."""

    anomaly_data = {
        "event_name": anomaly.get("EventName"),
        "timestamp": str(anomaly.get("Timestamp")),
        "severity": anomaly.get("Severity"),
        "detection_method": anomaly.get("DetectionMethod"),
        "metric": anomaly.get("MetricName"),
        "actual_value": anomaly.get("ActualValue"),
        "expected_value": anomaly.get("ExpectedValue"),
        "baseline_std": anomaly.get("BaselineStd"),
        "z_score": anomaly.get("ZScore"),
        "anomaly_score": anomaly.get("AnomalyScore"),
        "features_used": anomaly.get("FeaturesUsed")
    }

    prompt = f"""
You are PiAssistIT's AI infrastructure diagnostic agent.

Analyze the following telemetry anomaly.

ANOMALY:
{json.dumps(anomaly_data, indent=2, default=str)}

Return ONLY valid JSON with these exact keys:

{{
  "summary": "One sentence describing the anomaly",
  "probable_root_cause": "Most likely explanation, or insufficient evidence",
  "business_impact": "Potential operational impact",
  "confidence": "Low, Medium, or High",
  "recommended_actions": [
    "Specific troubleshooting action 1",
    "Specific troubleshooting action 2",
    "Specific troubleshooting action 3"
  ],
  "evidence": [
    "Telemetry evidence supporting the diagnosis"
  ],
  "additional_checks": [
    "Checks needed to confirm the root cause"
  ]
}}

Rules:
- Use only the supplied telemetry as evidence.
- Do not invent infrastructure failures, deployments, or incidents.
- Clearly distinguish observed facts from hypotheses.
- A statistical anomaly does not automatically mean an outage.
- If evidence is insufficient, explicitly say so.
- Recommendations must be practical and relevant to the event.
- Keep the response concise.
"""

    response = llm.invoke(prompt)

    result = clean_json_response(response.content)

    required_fields = [
        "summary",
        "probable_root_cause",
        "business_impact",
        "confidence",
        "recommended_actions",
        "evidence",
        "additional_checks"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"AI response is missing required field: {field}"
            )

    return result


def diagnose_new_anomalies():
    """Diagnose undiagnosed anomalies and persist their results."""

    anomaly_client, diagnosis_client = get_table_clients()
    llm = get_ai_model()

    anomalies = anomaly_client.query_entities(
        query_filter="Status eq 'New'"
    )

    processed = 0
    skipped = 0
    failed = 0

    for anomaly in anomalies:

        partition_key = anomaly["PartitionKey"]
        row_key = anomaly["RowKey"]

        # Do not diagnose the same anomaly repeatedly.
        try:
            diagnosis_client.get_entity(
                partition_key=partition_key,
                row_key=row_key
            )
            skipped += 1
            continue

        except ResourceNotFoundError:
            pass

        try:
            logger.info(
                "Diagnosing anomaly: %s | %s",
                anomaly.get("EventName"),
                row_key
            )

            diagnosis = build_diagnosis(anomaly, llm)

            entity = {
                "PartitionKey": partition_key,
                "RowKey": row_key,
                "EventName": str(anomaly.get("EventName", "")),
                "Severity": str(anomaly.get("Severity", "")),
                "DetectionMethod": str(
                    anomaly.get("DetectionMethod", "")
                ),
                "Summary": str(diagnosis["summary"]),
                "ProbableRootCause": str(
                    diagnosis["probable_root_cause"]
                ),
                "BusinessImpact": str(diagnosis["business_impact"]),
                "Confidence": str(diagnosis["confidence"]),
                "RecommendedActions": json.dumps(
                    diagnosis["recommended_actions"]
                ),
                "Evidence": json.dumps(diagnosis["evidence"]),
                "AdditionalChecks": json.dumps(
                    diagnosis["additional_checks"]
                ),
                "DiagnosisStatus": "Completed",
                "CreatedAtUtc": datetime.now(timezone.utc)
            }

            diagnosis_client.upsert_entity(
                entity=entity,
                mode=UpdateMode.REPLACE
            )

            processed += 1


            logger.info(
                "Diagnosis saved successfully: %s",
                row_key
            )

            # Send AI diagnosis to Log Analytics.
            diagnosis_event = {
                "event_name": "piassist_ai_diagnosis",
                "event_id": row_key,
                "event_name_source": anomaly.get("EventName"),
                "severity": anomaly.get("Severity"),
                "detection_method": anomaly.get("DetectionMethod"),
                "summary": diagnosis["summary"],
                "probable_root_cause": diagnosis["probable_root_cause"],
                "business_impact": diagnosis["business_impact"],
                "confidence": diagnosis["confidence"],
                "recommended_actions": diagnosis["recommended_actions"],
                "evidence": diagnosis["evidence"],
                "diagnosis_status": "Completed"
            }

            logger.info(
                "PIASSIST_DIAGNOSIS %s",
                json.dumps(diagnosis_event, default=str)
            )


        except Exception:
            failed += 1
            logger.exception(
                "Failed to diagnose anomaly: %s",
                row_key
            )

    logger.info(
        "Diagnostic agent completed | Diagnosed: %s | "
        "Already diagnosed: %s | Failed: %s",
        processed,
        skipped,
        failed
    )

    return {
        "diagnosed": processed,
        "skipped": skipped,
        "failed": failed
    }
