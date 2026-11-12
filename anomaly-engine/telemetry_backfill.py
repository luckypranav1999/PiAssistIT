
import json
import logging
import os

from azure.core.exceptions import ResourceNotFoundError

from anomaly_store import get_anomaly_table_client
from diagnostic_agent import get_table_clients
from azure.monitor.opentelemetry import configure_azure_monitor

from metric_loader import load_local_settings

logging.getLogger().setLevel(logging.INFO)

load_local_settings()

configure_azure_monitor(
    connection_string=os.getenv(
        "APPLICATIONINSIGHTS_CONNECTION_STRING"
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("telemetry_backfill")


def backfill_telemetry():

    anomaly_client, _ = get_table_clients()
    _, diagnosis_client = get_table_clients()

    anomalies = list(anomaly_client.list_entities())

    anomaly_count = 0
    diagnosis_count = 0

    logger.info("Found %s anomaly records.", len(anomalies))

    for anomaly in anomalies:

        partition_key = anomaly["PartitionKey"]
        row_key = anomaly["RowKey"]

        anomaly_event = {
            "event_name": "piassist_anomaly_detected",
            "event_id": row_key,
            "event_time": str(anomaly.get("Timestamp")),
            "event_name_source": anomaly.get("EventName"),
            "detection_method": anomaly.get("DetectionMethod"),
            "severity": anomaly.get("Severity"),
            "status": anomaly.get("Status"),
            "metric_name": anomaly.get("MetricName"),
            "actual_value": anomaly.get("ActualValue"),
            "expected_value": anomaly.get("ExpectedValue"),
            "z_score": anomaly.get("ZScore"),
            "anomaly_score": anomaly.get("AnomalyScore")
        }

        logger.info(
            "PIASSIST_ANOMALY %s",
            json.dumps(anomaly_event, default=str)
        )

        anomaly_count += 1

        try:
            diagnosis = diagnosis_client.get_entity(
                partition_key=partition_key,
                row_key=row_key
            )

            diagnosis_event = {
                "event_name": "piassist_ai_diagnosis",
                "event_id": row_key,
                "event_name_source": diagnosis.get("EventName"),
                "severity": diagnosis.get("Severity"),
                "detection_method": diagnosis.get("DetectionMethod"),
                "summary": diagnosis.get("Summary"),
                "probable_root_cause": diagnosis.get(
                    "ProbableRootCause"
                ),
                "business_impact": diagnosis.get("BusinessImpact"),
                "confidence": diagnosis.get("Confidence"),
                "recommended_actions": diagnosis.get(
                    "RecommendedActions"
                ),
                "evidence": diagnosis.get("Evidence"),
                "additional_checks": diagnosis.get(
                    "AdditionalChecks"
                ),
                "diagnosis_status": diagnosis.get(
                    "DiagnosisStatus"
                )
            }

            logger.info(
                "PIASSIST_DIAGNOSIS %s",
                json.dumps(diagnosis_event, default=str)
            )

            diagnosis_count += 1

        except ResourceNotFoundError:
            logger.info(
                "No diagnosis found for anomaly: %s",
                row_key
            )

    logger.info(
        "Telemetry backfill completed | Anomalies: %s | Diagnoses: %s",
        anomaly_count,
        diagnosis_count
    )


if __name__ == "__main__":
    backfill_telemetry()
