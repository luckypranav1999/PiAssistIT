
import os
import logging
import math
from datetime import timezone

import azure.functions as func

from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.data.tables import TableServiceClient, UpdateMode
from diagnostic_agent import diagnose_new_anomalies

from metric_loader import load_metrics
from statistical_detector import (
    detect_anomalies as detect_statistical_anomalies
)
from isolation_forest_detector import (
    detect_anomalies as detect_isolation_forest
)
from anomaly_store import save_anomalies


app = func.FunctionApp()


def clean_number(value):
    """Convert numeric values safely for Azure Table Storage."""

    if value is None:
        return None

    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def create_metric_entity(row):
    """Convert a Log Analytics aggregation row into a Table entity."""

    window_start = row[0]

    if window_start.tzinfo is None:
        window_start = window_start.replace(tzinfo=timezone.utc)
    else:
        window_start = window_start.astimezone(timezone.utc)

    event_name = str(row[1])

    timestamp_key = window_start.strftime("%Y%m%dT%H%M%SZ")

    entity = {
        "PartitionKey": window_start.strftime("%Y%m%d"),
        "RowKey": f"{timestamp_key}_{event_name}",
        "WindowStartUtc": window_start,
        "EventName": event_name,
        "EventCount": int(row[2] or 0),
    }

    optional_fields = {
        "AvgDurationMs": clean_number(row[3]),
        "MaxDurationMs": clean_number(row[4]),
        "AvgHallucinationScore": clean_number(row[5]),
        "UnsupportedClaims": row[6],
        "AvgRetrievedDocuments": clean_number(row[7]),
        "TotalTokens": row[8],
    }

    for field_name, value in optional_fields.items():
        if value is not None:
            entity[field_name] = value

    return entity


def run_anomaly_detection():
    """
    Load historical metrics, run both detectors,
    and persist detected anomalies.
    """

    logging.info("Starting anomaly detection pipeline.")

    metrics = load_metrics(days=30)

    if metrics.empty:
        logging.warning(
            "Anomaly detection skipped: no historical metrics available."
        )
        return

    logging.info(
        "Loaded %s historical metric records for detection.",
        len(metrics)
    )

    # Statistical detection
    statistical_results = detect_statistical_anomalies(metrics)

    statistical_count = (
        len(statistical_results)
        if statistical_results is not None
        else 0
    )

    logging.info(
        "Statistical detector completed. Anomalies: %s",
        statistical_count
    )

    # Isolation Forest detection
    isolation_results = detect_isolation_forest(metrics)

    isolation_count = (
        len(isolation_results)
        if isolation_results is not None
        else 0
    )

    logging.info(
        "Isolation Forest completed. Anomalies: %s",
        isolation_count
    )

    # Persist both detector outputs
    saved_count = save_anomalies(
        statistical_anomalies=statistical_results,
        isolation_anomalies=isolation_results
    )

    logging.info(
        "Anomaly detection pipeline completed. "
        "Statistical: %s | Isolation Forest: %s | Persisted: %s",
        statistical_count,
        isolation_count,
        saved_count
    )


@app.timer_trigger(
    schedule="0 */5 * * * *",
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True
)
def telemetry_connectivity_test(timer: func.TimerRequest):

    logging.info("PiAssistIT telemetry aggregation started.")

    try:
        workspace_id = os.environ["WORKSPACE_ID"]

        table_endpoint = os.environ["TABLE_STORAGE_ENDPOINT"]

        table_name = os.environ.get(
            "TABLE_NAME",
            "PiAssistMetrics"
        )

        credential = DefaultAzureCredential()

        # Log Analytics client
        logs_client = LogsQueryClient(credential)

        # Azure Table Storage client
        table_service = TableServiceClient(
            endpoint=table_endpoint,
            credential=credential
        )

        table_client = table_service.create_table_if_not_exists(
            table_name=table_name
        )

        query = """
        AppTraces
        | where TimeGenerated > ago(48h)
        | extend event_name = tostring(Properties["event_name"])
        | where isnotempty(event_name)
        | extend duration_ms = coalesce(
            todouble(Properties["duration_ms"]),
            todouble(Properties["llm_duration_ms"]),
            todouble(Properties["retrieval_duration_ms"])
        )
        | extend hallucination_score = todouble(Properties["hallucination_score"])
        | extend unsupported_claims_count = toint(Properties["unsupported_claims_count"])
        | extend retrieved_documents_count = toint(Properties["retrieved_documents_count"])
        | extend total_tokens = toint(Properties["total_tokens"])
        | summarize
            event_count = count(),
            avg_duration_ms = avg(duration_ms),
            max_duration_ms = max(duration_ms),
            avg_hallucination_score = avg(hallucination_score),
            unsupported_claims = sum(unsupported_claims_count),
            avg_retrieved_documents = avg(todouble(retrieved_documents_count)),
            total_tokens = sum(tolong(total_tokens))
            by bin(TimeGenerated, 5m), event_name
        | order by TimeGenerated desc
        """

        response = logs_client.query_workspace(
            workspace_id=workspace_id,
            query=query,
            timespan=None
        )

        if response.status == LogsQueryStatus.SUCCESS:

            if response.tables:

                rows = response.tables[0].rows
                stored_count = 0

                for row in rows:

                    entity = create_metric_entity(row)

                    table_client.upsert_entity(
                        entity=entity,
                        mode=UpdateMode.REPLACE
                    )

                    stored_count += 1

                    logging.info(
                        "Stored metric | Event: %s | Window: %s",
                        entity["EventName"],
                        entity["WindowStartUtc"]
                    )

                logging.info(
                    "Metric persistence completed. Records processed: %s",
                    stored_count
                )

            else:
                logging.warning("No aggregation results found.")

            # Run anomaly detection independently.
            try:
                run_anomaly_detection()

                logging.info("Starting AI diagnostic agent.")

                diagnosis_result = diagnose_new_anomalies()

                logging.info(
                    "AI diagnostic agent completed: %s",
                    diagnosis_result
                )

            except Exception:
                logging.exception(
                    "Anomaly detection or AI diagnosis failed. "
                    "Metric aggregation remains successful."
                )

            logging.info(
                "PiAssistIT telemetry aggregation completed."
            )

        else:
            logging.error(
                "Azure Monitor Logs returned partial or unsuccessful results."
            )

    except Exception:
        logging.exception(
            "Telemetry aggregation or storage failed."
        )
        raise
