
import json
import logging
import os
import re
import math
from datetime import datetime, timezone

import pandas as pd
from azure.data.tables import (
    TableServiceClient,
    UpdateMode
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

from metric_loader import load_local_settings


logger = logging.getLogger(__name__)

TABLE_NAME = "PiAssistAnomalies"


def get_anomaly_table_client():
    """Connect to Azure Table Storage and create the anomaly table."""

    load_local_settings()

    endpoint = os.getenv("TABLE_STORAGE_ENDPOINT")

    if not endpoint:
        raise ValueError(
            "TABLE_STORAGE_ENDPOINT is not configured."
        )

    credential = DefaultAzureCredential()

    service_client = TableServiceClient(
        endpoint=endpoint,
        credential=credential
    )

    table_client = service_client.create_table_if_not_exists(
        table_name=TABLE_NAME
    )

    logger.info(
        "Connected to Azure Table Storage: %s",
        TABLE_NAME
    )

    return table_client


def safe_key(value):
    """Remove invalid characters from Azure Table keys."""

    return re.sub(
        r"[/\\#?\x00-\x1f]",
        "_",
        str(value)
    )


def safe_float(value):
    """Return a finite float or None."""

    if value is None:
        return None

    try:
        number = float(value)

        if math.isfinite(number):
            return number

        return None

    except (TypeError, ValueError):
        return None


def normalize_timestamp(value):
    """Convert timestamps into timezone-aware UTC datetimes."""

    timestamp = pd.to_datetime(
        value,
        utc=True,
        errors="coerce"
    )

    if pd.isna(timestamp):
        raise ValueError(
            f"Invalid anomaly timestamp: {value}"
        )

    return timestamp.to_pydatetime()


def build_entity(
    timestamp,
    event_name,
    detection_method,
    severity,
    metric=None,
    actual_value=None,
    expected_value=None,
    baseline_std=None,
    z_score=None,
    anomaly_score=None,
    features_used=None
):
    """Build a standardized Azure Table entity."""

    timestamp = normalize_timestamp(timestamp)

    timestamp_key = timestamp.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    metric_key = safe_key(metric or "multivariate")
    event_key = safe_key(event_name)
    method_key = safe_key(detection_method)

    row_key = (
        f"{timestamp_key}_{event_key}_"
        f"{method_key}_{metric_key}"
    )

    entity = {
        "PartitionKey": timestamp.strftime("%Y%m%d"),
        "RowKey": row_key,
        "Timestamp": timestamp,
        "EventName": str(event_name),
        "DetectionMethod": str(detection_method),
        "Severity": str(severity),
        "Status": "New",
        "CreatedAtUtc": datetime.now(timezone.utc)
    }

    optional_fields = {
        "MetricName": metric,
        "ActualValue": actual_value,
        "ExpectedValue": expected_value,
        "BaselineStd": baseline_std,
        "ZScore": z_score,
        "AnomalyScore": anomaly_score
    }

    for field, value in optional_fields.items():

        if field == "MetricName":
            if value is not None and not pd.isna(value):
                entity[field] = str(value)
            continue

        numeric_value = safe_float(value)

        if numeric_value is not None:
            entity[field] = numeric_value

    if features_used is not None:

        if isinstance(features_used, (list, tuple)):
            features_used = json.dumps(list(features_used))
        else:
            features_used = str(features_used)

        entity["FeaturesUsed"] = features_used

    return entity


def normalize_statistical_anomalies(df):
    """Convert statistical detector output into storage entities."""

    entities = []

    if df is None or df.empty:
        return entities

    for _, row in df.iterrows():

        z_score = safe_float(row["ZScore"])

        anomaly_score = (
            abs(z_score)
            if z_score is not None
            else None
        )

        entity = build_entity(
            timestamp=row["Timestamp"],
            event_name=row["EventName"],
            detection_method="RollingZScore",
            severity=row["Severity"],
            metric=row["Metric"],
            actual_value=row["ActualValue"],
            expected_value=row["BaselineMean"],
            baseline_std=row["BaselineStd"],
            z_score=z_score,
            anomaly_score=anomaly_score
        )

        entities.append(entity)

    return entities


def normalize_isolation_forest_anomalies(df):
    """Convert Isolation Forest output into storage entities."""

    entities = []

    if df is None or df.empty:
        return entities

    for _, row in df.iterrows():

        entity = build_entity(
            timestamp=row["Timestamp"],
            event_name=row["EventName"],
            detection_method=row.get(
                "DetectionMethod",
                "IsolationForest"
            ),
            severity=row["Severity"],
            anomaly_score=row["AnomalyScore"],
            features_used=row.get("FeaturesUsed")
        )

        entities.append(entity)

    return entities


def save_anomalies(
    statistical_anomalies=None,
    isolation_anomalies=None
):
    """Persist anomalies without resetting existing statuses."""

    table_client = get_anomaly_table_client()

    entities = []

    entities.extend(
        normalize_statistical_anomalies(
            statistical_anomalies
        )
    )

    entities.extend(
        normalize_isolation_forest_anomalies(
            isolation_anomalies
        )
    )

    if not entities:
        logger.info("No anomalies to persist.")
        return 0

    saved_count = 0

    for entity in entities:

        partition_key = entity["PartitionKey"]
        row_key = entity["RowKey"]

        
        is_new = False

        try:
            existing = table_client.get_entity(
                partition_key=partition_key,
                row_key=row_key
            )

            # Preserve existing lifecycle information.
            entity["Status"] = existing.get(
                "Status",
                "New"
            )

            entity["CreatedAtUtc"] = existing.get(
                "CreatedAtUtc",
                entity["CreatedAtUtc"]
            )

        except ResourceNotFoundError:
            is_new = True


        table_client.upsert_entity(
            entity=entity,
            mode=UpdateMode.REPLACE
        )

        # Send only newly detected anomalies to Log Analytics.
        if is_new:
            anomaly_event = {
                "event_name": "piassist_anomaly_detected",
                "event_id": row_key,
                "event_time": str(entity["Timestamp"]),
                "event_name_source": entity["EventName"],
                "detection_method": entity["DetectionMethod"],
                "severity": entity["Severity"],
                "status": entity["Status"],
                "metric_name": entity.get("MetricName"),
                "actual_value": entity.get("ActualValue"),
                "expected_value": entity.get("ExpectedValue"),
                "z_score": entity.get("ZScore"),
                "anomaly_score": entity.get("AnomalyScore")
            }

            logger.info(
                "PIASSIST_ANOMALY %s",
                json.dumps(anomaly_event, default=str)
            )

        saved_count += 1

    logger.info(
        "Successfully persisted %s anomaly records.",
        saved_count
    )

    return saved_count
