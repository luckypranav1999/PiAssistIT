
import os
import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.data.tables import TableServiceClient


# --------------------------------------------------
# 1. Load local Azure Functions settings
# --------------------------------------------------

def load_local_settings():
    """
    Load Values from local.settings.json.
    Existing environment variables take precedence.
    """

    settings_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "local.settings.json"
    )

    if not os.path.exists(settings_path):
        logging.warning(
            "local.settings.json not found. "
            "Using existing environment variables."
        )
        return

    with open(settings_path, "r", encoding="utf-8") as file:
        settings = json.load(file)

    values = settings.get("Values", {})

    for key, value in values.items():
        if value is not None:
            os.environ.setdefault(key, str(value))

    logging.info("Local Azure settings loaded successfully.")


# --------------------------------------------------
# 2. Create Azure Table Storage client
# --------------------------------------------------

def get_table_client():
    """
    Create an authenticated Azure Table Storage client
    using DefaultAzureCredential.
    """

    load_local_settings()

    endpoint = os.environ.get("TABLE_STORAGE_ENDPOINT")

    table_name = os.environ.get(
        "TABLE_NAME",
        "PiAssistMetrics"
    )

    if not endpoint:
        raise ValueError(
            "TABLE_STORAGE_ENDPOINT is missing. "
            "Check local.settings.json."
        )

    credential = DefaultAzureCredential()

    service_client = TableServiceClient(
        endpoint=endpoint,
        credential=credential
    )

    table_client = service_client.get_table_client(
        table_name=table_name
    )

    logging.info(
        "Connected to Azure Table Storage: %s",
        table_name
    )

    return table_client


# --------------------------------------------------
# 3. Load historical telemetry records
# --------------------------------------------------

def load_metrics(days=30):
    """
    Load historical aggregated telemetry from Azure Table Storage.

    Args:
        days: Number of previous days to retrieve.

    Returns:
        Pandas DataFrame containing telemetry records.
    """

    if days < 1:
        raise ValueError("days must be at least 1.")

    table_client = get_table_client()

    start_date = (
        datetime.now(timezone.utc) - timedelta(days=days)
    )

    start_partition = start_date.strftime("%Y%m%d")

    query_filter = (
        f"PartitionKey ge '{start_partition}'"
    )

    logging.info(
        "Loading telemetry for the previous %s days.",
        days
    )

    entities = table_client.query_entities(
        query_filter=query_filter
    )

    records = []

    for entity in entities:

        records.append({
            "PartitionKey": entity.get("PartitionKey"),
            "RowKey": entity.get("RowKey"),
            "WindowStartUtc": entity.get("WindowStartUtc"),
            "EventName": entity.get("EventName"),
            "EventCount": entity.get("EventCount"),
            "AvgDurationMs": entity.get("AvgDurationMs"),
            "MaxDurationMs": entity.get("MaxDurationMs"),
            "AvgHallucinationScore": entity.get(
                "AvgHallucinationScore"
            ),
            "UnsupportedClaims": entity.get(
                "UnsupportedClaims"
            ),
            "AvgRetrievedDocuments": entity.get(
                "AvgRetrievedDocuments"
            ),
            "TotalTokens": entity.get("TotalTokens")
        })

    if not records:
        logging.warning("No telemetry records found.")
        return pd.DataFrame()

    # Convert records into DataFrame
    df = pd.DataFrame(records)

    # Convert timestamps to UTC datetime
    df["WindowStartUtc"] = pd.to_datetime(
        df["WindowStartUtc"],
        utc=True,
        errors="coerce"
    )

    # Convert metric columns to numeric values
    numeric_columns = [
        "EventCount",
        "AvgDurationMs",
        "MaxDurationMs",
        "AvgHallucinationScore",
        "UnsupportedClaims",
        "AvgRetrievedDocuments",
        "TotalTokens"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Sort records chronologically
    df = df.sort_values(
        by=["WindowStartUtc", "EventName"]
    ).reset_index(drop=True)

    logging.info(
        "Successfully loaded %s telemetry records.",
        len(df)
    )

    return df


# --------------------------------------------------
# 4. Display telemetry summary
# --------------------------------------------------

def display_metrics_summary(df):

    if df.empty:
        print("\nNo telemetry data available.")
        return

    print("\n" + "=" * 70)
    print("       PiAssistIT - Historical Telemetry Metrics")
    print("=" * 70)

    print("\nTotal records:", len(df))

    print("\nMetric records:")
    print(
        df.to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("Records by event type")
    print("-" * 70)

    print(
        df["EventName"].value_counts().to_string()
    )

    print("\n" + "-" * 70)
    print("Average duration by event type (ms)")
    print("-" * 70)

    if "AvgDurationMs" in df.columns:
        print(
            df.groupby("EventName")["AvgDurationMs"]
            .mean()
            .round(2)
            .to_string()
        )

    print("\n" + "=" * 70)


# --------------------------------------------------
# 5. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    try:

        metrics_df = load_metrics(days=30)

        display_metrics_summary(metrics_df)

    except Exception:

        logging.exception(
            "Failed to load PiAssistIT telemetry."
        )

        raise
