
import logging
from datetime import datetime, timezone

from anomaly_store import (
    get_anomaly_table_client,
    build_entity
)

logging.basicConfig(level=logging.INFO)


def main():

    table = get_anomaly_table_client()

    print("\n1. Creating temporary test anomaly...")

    entity = build_entity(
        timestamp=datetime.now(timezone.utc),
        event_name="persistence_test",
        detection_method="PersistenceTest",
        severity="WARNING",
        metric="TestMetric",
        actual_value=150.0,
        expected_value=100.0,
        baseline_std=10.0,
        z_score=5.0,
        anomaly_score=5.0,
        features_used=["TestMetric"]
    )

    partition_key = entity["PartitionKey"]
    row_key = entity["RowKey"]

    try:
        # Write test entity
        table.upsert_entity(entity=entity)

        print("SUCCESS: Test anomaly inserted.")

        # Read test entity
        print("\n2. Reading test anomaly...")

        stored_entity = table.get_entity(
            partition_key=partition_key,
            row_key=row_key
        )

        print("SUCCESS: Test anomaly retrieved.")
        print("Event:", stored_entity["EventName"])
        print("Metric:", stored_entity["MetricName"])
        print("Actual value:", stored_entity["ActualValue"])
        print("Severity:", stored_entity["Severity"])
        print("Status:", stored_entity["Status"])

        # Verify expected values
        assert stored_entity["EventName"] == "persistence_test"
        assert stored_entity["ActualValue"] == 150.0
        assert stored_entity["Status"] == "New"

        print("\n3. Deleting temporary test anomaly...")

    finally:
        # Clean up even if verification fails
        table.delete_entity(
            partition_key=partition_key,
            row_key=row_key
        )

        print("SUCCESS: Temporary test anomaly deleted.")

    print("\nPersistence test completed successfully.")


if __name__ == "__main__":
    main()
