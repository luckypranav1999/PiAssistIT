
from collections import defaultdict
from datetime import timezone

from azure.identity import DefaultAzureCredential
from azure.data.tables import TableServiceClient


TABLE_ENDPOINT = "https://piassistitanomalyenb923.table.core.windows.net/"
TABLE_NAME = "PiAssistMetrics"

credential = DefaultAzureCredential()

service = TableServiceClient(
    endpoint=TABLE_ENDPOINT,
    credential=credential
)

table = service.get_table_client(TABLE_NAME)

entities = list(table.list_entities())

print("\n========== PiAssistIT METRIC DATASET ==========")
print(f"Total records: {len(entities)}")

if not entities:
    print("No records found.")
    raise SystemExit(0)

event_groups = defaultdict(list)

for entity in entities:
    event_name = entity.get("EventName", "unknown")
    event_groups[event_name].append(entity)

print("\n========== EVENT DISTRIBUTION ==========")

for event_name, records in sorted(event_groups.items()):
    timestamps = [
        record["WindowStartUtc"]
        for record in records
        if record.get("WindowStartUtc")
    ]

    timestamps.sort()

    print(f"\nEvent: {event_name}")
    print(f"Records: {len(records)}")

    if timestamps:
        print(f"First observation: {timestamps[0]}")
        print(f"Latest observation: {timestamps[-1]}")

    metric_fields = [
        "AvgDurationMs",
        "MaxDurationMs",
        "AvgHallucinationScore",
        "UnsupportedClaims",
        "AvgRetrievedDocuments",
        "TotalTokens",
    ]

    for field in metric_fields:
        valid_values = [
            record[field]
            for record in records
            if record.get(field) is not None
        ]

        if valid_values:
            print(
                f"{field}: {len(valid_values)} observations"
            )

print("\n========== DATASET INSPECTION COMPLETE ==========")
