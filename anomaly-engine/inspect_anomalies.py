
from azure.identity import DefaultAzureCredential
from azure.data.tables import TableServiceClient

ENDPOINT = "https://piassistitanomalyenb923.table.core.windows.net/"
TABLE_NAME = "PiAssistAnomalies"

credential = DefaultAzureCredential()

service = TableServiceClient(
    endpoint=ENDPOINT,
    credential=credential
)

table = service.get_table_client(TABLE_NAME)

entities = list(table.list_entities())

entities.sort(
    key=lambda x: (
        x.get("WindowStartUtc").isoformat()
        if x.get("WindowStartUtc")
        else ""
    ),
    reverse=True
)

print(f"\nTotal anomaly records: {len(entities)}")

for entity in entities:
    print("\n" + "=" * 60)

    for key, value in entity.items():
        if key not in ("PartitionKey", "RowKey", "Timestamp", "etag"):
            print(f"{key}: {value}")
