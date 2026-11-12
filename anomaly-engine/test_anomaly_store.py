
import logging

from metric_loader import load_metrics
from statistical_detector import detect_anomalies as detect_statistical
from isolation_forest_detector import detect_anomalies as detect_isolation_forest
from anomaly_store import save_anomalies


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main():

    print("\nLoading historical telemetry...")

    metrics = load_metrics(days=30)

    if metrics.empty:
        print("No telemetry records available.")
        return

    print(f"Loaded {len(metrics)} metric records.")

    print("\nRunning Statistical Detector...")
    statistical_results = detect_statistical(metrics)

    print(
        f"Statistical anomalies: {len(statistical_results)}"
    )

    print("\nRunning Isolation Forest...")
    isolation_results = detect_isolation_forest(metrics)

    print(
        f"Isolation Forest anomalies: {len(isolation_results)}"
    )

    print("\nPersisting anomalies to Azure Table Storage...")

    saved = save_anomalies(
        statistical_anomalies=statistical_results,
        isolation_anomalies=isolation_results
    )

    print(f"\nCompleted. Records persisted: {saved}")


if __name__ == "__main__":
    main()
