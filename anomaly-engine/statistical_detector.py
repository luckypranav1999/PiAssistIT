
import logging

import pandas as pd

from metric_loader import load_metrics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROLLING_WINDOW = 20
MIN_PERIODS = 10

WARNING_THRESHOLD = 3.0
CRITICAL_THRESHOLD = 4.5

METRIC_COLUMNS = [
    "AvgDurationMs",
    "MaxDurationMs",
    "TotalTokens",
    "AvgHallucinationScore",
    "UnsupportedClaims",
    "AvgRetrievedDocuments"
]


# --------------------------------------------------
# 1. Detect anomalies using Rolling Z-score
# --------------------------------------------------

def detect_anomalies(
    df,
    window=ROLLING_WINDOW,
    min_periods=MIN_PERIODS
):
    """
    Detect anomalies using historical rolling Z-scores.

    The current observation is excluded from its baseline.

    Returns:
        DataFrame containing detected anomalies.
    """

    if df.empty:
        logging.warning("No data available for anomaly detection.")
        return pd.DataFrame()

    required_columns = [
        "EventName",
        "WindowStartUtc"
    ] + METRIC_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if min_periods < 2:
        raise ValueError("min_periods must be at least 2.")

    if window < min_periods:
        raise ValueError(
            "window must be greater than or equal to min_periods."
        )

    data = df.copy()

    data["WindowStartUtc"] = pd.to_datetime(
        data["WindowStartUtc"],
        utc=True,
        errors="coerce"
    )

    data = data.dropna(
        subset=["EventName", "WindowStartUtc"]
    )

    data = data.sort_values(
        ["EventName", "WindowStartUtc"]
    ).reset_index(drop=True)

    anomalies = []

    # Analyze each event type independently
    for event_name, group in data.groupby("EventName"):

        group = group.copy().reset_index(drop=True)

        for metric in METRIC_COLUMNS:

            values = pd.to_numeric(
                group[metric],
                errors="coerce"
            )

            # Historical baseline excludes current observation
            historical_values = values.shift(1)

            rolling_mean = historical_values.rolling(
                window=window,
                min_periods=min_periods
            ).mean()

            rolling_std = historical_values.rolling(
                window=window,
                min_periods=min_periods
            ).std(ddof=1)

            for index in range(len(group)):

                current_value = values.iloc[index]
                baseline_mean = rolling_mean.iloc[index]
                baseline_std = rolling_std.iloc[index]

                # Skip missing values or insufficient baseline
                if pd.isna(current_value):
                    continue

                if pd.isna(baseline_mean):
                    continue

                if pd.isna(baseline_std):
                    continue

                # Handle a zero-variance historical baseline
                if baseline_std == 0:

                    if current_value == baseline_mean:
                        continue

                    z_score = (
                        float("inf")
                        if current_value > baseline_mean
                        else float("-inf")
                    )

                else:

                    z_score = (
                        current_value - baseline_mean
                    ) / baseline_std

                absolute_z_score = abs(z_score)

                if absolute_z_score < WARNING_THRESHOLD:
                    continue

                if absolute_z_score >= CRITICAL_THRESHOLD:
                    severity = "CRITICAL"
                else:
                    severity = "WARNING"

                anomalies.append({
                    "Timestamp": group.loc[
                        index, "WindowStartUtc"
                    ],
                    "EventName": event_name,
                    "Metric": metric,
                    "ActualValue": current_value,
                    "BaselineMean": baseline_mean,
                    "BaselineStd": baseline_std,
                    "ZScore": z_score,
                    "Severity": severity
                })

    return pd.DataFrame(anomalies)


# --------------------------------------------------
# 2. Display anomaly results
# --------------------------------------------------

def display_anomalies(anomalies):

    print("\n" + "=" * 75)
    print("       PiAssistIT - Statistical Anomaly Detection")
    print("=" * 75)

    if anomalies.empty:

        print("\nNo anomalies detected.")

        print(
            "\nThis does not necessarily mean all metrics are normal."
        )

        print(
            "The detector may not yet have enough historical data."
        )

        return

    anomalies = anomalies.sort_values(
        "ZScore",
        key=lambda column: column.abs(),
        ascending=False
    )

    print(f"\nTotal anomalies detected: {len(anomalies)}")

    print("\nAnomaly details:\n")

    print(
        anomalies.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        )
    )

    print("\nSeverity summary:")

    print(
        anomalies["Severity"].value_counts().to_string()
    )

    print("\n" + "=" * 75)


# --------------------------------------------------
# 3. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    try:

        print("\nLoading historical telemetry...")

        metrics_df = load_metrics(days=30)

        if metrics_df.empty:

            print("No telemetry data available.")
        
        else:

            print(
                f"Loaded {len(metrics_df)} telemetry records."
            )

            print("\nRunning Rolling Z-score detector...")

            anomalies_df = detect_anomalies(metrics_df)

            display_anomalies(anomalies_df)

            print(
                "\nStatistical anomaly detection completed."
            )

    except Exception:

        logging.exception(
            "Statistical anomaly detection failed."
        )

        raise
