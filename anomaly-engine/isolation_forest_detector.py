
import logging

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


from metric_loader import load_metrics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MIN_RECORDS = 25
MIN_TRAIN_RECORDS = 20

CONTAMINATION = 0.05
N_ESTIMATORS = 200
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "EventCount",
    "AvgDurationMs",
    "MaxDurationMs",
    "AvgHallucinationScore",
    "UnsupportedClaims",
    "AvgRetrievedDocuments",
    "TotalTokens"
]


# --------------------------------------------------
# 1. Isolation Forest detector
# --------------------------------------------------

def detect_anomalies(df):
    """
    Detect unusual multivariate patterns using Isolation Forest.

    Each event type is analyzed independently because HTTP,
    LLM, retrieval and other telemetry have different distributions.

    The earliest observations train the model. The latest observations
    are evaluated against that historical baseline.
    """

    if df.empty:
        logging.warning("No telemetry data available.")
        return pd.DataFrame()

    required_columns = [
        "EventName",
        "WindowStartUtc"
    ] + FEATURE_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
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

    all_anomalies = []

    # Analyze each event type independently
    for event_name, group in data.groupby("EventName"):

        group = group.sort_values(
            "WindowStartUtc"
        ).reset_index(drop=True)

        if len(group) < MIN_RECORDS:

            logging.info(
                "Skipping %s: only %s records available. "
                "At least %s required.",
                event_name,
                len(group),
                MIN_RECORDS
            )
            continue

        # Identify usable numeric features
        numeric_data = group[FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        usable_features = []

        for column in FEATURE_COLUMNS:

            non_null_ratio = numeric_data[column].notna().mean()

            if non_null_ratio < 0.80:
                continue

            if numeric_data[column].nunique(dropna=True) < 2:
                continue

            usable_features.append(column)

        if len(usable_features) < 2:

            logging.info(
                "Skipping %s: fewer than two usable features.",
                event_name
            )
            continue

        features = numeric_data[usable_features]

        # Split chronologically: historical training vs recent testing
        split_index = int(len(group) * 0.80)

        train_data = features.iloc[:split_index].copy()
        test_data = features.iloc[split_index:].copy()

        if len(train_data) < MIN_TRAIN_RECORDS:

            logging.info(
                "Skipping %s: insufficient training records.",
                event_name
            )
            continue

        # Remove features that are entirely missing in training
        valid_features = [
            column for column in usable_features
            if train_data[column].notna().any()
        ]

        if len(valid_features) < 2:
            continue

        train_data = train_data[valid_features]
        test_data = test_data[valid_features]

        # Impute missing values using training medians only
        imputer = SimpleImputer(strategy="median")

        train_array = imputer.fit_transform(train_data)
        test_array = imputer.transform(test_data)

        # Train Isolation Forest on historical observations
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(train_array)

        # Evaluate recent observations
        predictions = model.predict(test_array)
        scores = model.decision_function(test_array)

        for position, prediction in enumerate(predictions):

            if prediction != -1:
                continue

            original_index = split_index + position

            all_anomalies.append({
                "Timestamp": group.loc[
                    original_index, "WindowStartUtc"
                ],
                "EventName": event_name,
                "DetectionMethod": "Isolation Forest",
                "AnomalyScore": float(scores[position]),
                "Severity": "WARNING",
                "FeaturesUsed": ", ".join(valid_features)
            })

    return pd.DataFrame(all_anomalies)


# --------------------------------------------------
# 2. Display results
# --------------------------------------------------

def display_anomalies(anomalies):

    print("\n" + "=" * 75)
    print("       PiAssistIT - Isolation Forest Detection")
    print("=" * 75)

    if anomalies.empty:

        print("\nNo anomalies detected.")

        print(
            "\nThis may mean there are no unusual observations "
            "or insufficient historical data."
        )

        return

    print(
        f"\nTotal anomalies detected: {len(anomalies)}"
    )

    print("\nAnomaly details:\n")

    print(
        anomalies.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
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

        print(
            f"\nLoaded {len(metrics_df)} records."
        )

        print("\nRunning Isolation Forest...")

        anomalies_df = detect_anomalies(metrics_df)

        display_anomalies(anomalies_df)

        print(
            "\nIsolation Forest execution completed."
        )

    except Exception:

        logging.exception(
            "Isolation Forest detection failed."
        )

        raise
