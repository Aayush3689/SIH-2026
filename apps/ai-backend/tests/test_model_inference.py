from __future__ import annotations

import json

import pandas as pd

from app.features.telemetry import (
    normalize_telemetry,
    validate_sample_count,
)

from app.features.runtime_features import (
    build_temporal_features,
)

from app.features.physics import (
    calculate_physics_features,
)

from app.features.fault_features import (
    add_fault_v4_features,
)

from app.features.rul_features import (
    RULFeatureBuilder,
)

from app.anomaly.model import (
    AnomalyModel,
)

from app.fault.model import (
    FaultModel,
)

from app.rul.model import (
    RULModel,
)


# =============================================================================
# CONFIG
# =============================================================================

DATASET = (
    "training/datasets/raw/"
    "dt_engine_telemetry_full.csv"
)


ANOMALY_MODEL = (
    "models/anomaly/"
    "final_anomaly_model.joblib"
)


FAULT_MODEL = (
    "models/fault_classifier_v4/"
    "final_fault_classifier_v4.joblib"
)


RUL_MODEL = (
    "models/rul_v4/"
    "final_rul_v4.joblib"
)


# =============================================================================
# HELPERS
# =============================================================================

def print_section(
    title: str,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print_section(
        "AI BACKEND MODEL INFERENCE TEST"
    )

    # =========================================================================
    # 1. LOAD TELEMETRY
    # =========================================================================

    print_section(
        "[1] Loading telemetry"
    )

    raw = pd.read_csv(
        DATASET
    )

    print(
        f"Rows    : {len(raw):,}"
    )

    print(
        f"Columns : {len(raw.columns)}"
    )

    # =========================================================================
    # 2. NORMALIZE
    # =========================================================================

    print_section(
        "[2] Normalizing telemetry"
    )

    df = normalize_telemetry(
        raw
    )

    print(
        f"Normalized rows: {len(df):,}"
    )

    # -------------------------------------------------------------------------
    # Runtime-required columns
    # -------------------------------------------------------------------------

    required_runtime = [
        "engine_id",
        "flight_id",
        "elapsed_s",
    ]

    missing = [
        column
        for column in required_runtime
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing runtime columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(
        "Runtime-required columns: OK"
    )

    # =========================================================================
    # 3. TEMPORAL FEATURES
    # =========================================================================

    print_section(
        "[3] Building temporal features"
    )

    df = build_temporal_features(
        df
    )

    print(
        "Temporal features: OK"
    )

    # =========================================================================
    # 4. PHYSICS FEATURES
    # =========================================================================

    print_section(
        "[4] Building physics features"
    )

    df = calculate_physics_features(
        df
    )

    print(
        "Physics features: OK"
    )

    # =========================================================================
    # 5. FAULT V4 FEATURES
    # =========================================================================

    print_section(
        "[5] Building Fault V4 features"
    )

    df = add_fault_v4_features(
        df
    )

    print(
        "Fault V4 features: OK"
    )

    # =========================================================================
    # 6. CURRENT ENGINE / FLIGHT
    # =========================================================================

    print_section(
        "[6] Selecting latest engine/flight"
    )

    latest_engine = (
        df["engine_id"].iloc[-1]
    )

    latest_flight = (
        df["flight_id"].iloc[-1]
    )

    current_flight = df[
        (df["engine_id"] == latest_engine)
        & (df["flight_id"] == latest_flight)
    ].copy()

    print(
        f"Engine : {latest_engine}"
    )

    print(
        f"Flight : {latest_flight}"
    )

    print(
        f"Samples: {len(current_flight)}"
    )

# =========================================================================
# 7. LOAD ANOMALY MODEL
# =========================================================================

    print_section(
        "[7] Loading anomaly model"
    )

    anomaly_model = AnomalyModel(
        ANOMALY_MODEL
    )

    anomaly_model.load()

    print(
        "Anomaly model: LOADED"
    )

    print(
        f"Features: {len(anomaly_model.feature_names)}"
    )

    # =========================================================================
    # 8. ANOMALY INFERENCE
    # =========================================================================

    print_section(
        "[8] Running anomaly inference"
    )

    latest = df.tail(
        1
    ).copy()

    anomaly_result = (
        anomaly_model.predict(
            latest
        )
    )

    print(
        json.dumps(
            anomaly_result,
            indent=2,
            default=str,
        )
    )

    # =========================================================================
    # 9. LOAD FAULT MODEL
    # =========================================================================

    print_section(
        "[9] Loading Fault V4 model"
    )

    fault_model = FaultModel(
        FAULT_MODEL
    )

    fault_model.load()

    print(
        "Fault V4 model: LOADED"
    )

    print(
        f"Features: {len(fault_model.feature_names)}"
    )

    print(
        f"Classes : {len(fault_model.class_names)}"
    )

    # =========================================================================
    # 10. FAULT INFERENCE
    # =========================================================================

    print_section(
        "[10] Running Fault V4 inference"
    )

    fault_result = (
        fault_model.predict(
            latest
        )
    )

    print(
        json.dumps(
            fault_result,
            indent=2,
            default=str,
        )
    )

    # =========================================================================
    # 11. LOAD RUL MODEL
    # =========================================================================

    print_section(
        "[11] Loading RUL V4 model"
    )

    rul_model = RULModel(
        RUL_MODEL
    )

    rul_model.load()

    print(
        "RUL V4 model: LOADED"
    )

    print(
        f"Window samples: "
        f"{rul_model.window_samples}"
    )

    print(
        f"Sensor features: "
        f"{len(rul_model.sensor_features)}"
    )

    print(
        f"Mission numeric features: "
        f"{len(rul_model.mission_numeric_features)}"
    )

    print(
        f"Mission categorical features: "
        f"{len(rul_model.mission_categorical_features)}"
    )

    # =========================================================================
    # 12. RUL WINDOW
    # =========================================================================

    print_section(
        "[12] Building RUL V4 input"
    )

    validate_sample_count(
        current_flight,
        minimum_samples=8,
    )

    rul_builder = RULFeatureBuilder(
        window_samples=8
    )

    rul_features = (
        rul_builder.build_from_model(
            df,
            rul_model,
        )
    )

    print(
        "RUL feature row: OK"
    )

    print(
        f"RUL input columns: "
        f"{len(rul_features.columns)}"
    )

    print()
    print(
        rul_features.to_string(
            index=False
        )
    )

    # =========================================================================
    # 13. RUL INFERENCE
    # =========================================================================

    print_section(
        "[13] Running RUL V4 inference"
    )

    rul_result = (
        rul_model.predict(
            rul_features
        )
    )

    print(
        json.dumps(
            rul_result,
            indent=2,
            default=str,
        )
    )

    # =========================================================================
    # 14. COMBINED RESULT
    # =========================================================================

    print_section(
        "[14] COMBINED AI RESULT"
    )

    latest_row = latest.iloc[0]

    timestamp = latest_row[
        "timestamp"
    ]

    if pd.notna(timestamp):
        timestamp = timestamp.isoformat()

    combined = {
        "engine_id": latest_row[
            "engine_id"
        ],

        "flight_id": latest_row[
            "flight_id"
        ],

        "timestamp": timestamp,

        "anomaly": anomaly_result,

        "fault": fault_result,

        "rul": rul_result,
    }

    print(
        json.dumps(
            combined,
            indent=2,
            default=str,
        )
    )

    # =========================================================================
    # 15. FINAL STATUS
    # =========================================================================

    print_section(
        "MODEL INFERENCE TEST PASSED"
    )

    print(
        "Telemetry            : OK"
    )

    print(
        "Temporal features    : OK"
    )

    print(
        "Physics features     : OK"
    )

    print(
        "Fault V4 features    : OK"
    )

    print(
        "Anomaly model        : OK"
    )

    print(
        "Fault V4 model       : OK"
    )

    print(
        "RUL V4 model         : OK"
    )

    print(
        "Combined inference   : OK"
    )


if __name__ == "__main__":
    main()