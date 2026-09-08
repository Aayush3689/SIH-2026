"""
FINAL ANOMALY MODEL TRAINING
============================

Label policy:
    RAW-FAULT-AWARE

    raw_fault_active == 1
        -> ACTIVE_FAULT

    4 flights immediately before the FIRST active-fault flight
        -> PRE_FAULT

    everything else
        -> NORMAL


Features:
    CORE_ONLY = 40 engine-condition / temporal / physics features

Split:
    TRAIN:
        ENG-2023000 ... ENG-2023005
        ENG-2023008

    VALIDATION:
        ENG-2023006

    TEST:
        ENG-2023007
        ENG-2023009

IMPORTANT:
    - Engine-wise split avoids engine leakage.
    - Test engines are never used for model selection.
    - Two baseline models are compared.
    - Best model is selected ONLY using validation Macro F1.
    - Selected model is retrained on Train + Validation.
    - Final test is performed once on unseen test engines.

Outputs:
    models/anomaly/final_anomaly_model.joblib
    models/anomaly/final_anomaly_metadata.json
    training/datasets/processed/anomaly/final_anomaly_model_comparison.csv
    training/datasets/processed/anomaly/final_anomaly_test_predictions.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =====================================================================
# PROJECT PATHS
# =====================================================================

# File:
# ai-backend/training/anomaly/train_final_anomaly_model.py
#
# parents[0] -> training/anomaly
# parents[1] -> training
# parents[2] -> ai-backend
#
TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent

RAW_PATH = (
    TRAINING_DIR
    / "datasets"
    / "raw"
    / "dt_engine_telemetry_full.csv"
)

FEATURE_PATH = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "anomaly_features.csv"
)

PROCESSED_ANOMALY_DIR = (
    TRAINING_DIR
    / "datasets"
    / "processed"
    / "anomaly"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
)

MODEL_PATH = (
    MODEL_DIR
    / "final_anomaly_model.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "final_anomaly_metadata.json"
)

COMPARISON_PATH = (
    PROCESSED_ANOMALY_DIR
    / "final_anomaly_model_comparison.csv"
)

TEST_PREDICTIONS_PATH = (
    PROCESSED_ANOMALY_DIR
    / "final_anomaly_test_predictions.csv"
)


# =====================================================================
# EXPERIMENT CONFIG
# =====================================================================

RANDOM_STATE = 42

PREFAULT_WINDOW = 4

SELECT_METRIC = "macro_f1"


# Same engine-wise split we have been using.
TRAIN_ENGINES = [
    "ENG-2023000",
    "ENG-2023001",
    "ENG-2023002",
    "ENG-2023003",
    "ENG-2023004",
    "ENG-2023005",
    "ENG-2023008",
]

VAL_ENGINES = [
    "ENG-2023006",
]

TEST_ENGINES = [
    "ENG-2023007",
    "ENG-2023009",
]


# =====================================================================
# LABEL DEFINITIONS
# =====================================================================

NORMAL = 0
PRE_FAULT = 1
ACTIVE_FAULT = 2

STATE_NAMES = {
    NORMAL: "NORMAL",
    PRE_FAULT: "PRE_FAULT",
    ACTIVE_FAULT: "ACTIVE_FAULT",
}


# =====================================================================
# CORE ONLY FEATURES
# =====================================================================

CORE_FEATURES = [
    # -------------------------------------------------------------
    # Raw engine condition
    # -------------------------------------------------------------

    "rpm",
    "throttle_position_pct",
    "cht_c",
    "egt_c",
    "oil_pressure_kpa",
    "oil_temperature_c",
    "fuel_flow_lph",
    "fuel_pressure_kpa",
    "intake_manifold_pressure_kpa",
    "engine_vibration_mm_s",
    "battery_voltage_v",
    "alternator_output_a",
    "injection_timing_deg_btdc",
    "engine_load_pct",

    # -------------------------------------------------------------
    # Rolling / temporal
    # -------------------------------------------------------------

    "vibration_mean_30s",
    "vibration_std_30s",
    "rpm_mean_30s",
    "rpm_std_30s",

    "egt_mean_60s",
    "egt_std_60s",

    "cht_mean_60s",
    "cht_std_60s",

    "oil_pressure_mean_60s",
    "oil_pressure_std_60s",

    "oil_temperature_mean_60s",
    "oil_temperature_std_60s",

    # -------------------------------------------------------------
    # Deltas
    # -------------------------------------------------------------

    "egt_delta",
    "cht_delta",
    "oil_pressure_delta",
    "vibration_delta",

    # -------------------------------------------------------------
    # Rates
    # -------------------------------------------------------------

    "egt_rate",
    "cht_rate",
    "oil_pressure_rate",
    "vibration_rate",

    # -------------------------------------------------------------
    # Physics-informed
    # -------------------------------------------------------------

    "egt_expected",
    "egt_residual",

    "cht_expected",
    "cht_residual",

    "oil_temperature_expected",
    "oil_temperature_residual",
]


# =====================================================================
# HELPERS
# =====================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def check_paths() -> None:
    if not RAW_PATH.exists():
        fail(
            "Raw telemetry file not found:\n"
            f"{RAW_PATH}"
        )

    if not FEATURE_PATH.exists():
        fail(
            "Feature file not found:\n"
            f"{FEATURE_PATH}"
        )

    PROCESSED_ANOMALY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# LOAD DATA
# =====================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading raw telemetry...")
    raw = pd.read_csv(RAW_PATH)

    print("Loading anomaly features...")
    features = pd.read_csv(FEATURE_PATH)

    return raw, features


# =====================================================================
# VALIDATION
# =====================================================================

def validate_data(
    raw: pd.DataFrame,
    features: pd.DataFrame,
) -> None:

    raw_required = {
        "engine_id",
        "flight_id",
        "timestamp",
        "fault_active_label",
    }

    missing_raw = (
        raw_required
        - set(raw.columns)
    )

    if missing_raw:
        fail(
            f"Raw data missing columns: "
            f"{sorted(missing_raw)}"
        )

    feature_required = {
        "engine_id",
        "flight_id",
    }

    missing_feature_keys = (
        feature_required
        - set(features.columns)
    )

    if missing_feature_keys:
        fail(
            f"Feature data missing columns: "
            f"{sorted(missing_feature_keys)}"
        )

    missing_features = [
        col
        for col in CORE_FEATURES
        if col not in features.columns
    ]

    if missing_features:
        fail(
            "Missing CORE_ONLY features:\n"
            f"{missing_features}"
        )

    if len(raw) != len(features):
        fail(
            "Raw and feature row counts do not match.\n"
            f"Raw: {len(raw):,}\n"
            f"Features: {len(features):,}"
        )

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"],
        errors="coerce",
    )

    if raw["timestamp"].isna().any():
        fail(
            "Raw dataset contains invalid timestamps."
        )


# =====================================================================
# BUILD FLIGHT TABLE
# =====================================================================

def build_flight_table(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one record per engine + flight.

    raw_fault_active = 1 if ANY row in that flight
    has fault_active_label == 1.
    """

    work = raw.copy()

    work["fault_active_label"] = (
        pd.to_numeric(
            work["fault_active_label"],
            errors="coerce",
        )
        .astype(int)
    )

    flights = (
        work
        .groupby(
            ["engine_id", "flight_id"],
            sort=False,
        )
        .agg(
            first_timestamp=(
                "timestamp",
                "min",
            ),
            last_timestamp=(
                "timestamp",
                "max",
            ),
            raw_fault_active=(
                "fault_active_label",
                "max",
            ),
        )
        .reset_index()
    )

    flights = (
        flights
        .sort_values(
            [
                "engine_id",
                "first_timestamp",
                "flight_id",
            ]
        )
        .reset_index(drop=True)
    )

    return flights


# =====================================================================
# BUILD FINAL RAW-AWARE LABEL
# =====================================================================

def build_labels(
    flights: pd.DataFrame,
) -> pd.DataFrame:
    """
    RAW-FAULT-AWARE + N=4 policy.

    NORMAL:
        default

    PRE_FAULT:
        4 flights immediately before first raw active fault

    ACTIVE_FAULT:
        raw_fault_active == 1

    IMPORTANT:
        We do not label every flight after first fault as active.
    """

    result = flights.copy()

    result["anomaly_state"] = NORMAL

    for engine_id, group in result.groupby(
        "engine_id",
        sort=False,
    ):

        group = (
            group
            .sort_values(
                [
                    "first_timestamp",
                    "flight_id",
                ]
            )
        )

        indices = group.index.to_list()

        raw_active = (
            group[
                "raw_fault_active"
            ]
            .to_numpy()
        )

        active_positions = np.flatnonzero(
            raw_active == 1
        )

        # Healthy engine
        if len(active_positions) == 0:
            continue

        # First active fault
        first_active_position = int(
            active_positions[0]
        )

        # ---------------------------------------------------------
        # PRE_FAULT
        # ---------------------------------------------------------

        pre_start = max(
            0,
            first_active_position
            - PREFAULT_WINDOW,
        )

        for position in range(
            pre_start,
            first_active_position,
        ):

            row_index = indices[position]

            result.loc[
                row_index,
                "anomaly_state",
            ] = PRE_FAULT

        # ---------------------------------------------------------
        # ACTIVE_FAULT
        # ---------------------------------------------------------

        for position in active_positions:

            row_index = indices[position]

            result.loc[
                row_index,
                "anomaly_state",
            ] = ACTIVE_FAULT

    return result


# =====================================================================
# ATTACH LABEL TO FEATURES
# =====================================================================

def attach_labels(
    features: pd.DataFrame,
    labeled_flights: pd.DataFrame,
) -> pd.DataFrame:

    keys = [
        "engine_id",
        "flight_id",
    ]

    labels = (
        labeled_flights[
            keys + ["anomaly_state"]
        ]
        .drop_duplicates()
    )

    merged = features.merge(
        labels,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    if merged["anomaly_state"].isna().any():

        count = int(
            merged["anomaly_state"]
            .isna()
            .sum()
        )

        fail(
            f"{count} feature rows have no anomaly label."
        )

    merged["anomaly_state"] = (
        merged["anomaly_state"]
        .astype(int)
    )

    return merged


# =====================================================================
# PREPARE X
# =====================================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    X = df[
        CORE_FEATURES
    ].copy()

    for column in CORE_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X


# =====================================================================
# FIT-SAFE IMPUTATION
# =====================================================================

def fit_imputation(
    X_train: pd.DataFrame,
) -> pd.Series:

    medians = X_train.median()

    # If a feature is completely missing in training,
    # use 0 as a final fallback.
    medians = medians.fillna(0.0)

    return medians


def apply_imputation(
    X: pd.DataFrame,
    medians: pd.Series,
) -> pd.DataFrame:

    result = X.copy()

    result = result.fillna(
        medians
    )

    return result


# =====================================================================
# MODEL BUILDERS
# =====================================================================

def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_hist_gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


# =====================================================================
# METRICS
# =====================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "normal_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[NORMAL],
                average="macro",
                zero_division=0,
            )
        ),

        "normal_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[NORMAL],
                average="macro",
                zero_division=0,
            )
        ),

        "prefault_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[PRE_FAULT],
                average="macro",
                zero_division=0,
            )
        ),

        "prefault_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[PRE_FAULT],
                average="macro",
                zero_division=0,
            )
        ),

        "active_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[ACTIVE_FAULT],
                average="macro",
                zero_division=0,
            )
        ),

        "active_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[ACTIVE_FAULT],
                average="macro",
                zero_division=0,
            )
        ),
    }


# =====================================================================
# EVALUATE ONE MODEL
# =====================================================================

def evaluate_model(
    model_name: str,
    model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[dict, object]:

    X_train = prepare_features(
        train_df
    )

    y_train = (
        train_df[
            "anomaly_state"
        ]
        .astype(int)
    )

    X_val = prepare_features(
        val_df
    )

    y_val = (
        val_df[
            "anomaly_state"
        ]
        .astype(int)
    )

    # -------------------------------------------------------------
    # Imputation is fitted ONLY on training data.
    # -------------------------------------------------------------

    medians = fit_imputation(
        X_train
    )

    X_train = apply_imputation(
        X_train,
        medians,
    )

    X_val = apply_imputation(
        X_val,
        medians,
    )

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # -------------------------------------------------------------
    # Predict validation
    # -------------------------------------------------------------

    y_pred = model.predict(
        X_val
    )

    metrics = calculate_metrics(
        y_val,
        y_pred,
    )

    metrics[
        "model"
    ] = model_name

    metrics[
        "validation_rows"
    ] = len(val_df)

    return metrics, model


# =====================================================================
# PRINT DISTRIBUTION
# =====================================================================

def print_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:

    counts = (
        df["anomaly_state"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    print()
    print(
        f"{name} distribution:"
    )

    for state in [
        NORMAL,
        PRE_FAULT,
        ACTIVE_FAULT,
    ]:

        count = int(
            counts.get(
                state,
                0,
            )
        )

        percentage = (
            count
            / total
            * 100
            if total
            else 0
        )

        print(
            f"  {STATE_NAMES[state]:12s}: "
            f"{count:8,d} "
            f"({percentage:6.2f}%)"
        )


# =====================================================================
# TRAIN FINAL MODEL
# =====================================================================

def train_final_model(
    model_name: str,
    train_val_df: pd.DataFrame,
):
    """
    Refit the selected model using ALL development engines:

        TRAIN + VALIDATION

    Test engines remain untouched.
    """

    X = prepare_features(
        train_val_df
    )

    y = (
        train_val_df[
            "anomaly_state"
        ]
        .astype(int)
    )

    medians = fit_imputation(
        X
    )

    X = apply_imputation(
        X,
        medians,
    )

    if model_name == "RandomForest":
        model = build_random_forest()

    elif model_name == "HistGradientBoosting":
        model = build_hist_gradient_boosting()

    else:
        fail(
            f"Unknown model: {model_name}"
        )

    model.fit(
        X,
        y,
    )

    return model, medians


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def get_feature_importance(
    model,
) -> dict:

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = (
            model.feature_importances_
        )

        pairs = list(
            zip(
                CORE_FEATURES,
                importance,
            )
        )

        pairs.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            feature: float(score)
            for feature, score
            in pairs[:20]
        }

    # HistGradientBoosting does not expose
    # feature_importances_ in the same way.
    return {}


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 78)
    print(
        "FINAL ANOMALY MODEL TRAINING"
    )
    print("=" * 78)

    # -------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------

    check_paths()

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    raw, features = load_data()

    validate_data(
        raw,
        features,
    )

    # -------------------------------------------------------------
    # Flight-level labels
    # -------------------------------------------------------------

    flights = build_flight_table(
        raw
    )

    labeled_flights = build_labels(
        flights
    )

    # -------------------------------------------------------------
    # Attach labels
    # -------------------------------------------------------------

    data = attach_labels(
        features,
        labeled_flights,
    )

    # -------------------------------------------------------------
    # Engine check
    # -------------------------------------------------------------

    available_engines = sorted(
        data["engine_id"]
        .dropna()
        .unique()
        .tolist()
    )

    expected_engines = sorted(
        set(
            TRAIN_ENGINES
            + VAL_ENGINES
            + TEST_ENGINES
        )
    )

    if available_engines != expected_engines:

        fail(
            "Engine set mismatch.\n"
            f"Available: {available_engines}\n"
            f"Expected : {expected_engines}"
        )

    # -------------------------------------------------------------
    # Create split
    # -------------------------------------------------------------

    train_df = data[
        data["engine_id"]
        .isin(TRAIN_ENGINES)
    ].copy()

    val_df = data[
        data["engine_id"]
        .isin(VAL_ENGINES)
    ].copy()

    test_df = data[
        data["engine_id"]
        .isin(TEST_ENGINES)
    ].copy()

    # -------------------------------------------------------------
    # Check leakage
    # -------------------------------------------------------------

    train_set = set(
        train_df["engine_id"]
    )

    val_set = set(
        val_df["engine_id"]
    )

    test_set = set(
        test_df["engine_id"]
    )

    if train_set & val_set:
        fail("Engine leakage between train and validation.")

    if train_set & test_set:
        fail("Engine leakage between train and test.")

    if val_set & test_set:
        fail("Engine leakage between validation and test.")

    # -------------------------------------------------------------
    # Print overview
    # -------------------------------------------------------------

    print()
    print(
        f"Total rows : {len(data):,}"
    )

    print(
        f"CORE features: {len(CORE_FEATURES)}"
    )

    print(
        f"Train engines: {TRAIN_ENGINES}"
    )

    print(
        f"Val engines  : {VAL_ENGINES}"
    )

    print(
        f"Test engines : {TEST_ENGINES}"
    )

    print_distribution(
        "TRAIN",
        train_df,
    )

    print_distribution(
        "VALIDATION",
        val_df,
    )

    print_distribution(
        "TEST",
        test_df,
    )

    # =================================================================
    # MODEL BENCHMARK
    # =================================================================

    models = {
        "RandomForest":
            build_random_forest(),

        "HistGradientBoosting":
            build_hist_gradient_boosting(),
    }

    validation_results = []

    trained_validation_models = {}

    print()
    print("=" * 78)
    print(
        "VALIDATION MODEL COMPARISON"
    )
    print("=" * 78)

    for model_name, model in models.items():

        print()
        print(
            f"Training {model_name}..."
        )

        metrics, trained_model = evaluate_model(
            model_name,
            model,
            train_df,
            val_df,
        )

        validation_results.append(
            metrics
        )

        trained_validation_models[
            model_name
        ] = trained_model

        print(
            f"  Accuracy      : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro F1      : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"  Macro Recall  : "
            f"{metrics['macro_recall']:.4f}"
        )

        print(
            f"  PRE_FAULT F1  : "
            f"{metrics['prefault_f1']:.4f}"
        )

        print(
            f"  PRE_FAULT Rec : "
            f"{metrics['prefault_recall']:.4f}"
        )

        print(
            f"  ACTIVE F1     : "
            f"{metrics['active_f1']:.4f}"
        )

        print(
            f"  ACTIVE Rec    : "
            f"{metrics['active_recall']:.4f}"
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df = (
        validation_df
        .sort_values(
            "macro_f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Select winner ONLY from validation
    # -------------------------------------------------------------

    best_model_name = (
        validation_df.iloc[0]["model"]
    )

    best_validation_macro_f1 = float(
        validation_df.iloc[0][
            "macro_f1"
        ]
    )

    print()
    print(
        "=" * 78
    )

    print(
        f"BEST MODEL ON VALIDATION: "
        f"{best_model_name}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_validation_macro_f1:.4f}"
    )

    print(
        "=" * 78
    )

    # =================================================================
    # FINAL REFIT
    # =================================================================

    print()
    print(
        "Refitting selected model on "
        "TRAIN + VALIDATION..."
    )

    train_val_df = pd.concat(
        [
            train_df,
            val_df,
        ],
        ignore_index=True,
    )

    final_model, final_medians = (
        train_final_model(
            best_model_name,
            train_val_df,
        )
    )

    # =================================================================
    # FINAL TEST
    # =================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL UNSEEN-ENGINE TEST"
    )

    print(
        "=" * 78
    )

    X_test = prepare_features(
        test_df
    )

    y_test = (
        test_df[
            "anomaly_state"
        ]
        .astype(int)
    )

    X_test = apply_imputation(
        X_test,
        final_medians,
    )

    y_test_pred = (
        final_model.predict(
            X_test
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
    )

    # -------------------------------------------------------------
    # Test metrics
    # -------------------------------------------------------------

    print()
    print(
        f"Accuracy      : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1      : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall  : "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print()
    print(
        "NORMAL"
    )

    print(
        f"  F1     : "
        f"{test_metrics['normal_f1']:.4f}"
    )

    print(
        f"  Recall : "
        f"{test_metrics['normal_recall']:.4f}"
    )

    print()
    print(
        "PRE_FAULT"
    )

    print(
        f"  F1     : "
        f"{test_metrics['prefault_f1']:.4f}"
    )

    print(
        f"  Recall : "
        f"{test_metrics['prefault_recall']:.4f}"
    )

    print()
    print(
        "ACTIVE_FAULT"
    )

    print(
        f"  F1     : "
        f"{test_metrics['active_f1']:.4f}"
    )

    print(
        f"  Recall : "
        f"{test_metrics['active_recall']:.4f}"
    )

    # =================================================================
    # CLASSIFICATION REPORT
    # =================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=" * 78
    )

    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=[
                NORMAL,
                PRE_FAULT,
                ACTIVE_FAULT,
            ],
            target_names=[
                "NORMAL",
                "PRE_FAULT",
                "ACTIVE_FAULT",
            ],
            zero_division=0,
        )
    )

    # =================================================================
    # CONFUSION MATRIX
    # =================================================================

    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=[
            NORMAL,
            PRE_FAULT,
            ACTIVE_FAULT,
        ],
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "Rows = actual"
    )

    print(
        "Cols = predicted"
    )

    print(
        cm
    )

    # =================================================================
    # PER-ENGINE TEST RESULTS
    # =================================================================

    per_engine_rows = []

    test_with_predictions = (
        test_df[
            [
                "engine_id",
                "flight_id",
                "timestamp",
                "anomaly_state",
            ]
        ]
        .copy()
    )

    test_with_predictions[
        "predicted_state"
    ] = y_test_pred

    # -------------------------------------------------------------
    # Save row-level test predictions
    # -------------------------------------------------------------

    prediction_output = (
        test_with_predictions
        .copy()
    )

    prediction_output[
        "actual_state_name"
    ] = prediction_output[
        "anomaly_state"
    ].map(
        STATE_NAMES
    )

    prediction_output[
        "predicted_state_name"
    ] = prediction_output[
        "predicted_state"
    ].map(
        STATE_NAMES
    )

    prediction_output.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Per-engine metrics
    # -------------------------------------------------------------

    for engine_id, group in (
        test_with_predictions
        .groupby("engine_id")
    ):

        engine_metrics = calculate_metrics(
            group["anomaly_state"],
            group["predicted_state"],
        )

        per_engine_rows.append(
            {
                "engine_id":
                    engine_id,

                "rows":
                    len(group),

                **engine_metrics,
            }
        )

    per_engine_df = pd.DataFrame(
        per_engine_rows
    )

    print()
    print(
        "PER-ENGINE TEST RESULTS"
    )

    print(
        per_engine_df[
            [
                "engine_id",
                "rows",
                "accuracy",
                "macro_f1",
                "prefault_recall",
                "active_recall",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # =================================================================
    # FEATURE IMPORTANCE
    # =================================================================

    importance = get_feature_importance(
        final_model
    )

    if importance:

        print()
        print(
            "TOP 20 FEATURE IMPORTANCE"
        )

        for rank, (
            feature,
            score,
        ) in enumerate(
            importance.items(),
            start=1,
        ):

            print(
                f"{rank:2d}. "
                f"{feature:35s} "
                f"{score:.6f}"
            )

    # =================================================================
    # SAVE MODEL
    # =================================================================

    model_bundle = {
        "model": final_model,
        "feature_names": CORE_FEATURES,
        "imputation_medians": final_medians.to_dict(),
        "label_mapping": STATE_NAMES,
        "label_policy": (
            "RAW_FAULT_AWARE"
        ),
        "prefault_window_flights":
            PREFAULT_WINDOW,
        "random_state":
            RANDOM_STATE,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    # =================================================================
    # SAVE METADATA
    # =================================================================

    metadata = {
        "model_name":
            best_model_name,

        "task":
            "3-state anomaly classification",

        "label_policy":
            "RAW_FAULT_AWARE",

        "prefault_window_flights":
            PREFAULT_WINDOW,

        "classes":
            STATE_NAMES,

        "feature_set":
            "CORE_ONLY",

        "feature_count":
            len(CORE_FEATURES),

        "features":
            CORE_FEATURES,

        "train_engines":
            TRAIN_ENGINES,

        "validation_engines":
            VAL_ENGINES,

        "test_engines":
            TEST_ENGINES,

        "validation_metrics":
            validation_df.to_dict(
                orient="records"
            ),

        "selected_validation_macro_f1":
            best_validation_macro_f1,

        "final_test_metrics":
            test_metrics,

        "test_confusion_matrix":
            cm.tolist(),

        "top_feature_importance":
            importance,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # =================================================================
    # SAVE MODEL COMPARISON
    # =================================================================

    validation_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # =================================================================
    # FINAL OUTPUT
    # =================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL MODEL SAVED"
    )

    print(
        "=" * 78
    )

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"\nMetadata:\n{METADATA_PATH}"
    )

    print(
        f"\nComparison:\n{COMPARISON_PATH}"
    )

    print(
        f"\nTest predictions:\n"
        f"{TEST_PREDICTIONS_PATH}"
    )

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 78
    )

    print(
        f"Selected model : "
        f"{best_model_name}"
    )

    print(
        f"Validation Macro F1 : "
        f"{best_validation_macro_f1:.4f}"
    )

    print(
        f"Test Macro F1       : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test PRE_FAULT F1   : "
        f"{test_metrics['prefault_f1']:.4f}"
    )

    print(
        f"Test PRE_FAULT Recall: "
        f"{test_metrics['prefault_recall']:.4f}"
    )

    print(
        f"Test ACTIVE F1      : "
        f"{test_metrics['active_f1']:.4f}"
    )

    print(
        f"Test ACTIVE Recall  : "
        f"{test_metrics['active_recall']:.4f}"
    )

    print()
    print(
        "Training completed."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()