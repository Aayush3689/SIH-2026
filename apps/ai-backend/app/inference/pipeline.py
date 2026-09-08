from pathlib import Path

from app.anomaly.model import AnomalyModel
from app.fault.model import FaultModel
from app.rul.model import RULModel

from app.features.telemetry import (
    normalize_telemetry,
    validate_sample_count,
)
from app.features.runtime_features import build_temporal_features
from app.features.physics import calculate_physics_features
from app.features.fault_features import add_fault_v4_features
from app.features.rul_features import RULFeatureBuilder


class InferencePipeline:
    """
    Complete AI inference pipeline.

    Flow:
        Raw Telemetry
            ↓
        Normalize
            ↓
        Temporal Features
            ↓
        Physics Features
            ↓
        Fault V4 Features
            ↓
        ┌──────────────┬──────────────┬──────────────┐
        ↓              ↓              ↓
      Anomaly         Fault           RUL
        └──────────────┴──────────────┴──────────────┘
                           ↓
                    Combined Result

    RUL requires at least 8 samples.
    Anomaly and Fault can run with a single sample.
    """

    RUL_MIN_SAMPLES = 8

    def __init__(self):
        base_dir = Path(__file__).resolve().parents[2]

        # ==========================================================
        # Model paths
        # ==========================================================

        anomaly_model_path = (
            base_dir
            / "models"
            / "anomaly"
            / "final_anomaly_model.joblib"
        )

        fault_model_path = (
            base_dir
            / "models"
            / "fault_classifier_v4"
            / "final_fault_classifier_v4.joblib"
        )

        rul_model_path = (
            base_dir
            / "models"
            / "rul_v4"
            / "final_rul_v4.joblib"
        )

        # ==========================================================
        # Load Anomaly Model
        # ==========================================================

        self.anomaly_model = AnomalyModel(
            anomaly_model_path
        )
        self.anomaly_model.load()

        # ==========================================================
        # Load Fault V4 Model
        # ==========================================================

        self.fault_model = FaultModel(
            fault_model_path
        )
        self.fault_model.load()

        # ==========================================================
        # Load RUL V4 Model
        # ==========================================================

        self.rul_model = RULModel(
            rul_model_path
        )
        self.rul_model.load()

        # ==========================================================
        # RUL Feature Builder
        # ==========================================================

        self.rul_features = RULFeatureBuilder()

    # =================================================================
    # Main inference
    # =================================================================

    def run(self, telemetry):
        """
        Run the complete AI inference pipeline.

        Parameters
        ----------
        telemetry : list[dict]
            Telemetry samples for the current engine/flight.

        Returns
        -------
        dict
            Combined anomaly, fault and RUL predictions.
        """

        if not telemetry:
            raise ValueError(
                "Telemetry payload cannot be empty."
            )

        # ==========================================================
        # 1. Normalize telemetry
        # ==========================================================

        df = normalize_telemetry(
            telemetry
        )

        # ==========================================================
        # 2. Minimum telemetry validation
        #
        # One sample is enough for anomaly + fault inference.
        # ==========================================================

        validate_sample_count(
            df,
            minimum_samples=1,
        )

        # ==========================================================
        # 3. Runtime-required columns
        # ==========================================================

        required_runtime_columns = [
            "engine_id",
            "flight_id",
            "elapsed_s",
        ]

        missing_columns = [
            column
            for column in required_runtime_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing runtime columns: {missing_columns}"
            )

        # ==========================================================
        # 4. Temporal features
        # ==========================================================

        df = build_temporal_features(
            df
        )

        # ==========================================================
        # 5. Physics features
        # ==========================================================

        df = calculate_physics_features(
            df
        )

        # ==========================================================
        # 6. Fault V4 features
        # ==========================================================

        df = add_fault_v4_features(
            df
        )

        # ==========================================================
        # 7. Identify current engine + flight
        # ==========================================================

        engine_id = df["engine_id"].iloc[-1]
        flight_id = df["flight_id"].iloc[-1]

        current_flight = df[
            (df["engine_id"] == engine_id)
            & (df["flight_id"] == flight_id)
        ].copy()

        if current_flight.empty:
            raise ValueError(
                "No telemetry available for current flight."
            )

        # ==========================================================
        # 8. Latest telemetry sample
        # ==========================================================

        latest = current_flight.iloc[
            [-1]
        ].copy()

        # ==========================================================
        # 9. Anomaly prediction
        # ==========================================================

        anomaly_result = self.anomaly_model.predict(
            latest
        )

        # ==========================================================
        # 10. Fault prediction
        # ==========================================================

        fault_result = self.fault_model.predict(
            latest
        )

        # ==========================================================
        # 11. RUL prediction
        #
        # RUL V4 requires at least 8 samples.
        # Otherwise return None instead of failing the request.
        # ==========================================================

        rul_result = None

        if len(current_flight) >= self.RUL_MIN_SAMPLES:
            rul_feature_row = self.rul_features.build(
                current_flight
            )

            rul_result = self.rul_model.predict(
                rul_feature_row
            )

        # ==========================================================
        # 12. Timestamp
        # ==========================================================

        timestamp = latest[
            "timestamp"
        ].iloc[0]

        if hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()
        else:
            timestamp = str(timestamp)

        # ==========================================================
        # 13. Combined result
        # ==========================================================

        return {
            "engine_id": str(engine_id),
            "flight_id": str(flight_id),
            "timestamp": timestamp,
            "anomaly": anomaly_result,
            "fault": fault_result,
            "rul": rul_result,
        }