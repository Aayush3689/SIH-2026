from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class RULModel:
    """
    Runtime wrapper for the final RUL V4 model.

    The saved artifact contains the complete sklearn Pipeline,
    including preprocessing and the trained regressor.
    """

    DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "rul_v4"
        / "final_rul_v4.joblib"
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
    ) -> None:

        self.model_path = Path(
            model_path
            if model_path is not None
            else self.DEFAULT_MODEL_PATH
        )

        self.pipeline: Any = None

        self.target: str | None = None
        self.window_samples: int | None = None
        self.windows_per_flight: int | None = None

        self.sensor_features: list[str] = []
        self.mission_numeric_features: list[str] = []
        self.mission_categorical_features: list[str] = []

        self.split_strategy: str | None = None
        self.evaluation_unit: str | None = None
        self.warning: str | None = None

        self._loaded = False

    # ========================================================
    # LOAD
    # ========================================================

    def load(self) -> None:
        """
        Load the complete RUL pipeline once.
        """

        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"RUL model not found:\n"
                f"{self.model_path}"
            )

        bundle = joblib.load(
            self.model_path
        )

        if not isinstance(bundle, dict):
            raise TypeError(
                "Expected RUL model bundle "
                "to be a dict."
            )

        required_keys = {
            "pipeline",
            "target",
            "window_samples",
            "windows_per_flight",
            "sensor_features",
            "mission_numeric_features",
            "mission_categorical_features",
        }

        missing = (
            required_keys
            - set(bundle.keys())
        )

        if missing:
            raise ValueError(
                "RUL model bundle is missing "
                f"required keys: {sorted(missing)}"
            )

        self.pipeline = bundle[
            "pipeline"
        ]

        self.target = bundle[
            "target"
        ]

        self.window_samples = int(
            bundle[
                "window_samples"
            ]
        )

        self.windows_per_flight = int(
            bundle[
                "windows_per_flight"
            ]
        )

        self.sensor_features = list(
            bundle[
                "sensor_features"
            ]
        )

        self.mission_numeric_features = list(
            bundle[
                "mission_numeric_features"
            ]
        )

        self.mission_categorical_features = list(
            bundle[
                "mission_categorical_features"
            ]
        )

        self.split_strategy = bundle.get(
            "split_strategy"
        )

        self.evaluation_unit = bundle.get(
            "evaluation_unit"
        )

        self.warning = bundle.get(
            "warning"
        )

        # ----------------------------------------------------
        # Validate pipeline
        # ----------------------------------------------------

        if not hasattr(
            self.pipeline,
            "predict",
        ):
            raise TypeError(
                "Saved RUL pipeline does not "
                "implement predict()."
            )

        if self.window_samples != 8:
            raise ValueError(
                "Unexpected RUL window size: "
                f"{self.window_samples}. "
                "Expected 8."
            )

        if self.windows_per_flight != 5:
            raise ValueError(
                "Unexpected windows_per_flight: "
                f"{self.windows_per_flight}. "
                "Expected 5."
            )

        if len(
            self.sensor_features
        ) != 17:
            raise ValueError(
                "Unexpected sensor feature count: "
                f"{len(self.sensor_features)}. "
                "Expected 17."
            )

        if len(
            self.mission_numeric_features
        ) != 9:
            raise ValueError(
                "Unexpected mission numeric "
                f"feature count: "
                f"{len(self.mission_numeric_features)}. "
                "Expected 9."
            )

        if len(
            self.mission_categorical_features
        ) != 5:
            raise ValueError(
                "Unexpected mission categorical "
                f"feature count: "
                f"{len(self.mission_categorical_features)}. "
                "Expected 5."
            )

        self._loaded = True

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features: pd.DataFrame,
    ) -> list[float]:
        """
        Predict RUL for one or more already-created
        model feature rows.

        IMPORTANT:
        The feature dataframe must follow the same
        structure expected by the saved training pipeline.
        """

        if not self._loaded:
            self.load()

        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if features.empty:
            return []

        predictions = self.pipeline.predict(
            features
        )

        # RUL cannot be negative for our application.
        predictions = [
            max(
                0.0,
                float(value),
            )
            for value in predictions
        ]

        return predictions

    # ========================================================
    # PREDICT ONE
    # ========================================================

    def predict_one(
        self,
        features: dict[str, Any],
    ) -> float:
        """
        Convenience method for one prepared feature row.
        """

        dataframe = pd.DataFrame(
            [features]
        )

        predictions = self.predict(
            dataframe
        )

        if not predictions:
            raise ValueError(
                "No RUL prediction generated."
            )

        return predictions[0]

    # ========================================================
    # INFO
    # ========================================================

    def info(self) -> dict[str, Any]:
        """
        Return runtime model metadata.
        """

        if not self._loaded:
            self.load()

        return {
            "model_type":
                type(
                    self.pipeline
                ).__name__,

            "model_path":
                str(
                    self.model_path
                ),

            "target":
                self.target,

            "window_samples":
                self.window_samples,

            "windows_per_flight":
                self.windows_per_flight,

            "sensor_feature_count":
                len(
                    self.sensor_features
                ),

            "mission_numeric_feature_count":
                len(
                    self.mission_numeric_features
                ),

            "mission_categorical_feature_count":
                len(
                    self.mission_categorical_features
                ),

            "split_strategy":
                self.split_strategy,

            "evaluation_unit":
                self.evaluation_unit,

            "warning":
                self.warning,

            "loaded":
                self.is_loaded,
        }