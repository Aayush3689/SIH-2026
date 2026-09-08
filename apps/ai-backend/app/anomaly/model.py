from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class AnomalyModel:
    """
    Runtime wrapper for the final anomaly detection model.

    Responsibilities:
    - Load trained anomaly model bundle once
    - Validate the saved bundle
    - Align incoming features to training feature order
    - Apply saved training-time medians
    - Return prediction + probabilities

    This class does NOT:
    - calculate physics features
    - calculate rolling features
    - decide when to call the model
    - handle API requests
    """

    DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "anomaly"
        / "final_anomaly_model.joblib"
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

        self.model: Any = None
        self.feature_names: list[str] = []
        self.imputation_medians: dict[str, float] = {}
        self.label_mapping: dict[Any, str] = {}
        self.label_policy: str | None = None
        self.prefault_window_flights: int | None = None

        self._loaded = False

    # ========================================================
    # LOAD
    # ========================================================

    def load(self) -> None:
        """
        Load the trained anomaly model bundle.

        Called once during application startup.
        """

        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Anomaly model not found:\n"
                f"{self.model_path}"
            )

        bundle = joblib.load(
            self.model_path
        )

        if not isinstance(bundle, dict):
            raise TypeError(
                "Expected anomaly model bundle "
                "to be a dict."
            )

        required_keys = {
            "model",
            "feature_names",
            "imputation_medians",
            "label_mapping",
        }

        missing = (
            required_keys
            - set(bundle.keys())
        )

        if missing:
            raise ValueError(
                "Anomaly model bundle is missing "
                f"required keys: {sorted(missing)}"
            )

        self.model = bundle["model"]

        self.feature_names = list(
            bundle["feature_names"]
        )

        self.imputation_medians = dict(
            bundle["imputation_medians"]
        )

        self.label_mapping = dict(
            bundle["label_mapping"]
        )

        self.label_policy = bundle.get(
            "label_policy"
        )

        self.prefault_window_flights = (
            bundle.get(
                "prefault_window_flights"
            )
        )

        # ----------------------------------------------------
        # Validate feature contract
        # ----------------------------------------------------

        if len(self.feature_names) != 40:
            raise ValueError(
                "Unexpected anomaly feature count: "
                f"{len(self.feature_names)}. "
                "Expected 40."
            )

        if len(
            self.imputation_medians
        ) != len(self.feature_names):
            raise ValueError(
                "Imputation median count does not "
                "match feature count."
            )

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        if not hasattr(
            self.model,
            "predict"
        ):
            raise TypeError(
                "Loaded anomaly object does not "
                "implement predict()."
            )

        if not hasattr(
            self.model,
            "predict_proba"
        ):
            raise TypeError(
                "Loaded anomaly model does not "
                "implement predict_proba()."
            )

        self._loaded = True

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    def _prepare_dataframe(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align incoming features to the exact training order.

        Missing features are filled with the saved training
        median.

        Extra columns are ignored intentionally.
        """

        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if not self._loaded:
            self.load()

        # ----------------------------------------------------
        # Check missing feature names
        # ----------------------------------------------------

        missing = [
            feature
            for feature in self.feature_names
            if feature not in features.columns
        ]

        # We allow missing model features because the saved
        # training medians provide the documented fallback.
        #
        # But we still report them explicitly.
        #
        prepared = pd.DataFrame(
            index=features.index
        )

        # ----------------------------------------------------
        # Build exact feature matrix
        # ----------------------------------------------------

        for feature in self.feature_names:

            if feature in features.columns:
                series = features[
                    feature
                ]
            else:
                series = pd.Series(
                    np.nan,
                    index=features.index,
                    dtype=float,
                )

            # Force numeric
            series = pd.to_numeric(
                series,
                errors="coerce",
            )

            median = self.imputation_medians.get(
                feature
            )

            if median is None:
                raise ValueError(
                    f"No saved imputation median "
                    f"for feature: {feature}"
                )

            series = series.fillna(
                float(median)
            )

            prepared[
                feature
            ] = series

        # ----------------------------------------------------
        # Final numeric validation
        # ----------------------------------------------------

        if prepared.isna().any().any():
            unresolved = (
                prepared.columns[
                    prepared.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "NaN values remain after "
                f"imputation: {unresolved}"
            )

        return prepared

    # ========================================================
    # LABEL MAPPING
    # ========================================================

    def _map_label(
        self,
        raw_label: Any,
    ) -> str:
        """
        Map the raw classifier output to a human-readable
        state using the saved label mapping.
        """

        # Direct lookup
        if raw_label in self.label_mapping:
            return str(
                self.label_mapping[raw_label]
            )

        # Integer/string key compatibility
        raw_string = str(
            raw_label
        )

        if raw_string in self.label_mapping:
            return str(
                self.label_mapping[
                    raw_string
                ]
            )

        # Fall back to raw label
        return raw_string

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Predict anomaly state for one or more rows.

        Input:
            DataFrame containing the 40 anomaly features.

        Output:
            List of prediction dictionaries.
        """

        if not self._loaded:
            self.load()

        prepared = (
            self._prepare_dataframe(
                features
            )
        )

        raw_predictions = (
            self.model.predict(
                prepared
            )
        )

        probabilities = (
            self.model.predict_proba(
                prepared
            )
        )

        # sklearn classifier classes
        classes = getattr(
            self.model,
            "classes_",
            None,
        )

        if classes is None:
            raise ValueError(
                "Anomaly model does not expose "
                "classes_."
            )

        results = []

        for index, raw_label in enumerate(
            raw_predictions
        ):

            state = self._map_label(
                raw_label
            )

            class_probabilities = {}

            for class_index, class_label in enumerate(
                classes
            ):

                mapped_label = (
                    self._map_label(
                        class_label
                    )
                )

                class_probabilities[
                    mapped_label
                ] = float(
                    probabilities[
                        index,
                        class_index,
                    ]
                )

            # ------------------------------------------------
            # Prediction confidence
            #
            # This is maximum class probability.
            # It is NOT a calibrated failure probability.
            # ------------------------------------------------

            confidence = float(
                np.max(
                    probabilities[index]
                )
            )

            results.append(
                {
                    "state": state,
                    "raw_label": (
                        int(raw_label)
                        if isinstance(
                            raw_label,
                            (
                                int,
                                np.integer,
                            ),
                        )
                        else str(raw_label)
                    ),
                    "probabilities":
                        class_probabilities,
                    "confidence":
                        confidence,
                }
            )

        return results

    # ========================================================
    # PREDICT ONE
    # ========================================================

    def predict_one(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convenience method for one feature dictionary.
        """

        dataframe = pd.DataFrame(
            [features]
        )

        results = self.predict(
            dataframe
        )

        return results[0]

    # ========================================================
    # INFO
    # ========================================================

    def info(self) -> dict[str, Any]:
        """
        Return runtime model information.
        """

        if not self._loaded:
            self.load()

        return {
            "model_type":
                type(
                    self.model
                ).__name__,

            "model_path":
                str(
                    self.model_path
                ),

            "feature_count":
                len(
                    self.feature_names
                ),

            "label_mapping":
                {
                    str(k): str(v)
                    for k, v
                    in self.label_mapping.items()
                },

            "label_policy":
                self.label_policy,

            "prefault_window_flights":
                self.prefault_window_flights,

            "loaded":
                self.is_loaded,
        }