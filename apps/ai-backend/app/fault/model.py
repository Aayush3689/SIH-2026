from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class FaultModel:
    """
    Runtime wrapper for the final Fault Classifier V4.

    Responsibilities:
    - Load trained fault-classifier bundle once.
    - Validate the saved feature contract.
    - Align incoming features to training feature order.
    - Apply saved training-time medians.
    - Return predicted fault + class probabilities.

    This class does NOT:
    - build rolling features
    - calculate physics residuals
    - decide whether anomaly is present
    - handle API requests
    """

    DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "fault_classifier_v4"
        / "final_fault_classifier_v4.joblib"
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

        self.model_name: str | None = None

        self.feature_names: list[str] = []

        self.imputation_medians: dict[
            str, float
        ] = {}

        self.class_names: list[str] = []

        self.target_column: str | None = None

        self.active_filter: str | None = None

        self.rolling_window_samples: (
            int | None
        ) = None

        self.rolling_scope: str | None = None

        self._loaded = False

    # ========================================================
    # LOAD
    # ========================================================

    def load(self) -> None:
        """
        Load the trained model bundle once.
        """

        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Fault classifier model not found:\n"
                f"{self.model_path}"
            )

        bundle = joblib.load(
            self.model_path
        )

        if not isinstance(bundle, dict):
            raise TypeError(
                "Expected fault classifier "
                "bundle to be a dict."
            )

        required_keys = {
            "model",
            "feature_names",
            "imputation_medians",
            "class_names",
        }

        missing = (
            required_keys
            - set(bundle.keys())
        )

        if missing:
            raise ValueError(
                "Fault classifier bundle is missing "
                f"required keys: {sorted(missing)}"
            )

        # ----------------------------------------------------
        # Load bundle
        # ----------------------------------------------------

        self.model = bundle["model"]

        self.model_name = bundle.get(
            "model_name"
        )

        self.feature_names = list(
            bundle["feature_names"]
        )

        self.imputation_medians = dict(
            bundle["imputation_medians"]
        )

        self.class_names = [
            str(x)
            for x in bundle["class_names"]
        ]

        self.target_column = bundle.get(
            "target_column"
        )

        self.active_filter = bundle.get(
            "active_filter"
        )

        self.rolling_window_samples = (
            bundle.get(
                "rolling_window_samples"
            )
        )

        self.rolling_scope = bundle.get(
            "rolling_scope"
        )

        # ----------------------------------------------------
        # Validate feature contract
        # ----------------------------------------------------

        if len(self.feature_names) != 40:
            raise ValueError(
                "Unexpected fault feature count: "
                f"{len(self.feature_names)}. "
                "Expected 40."
            )

        if len(
            self.imputation_medians
        ) != len(self.feature_names):

            raise ValueError(
                "Imputation median count "
                "does not match feature count."
            )

        if len(self.class_names) != 8:
            raise ValueError(
                "Unexpected fault class count: "
                f"{len(self.class_names)}. "
                "Expected 8."
            )

        # ----------------------------------------------------
        # Validate model API
        # ----------------------------------------------------

        if not hasattr(
            self.model,
            "predict",
        ):
            raise TypeError(
                "Loaded fault classifier does "
                "not implement predict()."
            )

        if not hasattr(
            self.model,
            "predict_proba",
        ):
            raise TypeError(
                "Loaded fault classifier does "
                "not implement predict_proba()."
            )

        if not hasattr(
            self.model,
            "classes_",
        ):
            raise TypeError(
                "Loaded fault classifier does "
                "not expose classes_."
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
        Build the exact 40-feature matrix expected by
        the trained model.

        Extra columns are ignored.

        Missing model features are filled using the
        saved training medians.
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

        prepared = pd.DataFrame(
            index=features.index
        )

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

            series = pd.to_numeric(
                series,
                errors="coerce",
            )

            median = (
                self.imputation_medians.get(
                    feature
                )
            )

            if median is None:
                raise ValueError(
                    "Missing saved median for "
                    f"fault feature: {feature}"
                )

            series = series.fillna(
                float(median)
            )

            prepared[
                feature
            ] = series

        if prepared.isna().any().any():

            unresolved = (
                prepared.columns[
                    prepared.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                "NaN values remain after "
                f"fault-feature imputation: "
                f"{unresolved}"
            )

        return prepared

    # ========================================================
    # CLASS NAME MAPPING
    # ========================================================

    def _resolve_class_name(
        self,
        raw_class: Any,
    ) -> str:
        """
        Resolve sklearn's raw class representation to
        the saved human-readable class name.

        class_names are expected to follow the model's
        class order.
        """

        # ----------------------------------------------------
        # Integer class index
        # ----------------------------------------------------

        if isinstance(
            raw_class,
            (
                int,
                np.integer,
            ),
        ):

            index = int(
                raw_class
            )

            if (
                0
                <= index
                < len(self.class_names)
            ):
                return self.class_names[
                    index
                ]

        # ----------------------------------------------------
        # String representation
        # ----------------------------------------------------

        raw_string = str(
            raw_class
        )

        # Exact class name match
        if raw_string in self.class_names:
            return raw_string

        # Try numeric-string index
        try:

            index = int(
                raw_string
            )

            if (
                0
                <= index
                < len(self.class_names)
            ):
                return self.class_names[
                    index
                ]

        except ValueError:
            pass

        return raw_string

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Predict fault type for one or more rows.
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

        classes = self.model.classes_

        results = []

        for row_index, raw_prediction in enumerate(
            raw_predictions
        ):

            predicted_fault = (
                self._resolve_class_name(
                    raw_prediction
                )
            )

            class_probabilities = {}

            for class_index, raw_class in enumerate(
                classes
            ):

                class_name = (
                    self._resolve_class_name(
                        raw_class
                    )
                )

                class_probabilities[
                    class_name
                ] = float(
                    probabilities[
                        row_index,
                        class_index,
                    ]
                )

            confidence = float(
                np.max(
                    probabilities[row_index]
                )
            )

            results.append(
                {
                    "fault": predicted_fault,
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
        Convenience method for a single feature dictionary.
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
        Runtime model information.
        """

        if not self._loaded:
            self.load()

        return {
            "model_type":
                type(
                    self.model
                ).__name__,

            "model_name":
                self.model_name,

            "model_path":
                str(
                    self.model_path
                ),

            "feature_count":
                len(
                    self.feature_names
                ),

            "class_count":
                len(
                    self.class_names
                ),

            "class_names":
                self.class_names,

            "target_column":
                self.target_column,

            "active_filter":
                self.active_filter,

            "rolling_window_samples":
                self.rolling_window_samples,

            "rolling_scope":
                self.rolling_scope,

            "loaded":
                self.is_loaded,
        }