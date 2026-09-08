from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_WINDOW_SAMPLES = 8


class RULFeatureBuilder:
    """
    Runtime feature builder for RUL V4.

    RUL V4 operates on the latest 8 samples of the current flight.

    The important difference from the previous implementation is:

        Previous:
            only model metadata groups were created

        Current:
            the actual sklearn pipeline input contract is inspected
            and every required engineered feature is constructed.

    Supported runtime feature patterns:

        <feature>
        <feature>_mean
        <feature>_std
        <feature>_last
        <feature>_slope

    Special features:

        sample_count
        flight_duration_min
    """

    def __init__(
        self,
        *,
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
    ) -> None:

        if window_samples <= 0:
            raise ValueError(
                "window_samples must be greater than zero."
            )

        self.window_samples = window_samples

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
    ) -> None:

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "telemetry must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Telemetry dataframe is empty."
            )

        required = {
            "timestamp",
            "engine_id",
            "flight_id",
        }

        missing = sorted(
            required - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Missing required telemetry columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing
                )
            )

    # =========================================================================
    # PREPARE
    # =========================================================================

    @staticmethod
    def _prepare(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        work = df.copy()

        work["timestamp"] = pd.to_datetime(
            work["timestamp"],
            errors="coerce",
        )

        if work["timestamp"].isna().any():
            raise ValueError(
                "Invalid timestamp found in telemetry."
            )

        return (
            work.sort_values(
                [
                    "engine_id",
                    "flight_id",
                    "timestamp",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    # =========================================================================
    # CURRENT FLIGHT
    # =========================================================================

    @staticmethod
    def _current_flight(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        latest_engine = df[
            "engine_id"
        ].iloc[-1]

        latest_flight = df[
            "flight_id"
        ].iloc[-1]

        flight = df[
            (df["engine_id"] == latest_engine)
            & (df["flight_id"] == latest_flight)
        ].copy()

        if flight.empty:
            raise ValueError(
                "No telemetry found for current engine/flight."
            )

        return flight

    # =========================================================================
    # LAST WINDOW
    # =========================================================================

    def _last_window(
        self,
        flight: pd.DataFrame,
    ) -> pd.DataFrame:

        if len(flight) < self.window_samples:
            raise ValueError(
                "Not enough telemetry samples for RUL V4 "
                f"window. Required={self.window_samples}, "
                f"available={len(flight)}."
            )

        return (
            flight.tail(
                self.window_samples
            )
            .copy()
            .reset_index(drop=True)
        )

    # =========================================================================
    # PIPELINE INPUT CONTRACT
    # =========================================================================

    @staticmethod
    def _get_pipeline_input_features(
        rul_model: Any,
    ) -> list[str]:
        """
        Extract the actual columns expected by the trained sklearn pipeline.

        The RUL model uses a sklearn ColumnTransformer. The metadata fields
        alone are not sufficient because the pipeline expects engineered
        columns such as rpm_mean, rpm_std, rpm_last, rpm_slope, etc.
        """

        pipeline = getattr(
            rul_model,
            "pipeline",
            None,
        )

        if pipeline is None:
            raise ValueError(
                "RUL model does not expose a fitted pipeline."
            )

        # ------------------------------------------------------------------
        # Preferred path:
        # sklearn Pipeline -> fitted ColumnTransformer
        # ------------------------------------------------------------------

        transformer = None

        if hasattr(
            pipeline,
            "named_steps",
        ):

            for step in pipeline.named_steps.values():

                if hasattr(
                    step,
                    "transformers_",
                ):

                    transformer = step
                    break

        # ------------------------------------------------------------------
        # Direct ColumnTransformer
        # ------------------------------------------------------------------

        if transformer is None and hasattr(
            pipeline,
            "transformers_",
        ):

            transformer = pipeline

        # ------------------------------------------------------------------
        # Feature names from fitted ColumnTransformer
        # ------------------------------------------------------------------

        if transformer is not None:

            features: list[str] = []

            for (
                _name,
                _estimator,
                columns,
            ) in transformer.transformers_:

                if columns == "drop":
                    continue

                if columns == "passthrough":
                    continue

                if isinstance(
                    columns,
                    (list, tuple),
                ):

                    features.extend(
                        str(column)
                        for column in columns
                    )

            if features:

                # Preserve order, remove duplicates.
                return list(
                    dict.fromkeys(
                        features
                    )
                )

        # ------------------------------------------------------------------
        # Fallback: feature_names_in_
        # ------------------------------------------------------------------

        if hasattr(
            pipeline,
            "feature_names_in_",
        ):

            return [
                str(column)
                for column
                in pipeline.feature_names_in_
            ]

        raise ValueError(
            "Unable to determine RUL pipeline input "
            "feature names."
        )

    # =========================================================================
    # BASE COLUMN
    # =========================================================================

    @staticmethod
    def _base_column(
        feature_name: str,
    ) -> tuple[str, str]:
        """
        Resolve:

            rpm_mean  -> (rpm, mean)
            rpm_std   -> (rpm, std)
            rpm_last  -> (rpm, last)
            rpm_slope -> (rpm, slope)
        """

        for suffix, operation in (
            ("_mean", "mean"),
            ("_std", "std"),
            ("_last", "last"),
            ("_slope", "slope"),
        ):

            if feature_name.endswith(
                suffix
            ):

                return (
                    feature_name[
                        :-len(suffix)
                    ],
                    operation,
                )

        return (
            feature_name,
            "direct",
        )

    # =========================================================================
    # NUMERIC CONVERSION
    # =========================================================================

    @staticmethod
    def _numeric_series(
        window: pd.DataFrame,
        column: str,
    ) -> pd.Series:

        if column not in window.columns:

            raise ValueError(
                f"Cannot construct RUL feature. "
                f"Base column '{column}' is missing."
            )

        return pd.to_numeric(
            window[column],
            errors="coerce",
        )

    # =========================================================================
    # SLOPE
    # =========================================================================

    @staticmethod
    def _calculate_slope(
        values: pd.Series,
        window: pd.DataFrame,
    ) -> float:
        """
        Calculate linear slope across the 8-sample window.

        x-axis:
            elapsed_s when available

        fallback:
            sample index
        """

        y = pd.to_numeric(
            values,
            errors="coerce",
        )

        valid = y.notna()

        if valid.sum() < 2:
            return np.nan

        y = y[valid].to_numpy(
            dtype=float
        )

        if (
            "elapsed_s" in window.columns
        ):

            elapsed = pd.to_numeric(
                window.loc[
                    valid,
                    "elapsed_s",
                ],
                errors="coerce",
            )

            if elapsed.notna().sum() >= 2:

                x = elapsed.to_numpy(
                    dtype=float
                )

            else:

                x = np.arange(
                    len(y),
                    dtype=float,
                )

        else:

            x = np.arange(
                len(y),
                dtype=float,
            )

        if len(
            np.unique(x)
        ) < 2:

            return 0.0

        slope = np.polyfit(
            x,
            y,
            1,
        )[0]

        return float(
            slope
        )

    # =========================================================================
    # NUMERIC FEATURE
    # =========================================================================

    @classmethod
    def _numeric_feature(
        cls,
        window: pd.DataFrame,
        feature_name: str,
    ) -> float:

        # ------------------------------------------------------------------
        # Direct column
        # ------------------------------------------------------------------

        if feature_name in window.columns:

            values = cls._numeric_series(
                window,
                feature_name,
            )

            return float(
                values.iloc[-1]
            )

        # ------------------------------------------------------------------
        # Special: sample_count
        # ------------------------------------------------------------------

        if feature_name == "sample_count":

            return float(
                len(window)
            )

        # ------------------------------------------------------------------
        # Special: flight_duration_min
        # ------------------------------------------------------------------

        if (
            feature_name
            == "flight_duration_min"
        ):

            if "elapsed_s" in window.columns:

                elapsed = pd.to_numeric(
                    window["elapsed_s"],
                    errors="coerce",
                )

                if elapsed.notna().any():

                    return float(
                        elapsed.max()
                        / 60.0
                    )

            duration = (
                window["timestamp"].max()
                - window["timestamp"].min()
            ).total_seconds()

            return float(
                duration / 60.0
            )

        # ------------------------------------------------------------------
        # Resolve suffix
        # ------------------------------------------------------------------

        base, operation = cls._base_column(
            feature_name
        )

        if base not in window.columns:

            raise ValueError(
                f"Cannot construct RUL feature "
                f"'{feature_name}'. "
                f"Base column '{base}' is missing."
            )

        values = cls._numeric_series(
            window,
            base,
        )

        # ------------------------------------------------------------------
        # Mean
        # ------------------------------------------------------------------

        if operation == "mean":

            return float(
                values.mean()
            )

        # ------------------------------------------------------------------
        # Standard deviation
        # ------------------------------------------------------------------

        if operation == "std":

            return float(
                values.std(
                    ddof=0
                )
            )

        # ------------------------------------------------------------------
        # Last
        # ------------------------------------------------------------------

        if operation == "last":

            return float(
                values.iloc[-1]
            )

        # ------------------------------------------------------------------
        # Slope
        # ------------------------------------------------------------------

        if operation == "slope":

            return cls._calculate_slope(
                values,
                window,
            )

        raise ValueError(
            f"Unsupported RUL feature "
            f"operation for '{feature_name}'."
        )

    # =========================================================================
    # CATEGORICAL FEATURE
    # =========================================================================

    @staticmethod
    def _categorical_feature(
        window: pd.DataFrame,
        feature_name: str,
    ) -> Any:

        if feature_name not in window.columns:

            raise ValueError(
                f"Missing RUL categorical feature: "
                f"{feature_name}"
            )

        values = (
            window[feature_name]
            .dropna()
        )

        if values.empty:
            return None

        # Runtime state = latest value.
        return values.iloc[-1]

    # =========================================================================
    # BUILD
    # =========================================================================

    def build(
        self,
        telemetry: pd.DataFrame,
        *,
        rul_model: Any,
    ) -> pd.DataFrame:
        """
        Build exactly the columns required by the trained RUL pipeline.
        """

        self._validate_dataframe(
            telemetry
        )

        rul_model.load()

        if (
            rul_model.window_samples
            != self.window_samples
        ):

            raise ValueError(
                "RUL window mismatch. "
                f"Builder={self.window_samples}, "
                f"Model={rul_model.window_samples}."
            )

        df = self._prepare(
            telemetry
        )

        flight = self._current_flight(
            df
        )

        window = self._last_window(
            flight
        )

        # ------------------------------------------------------------------
        # Get ACTUAL trained pipeline contract
        # ------------------------------------------------------------------

        required_features = (
            self._get_pipeline_input_features(
                rul_model
            )
        )

        if not required_features:

            raise ValueError(
                "RUL pipeline has no input features."
            )

        # ------------------------------------------------------------------
        # Build one row
        # ------------------------------------------------------------------

        row: dict[str, Any] = {}

        for feature in required_features:

            # Categorical
            if (
                feature in getattr(
                    rul_model,
                    "mission_categorical_features",
                    [],
                )
            ):

                row[feature] = (
                    self._categorical_feature(
                        window,
                        feature,
                    )
                )

                continue

            # Numeric
            row[feature] = (
                self._numeric_feature(
                    window,
                    feature,
                )
            )

        result = pd.DataFrame(
            [row]
        )

        # ------------------------------------------------------------------
        # Ensure exact column order
        # ------------------------------------------------------------------

        result = result[
            required_features
        ].copy()

        # ------------------------------------------------------------------
        # Numeric cleanup
        # ------------------------------------------------------------------

        categorical_features = set(
            getattr(
                rul_model,
                "mission_categorical_features",
                [],
            )
        )

        numeric_features = [
            feature
            for feature in required_features
            if feature not in categorical_features
        ]

        for feature in numeric_features:

            result[feature] = pd.to_numeric(
                result[feature],
                errors="coerce",
            )

        # ------------------------------------------------------------------
        # Final validation
        # ------------------------------------------------------------------

        missing = [
            feature
            for feature in required_features
            if feature not in result.columns
        ]

        if missing:

            raise ValueError(
                "RUL feature builder failed to create:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing
                )
            )

        return result

    # =========================================================================
    # BUILD FROM MODEL
    # =========================================================================

    def build_from_model(
        self,
        telemetry: pd.DataFrame,
        rul_model: Any,
    ) -> pd.DataFrame:
        """
        Build RUL input directly from the fitted RUL model's
        actual sklearn pipeline contract.
        """

        return self.build(
            telemetry,
            rul_model=rul_model,
        )