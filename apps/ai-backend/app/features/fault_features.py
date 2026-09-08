from __future__ import annotations

import numpy as np
import pandas as pd


ROLLING_WINDOW = 5


def rolling_mean_by_flight(
    df: pd.DataFrame,
    column: str,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """
    Exact same rolling mean logic used during Fault V4 training.

    Rolling is performed independently inside each flight.
    """
    return (
        df.groupby(
            "flight_id",
            sort=False,
        )[column]
        .transform(
            lambda s: s.rolling(
                window=window,
                min_periods=1,
            ).mean()
        )
    )


def rolling_std_by_flight(
    df: pd.DataFrame,
    column: str,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """
    Exact same rolling std logic used during Fault V4 training.

    ddof=0
    min_periods=2
    Rolling is performed independently inside each flight.
    """
    return (
        df.groupby(
            "flight_id",
            sort=False,
        )[column]
        .transform(
            lambda s: s.rolling(
                window=window,
                min_periods=2,
            ).std(ddof=0)
        )
    )


def add_fault_v4_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 13 Fault Classifier V4 derived features.

    IMPORTANT:
    - Input dataframe must already contain the 27 V2 features.
    - Required base columns:
        egt_residual
        cht_residual
        oil_temperature_residual
        fuel_pressure_kpa
        vibration_rate
        flight_id

    This intentionally reproduces the training-time feature logic.
    """

    df = df.copy()

    required_columns = {
        "flight_id",
        "egt_residual",
        "cht_residual",
        "oil_temperature_residual",
        "fuel_pressure_kpa",
        "vibration_rate",
    }

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            "Missing columns required for Fault V4 features:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    # -------------------------------------------------------------------------
    # Working signals
    # -------------------------------------------------------------------------

    # Absolute residuals capture magnitude independently of direction.
    df["_egt_abs_residual"] = df["egt_residual"].abs()

    df["_cht_abs_residual"] = df["cht_residual"].abs()

    df["_oil_temperature_abs_residual"] = (
        df["oil_temperature_residual"].abs()
    )

    df["_thermal_abs_residual_sum"] = (
        df["_egt_abs_residual"]
        + df["_cht_abs_residual"]
        + df["_oil_temperature_abs_residual"]
    )

    # -------------------------------------------------------------------------
    # Derived features
    # -------------------------------------------------------------------------

    derived = pd.DataFrame(index=df.index)

    # -------------------------------------------------------------------------
    # 1-3: Residual persistence
    # -------------------------------------------------------------------------

    derived["egt_residual_mean_5"] = rolling_mean_by_flight(
        df,
        "_egt_abs_residual",
        ROLLING_WINDOW,
    )

    derived["cht_residual_mean_5"] = rolling_mean_by_flight(
        df,
        "_cht_abs_residual",
        ROLLING_WINDOW,
    )

    derived["oil_temperature_residual_mean_5"] = rolling_mean_by_flight(
        df,
        "_oil_temperature_abs_residual",
        ROLLING_WINDOW,
    )

    # -------------------------------------------------------------------------
    # 4-6: Signed residual persistence
    # -------------------------------------------------------------------------

    derived["egt_signed_residual_mean_5"] = rolling_mean_by_flight(
        df,
        "egt_residual",
        ROLLING_WINDOW,
    )

    derived["cht_signed_residual_mean_5"] = rolling_mean_by_flight(
        df,
        "cht_residual",
        ROLLING_WINDOW,
    )

    derived["oil_temperature_signed_residual_mean_5"] = (
        rolling_mean_by_flight(
            df,
            "oil_temperature_residual",
            ROLLING_WINDOW,
        )
    )

    # -------------------------------------------------------------------------
    # 7: Total thermal deviation persistence
    # -------------------------------------------------------------------------

    derived["thermal_abs_residual_mean_5"] = rolling_mean_by_flight(
        df,
        "_thermal_abs_residual_sum",
        ROLLING_WINDOW,
    )

    # -------------------------------------------------------------------------
    # 8: Thermal residual interaction
    # -------------------------------------------------------------------------

    df["_thermal_residual_product"] = (
        df["egt_residual"]
        * df["cht_residual"]
    )

    derived["thermal_residual_product"] = (
        df["_thermal_residual_product"]
    )

    # -------------------------------------------------------------------------
    # 9: Thermal residual direction alignment
    # -------------------------------------------------------------------------

    derived["thermal_residual_same_sign"] = (
        (
            np.sign(df["egt_residual"])
            == np.sign(df["cht_residual"])
        )
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # 10-11: Fuel pressure persistence / variability
    # -------------------------------------------------------------------------

    derived["fuel_pressure_mean_5"] = rolling_mean_by_flight(
        df,
        "fuel_pressure_kpa",
        ROLLING_WINDOW,
    )

    derived["fuel_pressure_std_5"] = rolling_std_by_flight(
        df,
        "fuel_pressure_kpa",
        ROLLING_WINDOW,
    )

    # -------------------------------------------------------------------------
    # 12-13: Vibration trend persistence / variability
    # -------------------------------------------------------------------------

    derived["vibration_rate_mean_5"] = rolling_mean_by_flight(
        df,
        "vibration_rate",
        ROLLING_WINDOW,
    )

    derived["vibration_rate_std_5"] = rolling_std_by_flight(
        df,
        "vibration_rate",
        ROLLING_WINDOW,
    )

    # -------------------------------------------------------------------------
    # Clean
    # -------------------------------------------------------------------------

    derived = derived.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Add derived features to dataframe.
    for feature in derived.columns:
        df[feature] = derived[feature]

    # Remove internal helper columns.
    df.drop(
        columns=[
            "_egt_abs_residual",
            "_cht_abs_residual",
            "_oil_temperature_abs_residual",
            "_thermal_abs_residual_sum",
            "_thermal_residual_product",
        ],
        inplace=True,
        errors="ignore",
    )

    return df


def get_fault_v4_feature_names(
    v2_feature_names: list[str],
) -> list[str]:
    """
    Return the exact 40-feature order used by Fault V4:
    27 V2 features + 13 derived features.
    """

    derived_feature_names = [
        "egt_residual_mean_5",
        "cht_residual_mean_5",
        "oil_temperature_residual_mean_5",
        "egt_signed_residual_mean_5",
        "cht_signed_residual_mean_5",
        "oil_temperature_signed_residual_mean_5",
        "thermal_abs_residual_mean_5",
        "thermal_residual_product",
        "thermal_residual_same_sign",
        "fuel_pressure_mean_5",
        "fuel_pressure_std_5",
        "vibration_rate_mean_5",
        "vibration_rate_std_5",
    ]

    feature_names = (
        list(v2_feature_names)
        + derived_feature_names
    )

    if len(feature_names) != 40:
        raise ValueError(
            f"Fault V4 expects exactly 40 features, "
            f"got {len(feature_names)}."
        )

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            "Duplicate feature names detected in Fault V4 feature set."
        )

    return feature_names