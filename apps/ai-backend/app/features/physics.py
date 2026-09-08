from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

AI_BACKEND_DIR = Path(
    __file__
).resolve().parents[2]

PHYSICS_MODEL_DIR = (
    AI_BACKEND_DIR
    / "training"
    / "datasets"
    / "processed"
)


# =============================================================================
# PHYSICS MODEL CONTRACT
# =============================================================================

PHYSICS_INPUTS = [
    "rpm",
    "throttle_position_pct",
    "engine_load_pct",
    "fuel_flow_lph",
    "planned_altitude_ft",
    "intake_manifold_pressure_kpa",
    "ambient_temp_c",
    "airspeed_kts",
    "fuel_pressure_kpa",
    "injection_timing_deg_btdc",
]


PHYSICS_TARGETS = {
    "egt": "egt_c",
    "cht": "cht_c",
    "oil_temperature": "oil_temperature_c",
}


PHYSICS_OUTPUTS = {
    "egt": (
        "egt_expected",
        "egt_residual",
    ),
    "cht": (
        "cht_expected",
        "cht_residual",
    ),
    "oil_temperature": (
        "oil_temperature_expected",
        "oil_temperature_residual",
    ),
}


PHYSICS_MODEL_FILES = {
    "egt": "physics_egt_baseline.joblib",
    "cht": "physics_cht_baseline.joblib",
    "oil_temperature": (
        "physics_oil_temperature_baseline.joblib"
    ),
}


# =============================================================================
# HELPERS
# =============================================================================

def _validate_columns(
    df: pd.DataFrame,
) -> None:

    required = set(
        PHYSICS_INPUTS
        + list(PHYSICS_TARGETS.values())
    )

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing columns required for physics features:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


def _load_model(
    name: str,
):
    """
    Load one of the saved training-time physics baselines.
    """

    if name not in PHYSICS_MODEL_FILES:
        raise ValueError(
            f"Unknown physics model: {name}"
        )

    model_path = (
        PHYSICS_MODEL_DIR
        / PHYSICS_MODEL_FILES[name]
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Physics baseline model not found:\n"
            f"{model_path}"
        )

    model = joblib.load(
        model_path
    )

    if not hasattr(
        model,
        "predict",
    ):
        raise TypeError(
            f"Loaded physics model '{name}' "
            "does not implement predict()."
        )

    return model


def _numeric_frame(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convert physics inputs to numeric values.

    The saved sklearn pipeline contains the training-time
    median imputer, so NaNs are intentionally preserved here.
    """

    result = df[
        columns
    ].copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


# =============================================================================
# PUBLIC API
# =============================================================================

def calculate_physics_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate physics expected values and residuals using
    the exact saved training-time baseline models.

    Produces:

        egt_expected
        egt_residual

        cht_expected
        cht_residual

        oil_temperature_expected
        oil_temperature_residual

    IMPORTANT:
        The saved models already contain:
            SimpleImputer(strategy="median")
            StandardScaler()
            LinearRegression()

        Therefore runtime must NOT recreate those transformations
        manually.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Telemetry dataframe is empty."
        )

    _validate_columns(
        df
    )

    result = df.copy()

    # -------------------------------------------------------------------------
    # Load saved baseline models
    # -------------------------------------------------------------------------

    models = {
        name: _load_model(name)
        for name in PHYSICS_TARGETS
    }

    # -------------------------------------------------------------------------
    # Build numeric input frame
    # -------------------------------------------------------------------------

    X = _numeric_frame(
        result,
        PHYSICS_INPUTS,
    )

    # -------------------------------------------------------------------------
    # EGT
    # -------------------------------------------------------------------------

    egt_model = models["egt"]

    egt_expected = egt_model.predict(
        X
    )

    result[
        "egt_expected"
    ] = egt_expected

    result[
        "egt_residual"
    ] = (
        pd.to_numeric(
            result["egt_c"],
            errors="coerce",
        )
        - result["egt_expected"]
    )

    # -------------------------------------------------------------------------
    # CHT
    # -------------------------------------------------------------------------

    cht_model = models["cht"]

    cht_expected = cht_model.predict(
        X
    )

    result[
        "cht_expected"
    ] = cht_expected

    result[
        "cht_residual"
    ] = (
        pd.to_numeric(
            result["cht_c"],
            errors="coerce",
        )
        - result["cht_expected"]
    )

    # -------------------------------------------------------------------------
    # Oil temperature
    # -------------------------------------------------------------------------

    oil_temperature_model = models[
        "oil_temperature"
    ]

    oil_temperature_expected = (
        oil_temperature_model.predict(
            X
        )
    )

    result[
        "oil_temperature_expected"
    ] = oil_temperature_expected

    result[
        "oil_temperature_residual"
    ] = (
        pd.to_numeric(
            result["oil_temperature_c"],
            errors="coerce",
        )
        - result[
            "oil_temperature_expected"
        ]
    )

    # -------------------------------------------------------------------------
    # Clean infinite values
    # -------------------------------------------------------------------------

    physics_columns = [
        "egt_expected",
        "egt_residual",
        "cht_expected",
        "cht_residual",
        "oil_temperature_expected",
        "oil_temperature_residual",
    ]

    result[
        physics_columns
    ] = result[
        physics_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


# =============================================================================
# OPTIONAL MODEL INFORMATION
# =============================================================================

def get_physics_model_paths() -> dict[str, str]:
    """
    Return the resolved paths of all physics baseline models.
    """

    return {
        name: str(
            PHYSICS_MODEL_DIR
            / filename
        )
        for name, filename
        in PHYSICS_MODEL_FILES.items()
    }


def validate_physics_models() -> None:
    """
    Verify that all three saved physics models exist
    and can be loaded.
    """

    for name in PHYSICS_TARGETS:

        model = _load_model(
            name
        )

        if not hasattr(
            model,
            "predict",
        ):
            raise TypeError(
                f"Physics model '{name}' "
                "is invalid."
            )