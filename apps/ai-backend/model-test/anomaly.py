import joblib
import pandas as pd

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model_bundle = joblib.load(
    "models/anomaly/final_anomaly_model.joblib"
)

model = model_bundle["model"]

feature_names = model_bundle["feature_names"]

print("Model loaded successfully")
print("Number of features:", len(feature_names))


# --------------------------------------------------
# Test data
# --------------------------------------------------

test_data = pd.DataFrame([
    {
        # Put all 40 required features here
        "rpm": 4500,
        "throttle_position_pct": 65,
        "cht_c": 170,
        "egt_c": 750,
        "oil_pressure_kpa": 390,
        "oil_temperature_c": 104,
        "fuel_flow_lph": 25,
        "fuel_pressure_kpa": 300,
        "intake_manifold_pressure_kpa": 115,
        "engine_vibration_mm_s": 2.0,
        "battery_voltage_v": 12.6,
        "alternator_output_a": 18,
        "injection_timing_deg_btdc": 22,
        "engine_load_pct": 60,

        "vibration_mean_30s": 2.0,
        "vibration_std_30s": 0.3,
        "rpm_mean_30s": 4500,
        "rpm_std_30s": 100,

        "egt_mean_60s": 750,
        "egt_std_60s": 15,

        "cht_mean_60s": 170,
        "cht_std_60s": 4,

        "oil_pressure_mean_60s": 390,
        "oil_pressure_std_60s": 10,

        "oil_temperature_mean_60s": 104,
        "oil_temperature_std_60s": 3,

        "egt_delta": 0,
        "cht_delta": 0,
        "oil_pressure_delta": 0,
        "vibration_delta": 0,

        "egt_rate": 0,
        "cht_rate": 0,
        "oil_pressure_rate": 0,
        "vibration_rate": 0,

        "egt_expected": 750,
        "egt_residual": 0,

        "cht_expected": 170,
        "cht_residual": 0,

        "oil_temperature_expected": 104,
        "oil_temperature_residual": 0,
    }
])


# --------------------------------------------------
# Make sure feature order is exactly correct
# --------------------------------------------------

X = test_data[feature_names]


# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]


# --------------------------------------------------
# Mapping
# --------------------------------------------------

state_mapping = {
    0: "NORMAL",
    1: "PRE_FAULT",
    2: "ACTIVE_FAULT",
}

state = state_mapping[prediction]


print("\n==============================")
print("ANOMALY MODEL OUTPUT")
print("==============================")

print("State:", state)

print("\nProbabilities:")

for class_id, probability in zip(
    model.classes_,
    probabilities
):
    print(
        f"{state_mapping[int(class_id)]:15s}: "
        f"{probability:.4f}"
    )