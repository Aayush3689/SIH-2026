import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from training.preprocessing.dataset_loader import load_dataset
from training.features.temporal_features import add_temporal_features




def main() -> None:
    dataset = load_dataset()

    telemetry = dataset["telemetry"]

    features = add_temporal_features(telemetry)

    new_columns = [
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
        "egt_delta",
        "cht_delta",
        "oil_pressure_delta",
        "vibration_delta",
        "egt_rate",
        "cht_rate",
        "oil_pressure_rate",
        "vibration_rate",
    ]

    print("=" * 80)
    print("TEMPORAL FEATURE TEST")
    print("=" * 80)

    print(f"Original rows : {len(telemetry):,}")
    print(f"New rows      : {len(features):,}")

    print("\nNew features:")
    for column in new_columns:
        print(f"  - {column}")

    print("\nSample output:")
    print(
        features[
            [
                "engine_id",
                "flight_id",
                "timestamp",
                "engine_vibration_mm_s",
                "vibration_mean_30s",
                "vibration_std_30s",
                "egt_c",
                "egt_mean_60s",
                "egt_rate",
            ]
        ].head(15).to_string(index=False)
    )

    print("\nFeature test complete.")


if __name__ == "__main__":
    main()