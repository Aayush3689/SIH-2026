from pathlib import Path
import joblib


MODEL_PATHS = [
    Path("models/anomaly"),
    Path("models/fault_classifier_v4"),
    Path("models/rul_v4"),
]


def inspect_joblibs(folder: Path):

    print("\n" + "=" * 80)
    print(f"FOLDER: {folder}")
    print("=" * 80)

    if not folder.exists():
        print("Folder not found")
        return

    files = list(
        folder.glob("*.joblib")
    )

    if not files:
        print("No .joblib file found")
        return

    for path in files:

        print(f"\nFILE: {path}")

        obj = joblib.load(path)

        print(
            "Top-level type:",
            type(obj).__name__
        )

        if isinstance(obj, dict):

            print(
                "Bundle keys:"
            )

            for key, value in obj.items():

                if hasattr(value, "shape"):
                    value_info = (
                        f"{type(value).__name__}, "
                        f"shape={value.shape}"
                    )

                elif isinstance(value, (list, tuple)):
                    value_info = (
                        f"{type(value).__name__}, "
                        f"len={len(value)}"
                    )

                else:
                    value_info = (
                        type(value).__name__
                    )

                print(
                    f"  {key}: {value_info}"
                )

        else:

            print(
                "Object type:",
                type(obj)
            )

            if hasattr(
                obj,
                "feature_names_in_"
            ):
                print(
                    "feature_names_in_:",
                    list(
                        obj.feature_names_in_
                    )
                )

            if hasattr(
                obj,
                "n_features_in_"
            ):
                print(
                    "n_features_in_:",
                    obj.n_features_in_
                )

            if hasattr(
                obj,
                "classes_"
            ):
                print(
                    "classes_:",
                    obj.classes_
                )


def main():

    for folder in MODEL_PATHS:
        inspect_joblibs(folder)


if __name__ == "__main__":
    main()