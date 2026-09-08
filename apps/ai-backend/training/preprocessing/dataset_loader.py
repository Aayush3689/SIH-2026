from pathlib import Path
import pandas as pd

# Project root:
# ai-backend/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset directory:
RAW_DATA_DIR = PROJECT_ROOT / "training" / "datasets" / "raw"


METADATA_FILE = RAW_DATA_DIR / "dt_engine_metadata.csv"
TELEMETRY_FILE = RAW_DATA_DIR / "dt_engine_telemetry_full.csv"
MISSION_FILE = RAW_DATA_DIR / "dt_mission_log.csv"

def load_csv(file_path: Path): 
    if not file_path.exists(): 
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        return pd.read_csv(file_path)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset: {file_path}"
        ) from exc;


def load_metadata() -> pd.DataFrame:
    """Load engine metadata."""
    return load_csv(METADATA_FILE)


def load_telemetry() -> pd.DataFrame:
    """Load engine telemetry."""
    return load_csv(TELEMETRY_FILE)


def load_missions() -> pd.DataFrame:
    """Load mission logs."""
    return load_csv(MISSION_FILE)


def load_dataset() -> dict[str, pd.DataFrame]:
    """
    Load all DRDO datasets.

    Returns:
        Dictionary containing:
        - metadata
        - telemetry
        - missions
    """
    return {
        "metadata": load_metadata(),
        "telemetry": load_telemetry(),
        "missions": load_missions(),
    }