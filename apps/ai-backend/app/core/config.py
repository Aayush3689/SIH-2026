from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "UAV Engine Digital Twin AI Backend"
    app_version: str = "0.1.0"

    environment: str = "development"

    host: str = "0.0.0.0"
    port: int = 8000

    model_dir: str = "models"
    dataset_dir: str = "training/datasets"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()