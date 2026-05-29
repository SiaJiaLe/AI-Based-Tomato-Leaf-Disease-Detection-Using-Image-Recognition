"""Application configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: str = "./model_artifacts/best_model.pth"
    labels_path: str = "./model_artifacts/class_labels.json"
    database_url: str = (
        "postgresql+asyncpg://tomato:tomato@localhost:5432/tomato_disease"
    )
    upload_dir: str = "./uploads"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
