"""Application configuration (scaffold)."""
import os


class Settings:
  model_path: str = os.getenv("MODEL_PATH", "./model_artifacts/best_model.pth")
  labels_path: str = os.getenv("LABELS_PATH", "./model_artifacts/class_labels.json")
  database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./predictions.db")
  upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")


settings = Settings()
