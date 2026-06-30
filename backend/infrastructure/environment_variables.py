"""Legacy settings shim — proxies to shared.config.settings."""
from shared.config import settings as _settings


class _LegacySettings:
    PROJECT_NAME: str = "Tomato Disease Detection API"
    VERSION: str = "1.0.0"

    @property
    def MODEL_PATH(self) -> str:
        return _settings.model_path

    @property
    def LABELS_PATH(self) -> str:
        return _settings.labels_path


settings = _LegacySettings()
