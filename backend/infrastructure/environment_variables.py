import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Tomato Disease Detection API"
    VERSION: str = "1.0.0"
    
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/model/best_model.onnx")
    LABELS_PATH: str = os.getenv("LABELS_PATH", "/app/model/class_labels.json")

settings = Settings()
