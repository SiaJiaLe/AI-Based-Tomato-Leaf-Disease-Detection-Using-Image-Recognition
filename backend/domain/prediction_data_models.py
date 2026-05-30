from pydantic import BaseModel

class PredictionResult(BaseModel):
    disease: str
    confidence: float

class PredictionResponse(BaseModel):
    success: bool
    filename: str
    prediction: PredictionResult

class ErrorResponse(BaseModel):
    detail: str
