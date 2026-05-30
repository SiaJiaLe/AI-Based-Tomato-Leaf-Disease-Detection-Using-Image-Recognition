from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from infrastructure.environment_variables import settings
from infrastructure.onnx_inference_engine import onnx_engine
from application.disease_prediction_logic import prediction_service
from domain.prediction_data_models import PredictionResponse, ErrorResponse

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    onnx_engine.load()

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

@app.post("/api/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        result = prediction_service.predict(contents)
        return PredictionResponse(
            success=True,
            filename=file.filename,
            prediction=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
