"""FastAPI application entry point (scaffold)."""
from fastapi import FastAPI

from interface.middleware.cors import setup_cors
from interface.routers import health_router, prediction_router

app = FastAPI(title="Tomato Disease API", version="0.1.0")
setup_cors(app)
app.include_router(health_router.router, prefix="/api/v1", tags=["health"])
app.include_router(prediction_router.router, prefix="/api/v1", tags=["predictions"])
