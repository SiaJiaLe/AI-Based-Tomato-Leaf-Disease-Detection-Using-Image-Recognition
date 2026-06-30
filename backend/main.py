"""FastAPI application entry point — DDD-wired, production-ready."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from infrastructure.ml.resnet34_inferencer import ResNet34Inferencer
from infrastructure.persistence.database import engine
from infrastructure.persistence.models import Base
from interface.middleware.cors import setup_cors
from interface.routers import health_router, prediction_router
from interface.routers import session_router, treatment_router
from shared.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    # Create tables if they don't exist (Alembic handles production migrations;
    # this is a safety net for fresh dev containers).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Load ONNX model once — stored on app.state for injection into route handlers
    app.state.inferencer = ResNet34Inferencer.from_paths(
        settings.model_path, settings.labels_path
    )
    print(f"Model loaded: {settings.model_path}")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    await engine.dispose()


app = FastAPI(
    title="Tomato Leaf Disease Advisory API",
    version="1.0.0",
    description="ResNet34-based disease classification with severity estimation and treatment recommendations.",
    lifespan=lifespan,
)

setup_cors(app)

app.include_router(health_router.router, prefix="/api/v1", tags=["health"])
app.include_router(prediction_router.router, prefix="/api/v1", tags=["predictions"])
app.include_router(session_router.router, prefix="/api/v1", tags=["sessions"])
app.include_router(treatment_router.router, prefix="/api/v1", tags=["treatments"])


@app.get("/")
def root():
    return {"message": "Tomato Leaf Disease Advisory API — see /docs for endpoints"}
