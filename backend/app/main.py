"""
app/main.py
-----------
FastAPI application entry point for Iris Classifier.

Run with:
    uvicorn app.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_NAME, APP_VERSION, APP_DESCRIPTION
from app.services.pipeline import pipeline
from app.models.schemas import HealthResponse
from app.api.v1.endpoints.predict    import router as predict_router
from app.api.v1.endpoints.model_info import router as model_router


# ── Startup: train the model once ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pipeline already built eagerly at module import; just log it
    print(f"[Startup] Pipeline ready. Accuracy={pipeline.metrics.get('accuracy','?')}")
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router,  prefix="/api/v1")
app.include_router(model_router,    prefix="/api/v1")


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=APP_NAME,
        version=APP_VERSION,
        trained=pipeline._trained,
    )
