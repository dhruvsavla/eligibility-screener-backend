import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from app.config import settings
from app.database import init_db
from app.services.concept_matcher import snomed_matcher
from app.services.ner_service import ner_service
from app.routers import protocols, patients, screening, health, evaluation


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== STARTING ELIGIBILITY SCREENER API ===")
    os.makedirs("logs", exist_ok=True)

    logger.info("Initializing database...")
    await init_db()

    logger.info("Running evaluation schema migration...")
    from app.routers.evaluation import _ensure_gt_columns
    await _ensure_gt_columns()

    logger.info("Building FAISS SNOMED index...")
    snomed_matcher.build_index()

    logger.info("Loading spaCy NER model...")
    ner_service.load_model()

    logger.info("=== API READY ===")
    yield
    logger.info("=== SHUTTING DOWN ===")


app = FastAPI(
    title="Automated Eligibility Screener API",
    description="Clinical trial eligibility screening using Claude Sonnet, scispaCy, "
                "LangChain, FHIR R4, OMOP/SNOMED-CT, and Synthea",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = int((time.time() - start) * 1000)
    logger.info(
        "{} {} → {} ({}ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on {} {}: {}", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "timestamp": time.time(),
        },
    )


app.include_router(health.router)
app.include_router(protocols.router)
app.include_router(patients.router)
app.include_router(screening.router)
app.include_router(evaluation.router)
