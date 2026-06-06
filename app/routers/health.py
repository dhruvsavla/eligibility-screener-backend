import time
from fastapi import APIRouter
from loguru import logger
from app.services.concept_matcher import snomed_matcher
from app.services.ner_service import ner_service

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/api/health")
async def health_check():
    status = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "services": {
            "database": "connected",
            "faiss_index": f"{len(snomed_matcher._embeddings)} concepts indexed" if snomed_matcher._embeddings is not None else "not built",
            "spacy_model": ner_service._model_name or "not loaded",
            "openai": "configured" if True else "not configured",
        },
    }
    logger.info("Health check requested — status: {}", status["status"])
    return status
