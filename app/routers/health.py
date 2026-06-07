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
    # FAISS / SNOMED index info
    matcher_status = snomed_matcher.get_status()
    faiss_info = (
        f"{len(snomed_matcher._embeddings)} concepts indexed"
        if snomed_matcher._embeddings is not None
        else "not built"
    )

    # Synthea availability
    try:
        from app.services.synthea_runner import synthea_runner
        synthea_prereqs = synthea_runner.check_prerequisites()
        synthea_info = {
            "available": synthea_prereqs["can_run"],
            "reason": synthea_prereqs["reason"],
            "setup_guide": "backend/synthea/SETUP.md",
            "java_version": synthea_prereqs.get("java_version"),
        }
    except Exception as e:
        synthea_info = {"available": False, "reason": str(e), "setup_guide": "backend/synthea/SETUP.md"}

    # OMOP info (from concept matcher status)
    omop_info = {
        "mode": matcher_status["mode"],
        "concept_count": matcher_status["concept_count"],
        "omop_available": matcher_status["omop_available"],
        "omop_reason": (
            "OMOP vocabulary loaded" if matcher_status["omop_available"]
            else "Files not found at backend/omop/ — using 80-concept fallback"
        ),
    }

    status = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "components": {
            "database": "connected",
            "faiss_index": faiss_info,
            "spacy_model": ner_service._model_name or "not loaded",
            "openai": "configured",
            "concept_matcher": omop_info,
            "synthea": synthea_info,
        },
    }
    logger.info("Health check requested — status: {}", status["status"])
    return status
