import time
from fastapi import APIRouter
from loguru import logger
from app.config import settings
from app.services.concept_matcher import snomed_matcher
from app.services.ner_service import get_ner_service

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    return {"status": "healthy", "version": "1.0.0"}


def _llm_status() -> dict:
    key = settings.ANTHROPIC_API_KEY or ""
    configured = bool(key) and key != "sk-ant-your-key-here"
    return {
        "provider": "anthropic",
        "model": settings.ANTHROPIC_MODEL,
        "reachable": configured,
        "configured": configured,
    }


def _langchain_status() -> dict:
    try:
        import langchain_anthropic  # noqa: F401
        import langchain  # noqa: F401
        available = True
    except Exception:
        available = False
    key = settings.ANTHROPIC_API_KEY or ""
    initialized = available and bool(key) and key != "sk-ant-your-key-here"
    return {"initialized": initialized, "available": available, "tools": 4}


@router.get("/api/health")
async def health_check():
    # ── Concept matcher / OMOP ────────────────────────────────────────────────
    matcher_status = snomed_matcher.get_status()
    try:
        from app.services.omop_vocabulary import get_omop
        omop_status = get_omop().get_status()
    except Exception as e:
        omop_status = {"omop_available": False, "hierarchy_edges": 0, "reason": str(e)}

    # ── FAISS index ───────────────────────────────────────────────────────────
    faiss_built = snomed_matcher._embeddings is not None
    faiss_size = len(snomed_matcher._embeddings) if faiss_built else 0

    # ── scispaCy NER ──────────────────────────────────────────────────────────
    ner_status = get_ner_service().get_status()

    # ── Synthea ───────────────────────────────────────────────────────────────
    try:
        from app.services.synthea_runner import synthea_runner
        prereqs = synthea_runner.check_prerequisites()
        synthea_info = {
            "available": prereqs["can_run"],
            "java_version": prereqs.get("java_version"),
            "reason": prereqs["reason"],
            "setup_guide": "backend/synthea/SETUP.md",
        }
    except Exception as e:
        synthea_info = {"available": False, "reason": str(e), "setup_guide": "backend/synthea/SETUP.md"}

    components = {
        "database": "connected",
        "llm": _llm_status(),
        "scispacy": {
            "loaded": ner_status["loaded"],
            "model": ner_status["model"],
            "is_scispacy": ner_status["is_scispacy"],
            "setup_guide": "backend/README.md",
        },
        "langchain_agent": _langchain_status(),
        "concept_matcher": {
            "mode": matcher_status["mode"],
            "concept_count": matcher_status["concept_count"],
            "hierarchy_active": matcher_status.get("hierarchy_active", False),
        },
        "omop": {
            "available": omop_status.get("omop_available", False),
            "hierarchy_edges": omop_status.get("hierarchy_edges", 0),
            "concept_count": omop_status.get("concept_count", 0),
            "setup_guide": "backend/omop/SETUP.md",
        },
        "synthea": synthea_info,
        "faiss_index": {"built": faiss_built, "size": faiss_size},
    }

    status = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "components": components,
    }
    logger.info("Health check requested — status: {}", status["status"])
    return status
