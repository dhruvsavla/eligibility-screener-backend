import time
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from app import database as db
from app.models.protocol import (
    Protocol,
    FetchProtocolsRequest,
    UploadProtocolRequest,
    CriterionRuleCreate,
)
from app.services.trials_fetcher import clinical_trials_client
from app.services.criteria_extractor import criteria_extractor
from app.services.concept_matcher import snomed_matcher
from app.services.pdf_protocol_parser import PDFProtocolParser

router = APIRouter(prefix="/api/protocols", tags=["protocols"])


async def _save_protocol_and_rules(
    nct_id: str,
    title: str,
    condition: str,
    phase: str,
    sponsor: str,
    raw_criteria_text: str,
    rules: list[CriterionRuleCreate],
) -> dict:
    existing = await db.fetch_one(
        "SELECT id FROM protocols WHERE nct_id = ?", (nct_id,)
    )
    if existing:
        protocol_id = existing["id"]
        await db.execute("DELETE FROM criterion_rules WHERE protocol_id = ?", (protocol_id,))
        await db.execute(
            "UPDATE protocols SET title=?, condition=?, phase=?, sponsor=?, raw_criteria_text=? WHERE id=?",
            (title, condition, phase, sponsor, raw_criteria_text, protocol_id),
        )
    else:
        protocol_id = await db.execute(
            "INSERT INTO protocols (nct_id, title, condition, phase, sponsor, raw_criteria_text) VALUES (?,?,?,?,?,?)",
            (nct_id, title, condition, phase, sponsor, raw_criteria_text),
        )

    for rule in rules:
        matches = snomed_matcher.find_best_match(rule.concept, top_k=1)
        snomed_code = matches[0]["code"] if matches else None
        await db.execute(
            """INSERT INTO criterion_rules
               (protocol_id, criterion_text, concept, operator, value, required, criterion_type, snomed_code, confidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                protocol_id,
                rule.criterion_text,
                rule.concept,
                rule.operator,
                rule.value,
                1 if rule.required else 0,
                rule.criterion_type.value,
                snomed_code,
                rule.confidence,
            ),
        )

    return await _get_protocol_full(protocol_id)


async def _get_protocol_full(protocol_id: int) -> dict:
    proto = await db.fetch_one("SELECT * FROM protocols WHERE id = ?", (protocol_id,))
    if not proto:
        return {}
    rules = await db.fetch_all(
        "SELECT * FROM criterion_rules WHERE protocol_id = ? ORDER BY criterion_type", (protocol_id,)
    )
    proto["criterion_rules"] = rules
    proto["criteria_count"] = len(rules)
    return proto


@router.get("")
async def list_protocols():
    protocols = await db.fetch_all(
        """SELECT p.*, COUNT(cr.id) as criteria_count
           FROM protocols p
           LEFT JOIN criterion_rules cr ON cr.protocol_id = p.id
           GROUP BY p.id
           ORDER BY p.created_at DESC"""
    )
    logger.info("Listed {} protocols", len(protocols))
    return protocols


@router.get("/{protocol_id}")
async def get_protocol(protocol_id: int):
    proto = await _get_protocol_full(protocol_id)
    if not proto:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id} not found")
    return proto


@router.post("/fetch")
async def fetch_protocols(request: FetchProtocolsRequest):
    start = time.time()
    logger.info("Fetching protocols: condition='{}' phase='{}' count={}", request.condition, request.phase, request.count)

    trials = clinical_trials_client.fetch_trials(request.condition, request.phase, request.count)
    if not trials:
        raise HTTPException(status_code=404, detail="No trials found for given criteria")

    results = []
    for trial in trials:
        nct_id = trial["nct_id"]
        raw_text = trial.get("eligibility_criteria", "")

        logger.info("Extracting criteria for {} with GPT-4o...", nct_id)
        extract_start = time.time()
        rules = criteria_extractor.extract(raw_text, nct_id)
        logger.info("✓ Extracted {} rules for {} in {}ms", len(rules), nct_id, int((time.time() - extract_start) * 1000))

        proto = await _save_protocol_and_rules(
            nct_id=nct_id,
            title=trial.get("title", ""),
            condition=trial.get("condition", request.condition),
            phase=trial.get("phase", request.phase),
            sponsor=trial.get("sponsor", ""),
            raw_criteria_text=raw_text,
            rules=rules,
        )
        results.append(proto)

    elapsed = int((time.time() - start) * 1000)
    logger.info("✓ Fetch pipeline complete: {} protocols in {}ms", len(results), elapsed)
    return results


@router.post("/upload")
async def upload_protocol(request: UploadProtocolRequest):
    logger.info("Manual upload for NCT ID: {}", request.nct_id)
    rules = criteria_extractor.extract(request.raw_criteria_text, request.nct_id)
    proto = await _save_protocol_and_rules(
        nct_id=request.nct_id,
        title=request.title,
        condition=request.condition or "",
        phase="",
        sponsor="",
        raw_criteria_text=request.raw_criteria_text,
        rules=rules,
    )
    return proto


class FetchPDFRequest(BaseModel):
    nct_id: str


@router.post("/fetch-pdf")
async def fetch_protocol_pdf(request: FetchPDFRequest):
    """Fetch a specific protocol by NCT ID using the full PDF pipeline."""
    start = time.time()
    nct_id = request.nct_id.strip().upper()
    logger.info("Fetching protocol {} with PDF pipeline...", nct_id)

    # Get base trial data from API
    trial = clinical_trials_client.fetch_by_nct_id(nct_id)
    api_criteria_text = trial.get("eligibility_criteria", "")

    # Try PDF enrichment
    pdf_parser = PDFProtocolParser()
    pdf_url = pdf_parser.get_protocol_pdf_url(nct_id)
    pdf_used = False
    pdf_page_count = 0
    enrichment_chars = 0

    if pdf_url:
        try:
            full_text = pdf_parser.download_and_extract_text(pdf_url)
            import io
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(
                    __import__("requests").get(pdf_url, timeout=30).content
                )) as pdf:
                    pdf_page_count = len(pdf.pages)
            except Exception:
                pass
            pdf_section = pdf_parser.extract_eligibility_section(full_text, nct_id)
            if pdf_section.strip():
                merged = pdf_parser.merge_criteria(api_criteria_text, pdf_section, nct_id)
                enrichment_chars = len(merged) - len(api_criteria_text)
                trial["eligibility_criteria"] = merged
                pdf_used = True
        except Exception as e:
            logger.warning("⚠ PDF pipeline failed for {}: {} — using API text", nct_id, e)

    # Extract criteria and save
    rules = criteria_extractor.extract(trial["eligibility_criteria"], nct_id)
    proto = await _save_protocol_and_rules(
        nct_id=nct_id,
        title=trial.get("title", ""),
        condition=trial.get("condition", ""),
        phase=trial.get("phase", ""),
        sponsor=trial.get("sponsor", ""),
        raw_criteria_text=trial["eligibility_criteria"],
        rules=rules,
    )

    elapsed = int((time.time() - start) * 1000)
    logger.info("✓ PDF fetch pipeline complete for {} in {}ms (pdf_used={})", nct_id, elapsed, pdf_used)

    return {
        "protocol": proto,
        "pdf_used": pdf_used,
        "pdf_page_count": pdf_page_count,
        "enrichment_chars": enrichment_chars,
    }


@router.delete("/{protocol_id}")
async def delete_protocol(protocol_id: int):
    proto = await db.fetch_one("SELECT id FROM protocols WHERE id = ?", (protocol_id,))
    if not proto:
        raise HTTPException(status_code=404, detail="Protocol not found")
    await db.execute("DELETE FROM protocols WHERE id = ?", (protocol_id,))
    logger.info("Deleted protocol {}", protocol_id)
    return {"detail": f"Protocol {protocol_id} deleted"}
