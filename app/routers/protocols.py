import csv
import io
import json
import time
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
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
from app.services.ner_service import get_ner_service
from app.services.concept_matcher import snomed_matcher
from app.services.pdf_protocol_parser import PDFProtocolParser

router = APIRouter(prefix="/api/protocols", tags=["protocols"])


def _ingest_criteria(raw_text: str, nct_id: str) -> tuple[list[CriterionRuleCreate], str]:
    """Canonical protocol ingestion path: the LangChain document agent.

    Tries the agent (which runs scispaCy NER + SNOMED lookups + Claude extraction).
    If LangChain / the agent are unavailable, falls back to running scispaCy NER
    directly and feeding the entities to the Claude extractor — so ingestion always
    produces rules. Returns (rules, agent_trace).
    """
    try:
        from app.services.langchain_agent import get_protocol_agent
        result = get_protocol_agent().process_protocol(raw_text, nct_id)
        return result["rules"], result.get("agent_trace", "")
    except Exception as e:
        logger.warning(
            "LangChain agent unavailable for {} ({}) — using direct scispaCy+Claude extraction",
            nct_id, e,
        )
        try:
            entities = get_ner_service().extract_entities(raw_text)
        except Exception as ner_err:
            logger.warning("scispaCy NER failed: {} — extracting without entity hints", ner_err)
            entities = []
        rules = criteria_extractor.extract(raw_text, nct_id, ner_entities=entities)
        return rules, f"[direct extraction fallback: {e}]"


async def _save_protocol_and_rules(
    nct_id: str,
    title: str,
    condition: str,
    phase: str,
    sponsor: str,
    raw_criteria_text: str,
    rules: list[CriterionRuleCreate],
    agent_trace: str = "",
) -> dict:
    existing = await db.fetch_one(
        "SELECT id FROM protocols WHERE nct_id = ?", (nct_id,)
    )
    if existing:
        protocol_id = existing["id"]
        await db.execute("DELETE FROM criterion_rules WHERE protocol_id = ?", (protocol_id,))
        await db.execute(
            "UPDATE protocols SET title=?, condition=?, phase=?, sponsor=?, raw_criteria_text=?, agent_trace=? WHERE id=?",
            (title, condition, phase, sponsor, raw_criteria_text, agent_trace, protocol_id),
        )
    else:
        protocol_id = await db.execute(
            "INSERT INTO protocols (nct_id, title, condition, phase, sponsor, raw_criteria_text, agent_trace) VALUES (?,?,?,?,?,?,?)",
            (nct_id, title, condition, phase, sponsor, raw_criteria_text, agent_trace),
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


@router.post("/seed-all")
async def seed_all_protocols():
    """Seed the 10 canonical protocols (diabetes/oncology/cardiology) and flag the flagship.

    Bad NCT IDs are skipped gracefully. The first successfully-seeded diabetes
    protocol becomes the flagship that carries the 100-patient ground truth set.
    """
    from app.data.seed_protocols import seed_all
    result = await seed_all()
    return result


@router.get("/flagship")
async def get_flagship_protocol():
    """Return the current flagship protocol (the one carrying ground truth), if any."""
    proto = await db.fetch_one(
        "SELECT * FROM protocols WHERE is_flagship = 1 ORDER BY id LIMIT 1"
    )
    if not proto:
        raise HTTPException(status_code=404, detail="No flagship protocol set. Run /seed-all first.")
    return await _get_protocol_full(proto["id"])


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

        logger.info("Extracting criteria for {} via LangChain agent (Claude Sonnet)...", nct_id)
        extract_start = time.time()
        rules, agent_trace = _ingest_criteria(raw_text, nct_id)
        logger.info("✓ Extracted {} rules for {} in {}ms", len(rules), nct_id, int((time.time() - extract_start) * 1000))

        proto = await _save_protocol_and_rules(
            nct_id=nct_id,
            title=trial.get("title", ""),
            condition=trial.get("condition", request.condition),
            phase=trial.get("phase", request.phase),
            sponsor=trial.get("sponsor", ""),
            raw_criteria_text=raw_text,
            rules=rules,
            agent_trace=agent_trace,
        )
        results.append(proto)

    elapsed = int((time.time() - start) * 1000)
    logger.info("✓ Fetch pipeline complete: {} protocols in {}ms", len(results), elapsed)
    return results


@router.post("/upload")
async def upload_protocol(request: UploadProtocolRequest):
    logger.info("Manual upload for NCT ID: {}", request.nct_id)
    rules, agent_trace = _ingest_criteria(request.raw_criteria_text, request.nct_id)
    proto = await _save_protocol_and_rules(
        nct_id=request.nct_id,
        title=request.title,
        condition=request.condition or "",
        phase="",
        sponsor="",
        raw_criteria_text=request.raw_criteria_text,
        rules=rules,
        agent_trace=agent_trace,
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
            pdf_section = pdf_parser.locate_eligibility_section(full_text)
            if pdf_section.strip():
                merged = pdf_parser.merge_criteria(api_criteria_text, pdf_section, nct_id)
                enrichment_chars = len(merged) - len(api_criteria_text)
                trial["eligibility_criteria"] = merged
                pdf_used = True
        except Exception as e:
            logger.warning("⚠ PDF pipeline failed for {}: {} — using API text", nct_id, e)

    # Extract criteria and save (via LangChain agent)
    rules, agent_trace = _ingest_criteria(trial["eligibility_criteria"], nct_id)
    proto = await _save_protocol_and_rules(
        nct_id=nct_id,
        title=trial.get("title", ""),
        condition=trial.get("condition", ""),
        phase=trial.get("phase", ""),
        sponsor=trial.get("sponsor", ""),
        raw_criteria_text=trial["eligibility_criteria"],
        rules=rules,
        agent_trace=agent_trace,
    )

    elapsed = int((time.time() - start) * 1000)
    logger.info("✓ PDF fetch pipeline complete for {} in {}ms (pdf_used={})", nct_id, elapsed, pdf_used)

    return {
        "protocol": proto,
        "pdf_used": pdf_used,
        "pdf_page_count": pdf_page_count,
        "enrichment_chars": enrichment_chars,
    }


def _strip_ext(filename: str) -> str:
    for ext in (".pdf", ".csv", ".json"):
        if filename.lower().endswith(ext):
            return filename[: -len(ext)]
    return filename


def _parse_csv_protocol(raw_bytes: bytes) -> tuple[list[CriterionRuleCreate], str, int]:
    """Parse a CSV file into criterion rules.

    Accepts two layouts:
    - Structured: columns include 'criterion_text' (and optionally criterion_type/required/concept/operator/value)
    - Unstructured: any other CSV — join all cell values as raw text for Claude Sonnet
    """
    from app.models.protocol import CriterionType

    text = raw_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], "", 0

    fieldnames = [f.lower().strip() for f in (reader.fieldnames or [])]

    if "criterion_text" in fieldnames:
        # Structured CSV — map directly, skip Claude Sonnet
        rules: list[CriterionRuleCreate] = []
        for row in rows:
            normed = {k.lower().strip(): v for k, v in row.items()}
            ctext = normed.get("criterion_text", "").strip()
            if not ctext:
                continue
            raw_type = normed.get("criterion_type", normed.get("type", "inclusion")).lower()
            ctype = CriterionType.EXCLUSION if "exclu" in raw_type else CriterionType.INCLUSION
            required_val = normed.get("required", "true").lower()
            required = required_val not in ("false", "0", "no")
            rules.append(
                CriterionRuleCreate(
                    criterion_text=ctext,
                    concept=normed.get("concept", ctext[:60]),
                    operator=normed.get("operator", "presence"),
                    value=normed.get("value", ""),
                    required=required,
                    criterion_type=ctype,
                    confidence=float(normed.get("confidence", 0.9)),
                )
            )
        raw_text = "\n".join(r.get("criterion_text", "") for r in rows if r.get("criterion_text"))
        return rules, raw_text, len(rows)

    # Unstructured — flatten to text for Claude Sonnet
    raw_text = "\n".join(" | ".join(str(v) for v in row.values()) for row in rows)
    return [], raw_text, len(rows)


def _parse_json_protocol(raw_bytes: bytes) -> tuple[list[CriterionRuleCreate], str, str]:
    """Parse a JSON file. Returns (direct_rules, eligibility_text, source_label).

    Handles:
    - ClinicalTrials.gov v2 API study JSON
    - Array of criterion objects with 'criterion_text'
    - Object with eligibility_criteria / eligibilityCriteria string field
    - Fallback: serialise as text for Claude Sonnet
    """
    from app.models.protocol import CriterionType

    try:
        data = json.loads(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # ClinicalTrials.gov v2 study format
    if isinstance(data, dict) and "protocolSection" in data:
        em = data["protocolSection"].get("eligibilityModule", {})
        text = em.get("eligibilityCriteria", "")
        if text:
            return [], text, "ct_gov_json"

    # Array of criterion objects
    if isinstance(data, list) and data and isinstance(data[0], dict) and "criterion_text" in data[0]:
        rules: list[CriterionRuleCreate] = []
        for item in data:
            ctext = str(item.get("criterion_text", "")).strip()
            if not ctext:
                continue
            raw_type = str(item.get("criterion_type", item.get("type", "inclusion"))).lower()
            ctype = CriterionType.EXCLUSION if "exclu" in raw_type else CriterionType.INCLUSION
            required_val = str(item.get("required", "true")).lower()
            rules.append(
                CriterionRuleCreate(
                    criterion_text=ctext,
                    concept=str(item.get("concept", ctext[:60])),
                    operator=str(item.get("operator", "presence")),
                    value=str(item.get("value", "")),
                    required=required_val not in ("false", "0", "no"),
                    criterion_type=ctype,
                    confidence=float(item.get("confidence", 0.9)),
                )
            )
        raw_text = "\n".join(r.criterion_text for r in rules)
        return rules, raw_text, "json_array"

    # Object with a known eligibility string field
    if isinstance(data, dict):
        for key in ("eligibility_criteria", "eligibilityCriteria", "eligibility", "criteria", "text"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return [], val, f"json_field:{key}"

    # Fallback — dump JSON as text for Claude Sonnet
    fallback = json.dumps(data, indent=2)[:8000]
    return [], fallback, "json_raw"


@router.post("/upload-file")
async def upload_protocol_file(
    file: UploadFile = File(...),
    title: str = Form(""),
    condition: str = Form(""),
):
    """Upload a protocol file (PDF, CSV, or JSON) and extract I/E criteria."""
    fname = (file.filename or "").lower()
    if not any(fname.endswith(ext) for ext in (".pdf", ".csv", ".json")):
        raise HTTPException(status_code=400, detail="Only PDF, CSV, and JSON files are supported")

    start = time.time()
    logger.info("Processing uploaded protocol file: {}", file.filename)

    raw_bytes = await file.read()
    if len(raw_bytes) < 2:
        raise HTTPException(status_code=400, detail="Uploaded file appears to be empty")

    nct_id = f"UPLOAD-{int(time.time())}"
    direct_rules: list[CriterionRuleCreate] = []
    eligibility_text = ""
    source_summary = ""
    record_count = 0

    # ── PDF ──────────────────────────────────────────────────────────────────
    if fname.endswith(".pdf"):
        text = ""
        method = "pdfplumber"
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                record_count = len(pdf.pages)
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            logger.warning("⚠ pdfplumber failed: {} — trying PyMuPDF", e)

        if len(text.strip()) < 100:
            method = "PyMuPDF"
            try:
                import fitz
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                record_count = len(doc)
                text = "\n".join(page.get_text() for page in doc)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Could not extract text from PDF: {e}")

        if len(text.strip()) < 100:
            raise HTTPException(
                status_code=422,
                detail="PDF appears to be scanned or image-only — no text content found",
            )

        pdf_parser = PDFProtocolParser()
        # locate_eligibility_section searches the FULL extracted text — never truncates
        # before searching, so deep eligibility sections (e.g. pages 30-33 of 87) are found.
        eligibility_text = pdf_parser.locate_eligibility_section(text)
        source_summary = f"{record_count} pages via {method}"

    # ── CSV ──────────────────────────────────────────────────────────────────
    elif fname.endswith(".csv"):
        try:
            direct_rules, eligibility_text, record_count = _parse_csv_protocol(raw_bytes)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")
        source_summary = f"{record_count} rows"

    # ── JSON ─────────────────────────────────────────────────────────────────
    elif fname.endswith(".json"):
        direct_rules, eligibility_text, json_source = _parse_json_protocol(raw_bytes)
        record_count = len(direct_rules) if direct_rules else 1
        source_summary = json_source

    # ── Extract criteria (skip if already parsed from structured CSV/JSON) ───
    agent_trace = ""
    if direct_rules:
        rules = direct_rules
        logger.info("✓ {} rules mapped directly from structured {}", len(rules), fname.rsplit(".", 1)[-1].upper())
    else:
        if not eligibility_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any eligibility content from file")
        logger.info("Extracting criteria via LangChain agent (Claude Sonnet) ({} chars)...", len(eligibility_text))
        rules, agent_trace = _ingest_criteria(eligibility_text, nct_id)
        logger.info("✓ Extracted {} rules", len(rules))

    file_title = title.strip() or _strip_ext(file.filename or "Uploaded Protocol")
    proto = await _save_protocol_and_rules(
        nct_id=nct_id,
        title=file_title,
        condition=condition.strip(),
        phase="",
        sponsor=f"Uploaded {fname.rsplit('.', 1)[-1].upper()}",
        raw_criteria_text=eligibility_text or "\n".join(r.criterion_text for r in rules),
        rules=rules,
        agent_trace=agent_trace,
    )

    elapsed = int((time.time() - start) * 1000)
    logger.info("✓ File upload pipeline complete in {}ms ({} rules, {})", elapsed, len(rules), source_summary)
    return {
        "protocol": proto,
        "file_type": fname.rsplit(".", 1)[-1].upper(),
        "source_summary": source_summary,
        "record_count": record_count,
        "rules_extracted": len(rules),
        # kept for backward compat
        "pages_extracted": record_count,
        "extraction_method": source_summary,
    }


# Backward-compat alias
@router.post("/upload-pdf")
async def upload_protocol_pdf(
    file: UploadFile = File(...),
    title: str = Form(""),
    condition: str = Form(""),
):
    return await upload_protocol_file(file=file, title=title, condition=condition)


# ── Gold annotations + extraction accuracy (Measure 1) ───────────────────────

class GoldAnnotation(BaseModel):
    criterion_text: str
    concept: str
    operator: str
    value: str = ""
    required: bool = True
    criterion_type: str = "inclusion"


class SaveGoldAnnotationsRequest(BaseModel):
    annotations: list[GoldAnnotation]


async def _replace_gold_annotations(protocol_id: int, annotations: list[GoldAnnotation]) -> int:
    await db.execute("DELETE FROM gold_annotations WHERE protocol_id = ?", (protocol_id,))
    for a in annotations:
        ctype = "exclusion" if "exclu" in a.criterion_type.lower() else "inclusion"
        await db.execute(
            """INSERT INTO gold_annotations
               (protocol_id, criterion_text, concept, operator, value, required, criterion_type, annotator)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                protocol_id, a.criterion_text, a.concept, a.operator, a.value,
                1 if a.required else 0, ctype, "human",
            ),
        )
    return len(annotations)


@router.post("/{protocol_id}/gold-annotations")
async def save_gold_annotations(protocol_id: int, request: SaveGoldAnnotationsRequest):
    """Replace all human gold annotations for a protocol."""
    proto = await db.fetch_one("SELECT id FROM protocols WHERE id = ?", (protocol_id,))
    if not proto:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id} not found")
    saved = await _replace_gold_annotations(protocol_id, request.annotations)
    logger.info("Saved {} gold annotations for protocol {}", saved, protocol_id)
    return {"saved": saved}


@router.get("/{protocol_id}/gold-annotations")
async def get_gold_annotations(protocol_id: int):
    """Return all gold annotations for a protocol."""
    rows = await db.fetch_all(
        "SELECT * FROM gold_annotations WHERE protocol_id = ? ORDER BY id", (protocol_id,)
    )
    return rows


@router.post("/{protocol_id}/gold-annotations/import")
async def import_gold_annotations(protocol_id: int, file: UploadFile = File(...)):
    """Import gold annotations from an uploaded JSON file (offline hand-labeling)."""
    proto = await db.fetch_one("SELECT id FROM protocols WHERE id = ?", (protocol_id,))
    if not proto:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id} not found")

    raw = await file.read()
    try:
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if isinstance(data, dict) and "annotations" in data:
        data = data["annotations"]
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array of annotation objects")

    annotations: list[GoldAnnotation] = []
    for item in data:
        try:
            annotations.append(GoldAnnotation(**item))
        except Exception as e:
            logger.warning("Skipping malformed gold annotation {}: {}", item, e)
    saved = await _replace_gold_annotations(protocol_id, annotations)
    logger.info("Imported {} gold annotations for protocol {} from {}", saved, protocol_id, file.filename)
    return {"saved": saved}


@router.post("/{protocol_id}/extraction-accuracy")
async def run_extraction_accuracy(protocol_id: int):
    """Measure 1: compare Claude-extracted rules vs human gold annotations."""
    from app.services.extraction_evaluator import extraction_evaluator

    proto = await db.fetch_one("SELECT id, title FROM protocols WHERE id = ?", (protocol_id,))
    if not proto:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id} not found")

    gold_rows = await db.fetch_all(
        "SELECT criterion_text, concept, operator, value, required, criterion_type FROM gold_annotations WHERE protocol_id = ?",
        (protocol_id,),
    )
    if not gold_rows:
        raise HTTPException(
            status_code=404,
            detail="No gold annotations found. Hand-label rules on the Annotation page first.",
        )
    pred_rows = await db.fetch_all(
        "SELECT criterion_text, concept, operator, value, required, criterion_type FROM criterion_rules WHERE protocol_id = ?",
        (protocol_id,),
    )

    result = extraction_evaluator.evaluate(
        gold_rules=[dict(r) for r in gold_rows],
        predicted_rules=[dict(r) for r in pred_rows],
        protocol_title=proto.get("title", ""),
    )

    await db.execute(
        """INSERT INTO extraction_accuracy_runs
           (protocol_id, gold_count, extracted_count, matched_count, precision, recall, f1, details_json)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            protocol_id, result["gold_count"], result["extracted_count"], result["matched_count"],
            result["precision"], result["recall"], result["f1"], json.dumps(result),
        ),
    )
    return {**result, "protocol_id": protocol_id, "protocol_title": proto.get("title", "")}


@router.delete("/{protocol_id}")
async def delete_protocol(protocol_id: int):
    proto = await db.fetch_one("SELECT id FROM protocols WHERE id = ?", (protocol_id,))
    if not proto:
        raise HTTPException(status_code=404, detail="Protocol not found")
    await db.execute("DELETE FROM protocols WHERE id = ?", (protocol_id,))
    logger.info("Deleted protocol {}", protocol_id)
    return {"detail": f"Protocol {protocol_id} deleted"}
