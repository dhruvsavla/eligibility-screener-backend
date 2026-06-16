import json
import time
from fastapi import APIRouter, HTTPException
from loguru import logger
from app import database as db
from app.models.screening import (
    ScreenRequest,
    BatchScreenRequest,
    OverrideRequest,
    ScreeningStats,
)
from app.models.protocol import CriterionRule, CriterionType
from app.services.fhir_parser import fhir_parser
from app.services.scoring_engine import scoring_engine
from app.services.rationale_generator import rationale_generator

router = APIRouter(prefix="/api", tags=["screening"])


async def _run_screening(patient_db_id: int, protocol_id: int, use_gpt_rationale: bool = True) -> dict:
    patient_row = await db.fetch_one("SELECT * FROM patients WHERE id = ?", (patient_db_id,))
    if not patient_row:
        raise HTTPException(status_code=404, detail=f"Patient {patient_db_id} not found")

    protocol_row = await db.fetch_one("SELECT * FROM protocols WHERE id = ?", (protocol_id,))
    if not protocol_row:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id} not found")

    rule_rows = await db.fetch_all(
        "SELECT * FROM criterion_rules WHERE protocol_id = ? ORDER BY criterion_type", (protocol_id,)
    )
    if not rule_rows:
        raise HTTPException(status_code=400, detail="Protocol has no criterion rules")

    fhir_json_str = patient_row.get("fhir_json", "{}")
    try:
        fhir_bundle = json.loads(fhir_json_str)
        t0 = time.time()
        patient_data = fhir_parser.parse_bundle(fhir_bundle)
        logger.info("FHIR parsing took {}ms", int((time.time() - t0) * 1000))
    except Exception as e:
        logger.error("FHIR parse failed for patient {}: {}", patient_db_id, e)
        raise HTTPException(status_code=500, detail=f"FHIR parse error: {e}")

    rules = []
    for row in rule_rows:
        try:
            rule = CriterionRule(
                id=row["id"],
                protocol_id=row["protocol_id"],
                criterion_text=row.get("criterion_text", ""),
                concept=row.get("concept", ""),
                operator=row.get("operator", "presence"),
                value=row.get("value", ""),
                required=bool(row.get("required", True)),
                criterion_type=CriterionType(row.get("criterion_type", "inclusion")),
                snomed_code=row.get("snomed_code"),
                confidence=float(row.get("confidence", 0.5)),
            )
            rules.append(rule)
        except Exception as e:
            logger.warning("Skipping malformed rule row {}: {}", row.get("id"), e)

    t0 = time.time()
    scoring_result = scoring_engine.evaluate_patient(patient_data, rules)
    logger.info("Scoring took {}ms", int((time.time() - t0) * 1000))

    t0 = time.time()
    if use_gpt_rationale:
        rationale = rationale_generator.generate(
            patient_data, rules, scoring_result.evaluations,
            scoring_result.fit_score, scoring_result.overall_verdict.value
        )
    else:
        rationale = rationale_generator._fallback_rationale(
            patient_data, scoring_result.evaluations,
            scoring_result.fit_score, scoring_result.overall_verdict.value
        )
    logger.info("Rationale generation took {}ms (gpt={})", int((time.time() - t0) * 1000), use_gpt_rationale)

    score_breakdown_json = json.dumps(scoring_result.score_breakdown)

    result_id = await db.execute(
        """INSERT INTO screening_results
           (patient_id, protocol_id, fit_score, confidence_low, confidence_high,
            overall_verdict, rationale_summary, score_breakdown_json)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            patient_db_id,
            protocol_id,
            scoring_result.fit_score,
            scoring_result.confidence_low,
            scoring_result.confidence_high,
            scoring_result.overall_verdict.value,
            rationale,
            score_breakdown_json,
        ),
    )

    for eval_item in scoring_result.evaluations:
        await db.execute(
            """INSERT INTO criterion_evaluations
               (result_id, criterion_id, criterion_text, concept, criterion_type, status, explanation, data_found)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                result_id,
                eval_item.criterion_id,
                eval_item.criterion_text,
                eval_item.concept,
                eval_item.criterion_type if isinstance(eval_item.criterion_type, str) else eval_item.criterion_type.value,
                eval_item.status.value if hasattr(eval_item.status, "value") else eval_item.status,
                eval_item.explanation,
                eval_item.data_found,
            ),
        )

    return await _get_full_result(result_id, patient_row, protocol_row)


async def _get_full_result(result_id: int, patient_row: dict = None, protocol_row: dict = None) -> dict:
    result = await db.fetch_one("SELECT * FROM screening_results WHERE id = ?", (result_id,))
    if not result:
        return {}

    if not patient_row:
        patient_row = await db.fetch_one("SELECT * FROM patients WHERE id = ?", (result["patient_id"],)) or {}
    if not protocol_row:
        protocol_row = await db.fetch_one("SELECT * FROM protocols WHERE id = ?", (result["protocol_id"],)) or {}

    evals = await db.fetch_all(
        "SELECT * FROM criterion_evaluations WHERE result_id = ? ORDER BY id", (result_id,)
    )

    # Parse score_breakdown_json → score_breakdown dict
    score_breakdown = None
    raw_json = result.get("score_breakdown_json")
    if raw_json:
        try:
            score_breakdown = json.loads(raw_json)
        except Exception:
            pass

    result_dict = dict(result)
    result_dict.pop("score_breakdown_json", None)  # exclude raw JSON field from response

    return {
        **result_dict,
        "patient_name": patient_row.get("name", ""),
        "protocol_title": protocol_row.get("title", ""),
        "criterion_evaluations": [dict(e) for e in evals],
        "score_breakdown": score_breakdown,
    }


@router.post("/screen")
async def screen_patient(request: ScreenRequest):
    start = time.time()
    logger.info("=== SCREEN REQUEST: patient={} protocol={} ===", request.patient_id, request.protocol_id)
    result = await _run_screening(request.patient_id, request.protocol_id)
    logger.info("✓ Full screening pipeline done in {}ms", int((time.time() - start) * 1000))
    return result


@router.post("/screen/batch")
async def batch_screen(request: BatchScreenRequest):
    start = time.time()
    logger.info(
        "=== BATCH SCREEN: {} patients against protocol {} ===",
        len(request.patient_ids), request.protocol_id
    )
    results = []
    for i, pid in enumerate(request.patient_ids):
        try:
            logger.info("  Screening patient {} ({}/{})", pid, i + 1, len(request.patient_ids))
            # Skip Claude Sonnet rationale in batch mode — use fast fallback instead.
            # Full rationale can be fetched per-patient from the Results page.
            result = await _run_screening(pid, request.protocol_id, use_gpt_rationale=False)
            results.append(result)
        except HTTPException as e:
            logger.warning("Skipping patient {}: {}", pid, e.detail)
            results.append({"patient_id": pid, "error": e.detail})
        except Exception as e:
            logger.error("Unexpected error screening patient {}: {}", pid, e)
            results.append({"patient_id": pid, "error": str(e)})

    logger.info(
        "✓ Batch screening complete: {} results in {}ms",
        len(results), int((time.time() - start) * 1000)
    )
    return results


@router.get("/results")
async def list_results(protocol_id: int = None, verdict: str = None, skip: int = 0, limit: int = 50):
    conditions = []
    params = []
    if protocol_id:
        conditions.append("sr.protocol_id = ?")
        params.append(protocol_id)
    if verdict:
        conditions.append("sr.overall_verdict = ?")
        params.append(verdict)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT sr.*, p.name as patient_name, pr.title as protocol_title
        FROM screening_results sr
        LEFT JOIN patients p ON p.id = sr.patient_id
        LEFT JOIN protocols pr ON pr.id = sr.protocol_id
        {where}
        ORDER BY sr.created_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, skip])
    rows = await db.fetch_all(query, tuple(params))
    logger.info("Listed {} results", len(rows))
    return rows


@router.get("/results/stats/{protocol_id}")
async def get_stats(protocol_id: int):
    row = await db.fetch_one("SELECT id FROM protocols WHERE id = ?", (protocol_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Protocol not found")

    stats = await db.fetch_one(
        """SELECT
             COUNT(*) as total_screened,
             SUM(CASE WHEN overall_verdict='ELIGIBLE' THEN 1 ELSE 0 END) as eligible_count,
             SUM(CASE WHEN overall_verdict='INELIGIBLE' THEN 1 ELSE 0 END) as ineligible_count,
             SUM(CASE WHEN overall_verdict='REVIEW_NEEDED' THEN 1 ELSE 0 END) as review_needed_count,
             AVG(fit_score) as average_fit_score
           FROM screening_results WHERE protocol_id = ?""",
        (protocol_id,),
    )
    return ScreeningStats(
        protocol_id=protocol_id,
        total_screened=stats["total_screened"] or 0,
        eligible_count=stats["eligible_count"] or 0,
        ineligible_count=stats["ineligible_count"] or 0,
        review_needed_count=stats["review_needed_count"] or 0,
        average_fit_score=round(stats["average_fit_score"] or 0, 1),
    )


@router.get("/results/{result_id}")
async def get_result(result_id: int):
    result = await _get_full_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")
    return result


@router.put("/results/{result_id}/override")
async def override_result(result_id: int, request: OverrideRequest):
    row = await db.fetch_one("SELECT id FROM screening_results WHERE id = ?", (result_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")

    await db.execute(
        "UPDATE screening_results SET override_verdict=?, override_reason=? WHERE id=?",
        (request.override_verdict.value, request.override_reason, result_id),
    )
    logger.info(
        "Override applied to result {}: {} — {}",
        result_id, request.override_verdict.value, request.override_reason
    )
    return await _get_full_result(result_id)
