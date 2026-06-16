"""
Evaluation router — ground truth evaluation and accuracy reporting.
"""

import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from app import database as db
from app.services.evaluator import evaluator, GroundTruthPatient, REFERENCE_RULES
from app.services.report_generator import report_generator
from app.services.fhir_parser import fhir_parser

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class BuildGroundTruthRequest(BaseModel):
    protocol_id: int
    count: int = 100


class RunEvaluationRequest(BaseModel):
    protocol_id: int


async def _ensure_gt_columns():
    """Add ground truth columns if they don't exist yet (idempotent migration)."""
    migrations = [
        "ALTER TABLE patients ADD COLUMN is_ground_truth INTEGER DEFAULT 0",
        "ALTER TABLE patients ADD COLUMN ground_truth_verdict TEXT",
        "ALTER TABLE patients ADD COLUMN ground_truth_reason TEXT",
        "ALTER TABLE patients ADD COLUMN failure_mode TEXT",
        "ALTER TABLE patients ADD COLUMN ground_truth_protocol_id INTEGER",
    ]
    for sql in migrations:
        try:
            await db.execute(sql)
        except Exception:
            pass  # column already exists

    await db.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol_id INTEGER NOT NULL,
            run_date TEXT NOT NULL,
            patient_count INTEGER DEFAULT 0,
            sensitivity REAL,
            specificity REAL,
            ppv REAL,
            npv REAL,
            f1_score REAL,
            accuracy REAL,
            tp INTEGER DEFAULT 0,
            tn INTEGER DEFAULT 0,
            fp INTEGER DEFAULT 0,
            fn INTEGER DEFAULT 0,
            metrics_json TEXT,
            html_report TEXT,
            csv_data TEXT
        )
    """)


@router.post("/build-ground-truth")
async def build_ground_truth(request: BuildGroundTruthRequest):
    """Build the annotated ground truth patient set and save to DB."""
    await _ensure_gt_columns()

    # Check if protocol exists
    proto = await db.fetch_one("SELECT id, title FROM protocols WHERE id = ?", (request.protocol_id,))
    if not proto:
        raise HTTPException(status_code=404, detail=f"Protocol {request.protocol_id} not found")

    logger.info(
        "Building ground truth set for protocol {} '{}'",
        request.protocol_id, proto["title"]
    )

    # Delete existing ground truth patients — must remove child rows first to
    # satisfy the FK constraint: screening_results → patients, criterion_evaluations → screening_results
    gt_patient_ids = await db.fetch_all(
        "SELECT id FROM patients WHERE is_ground_truth = 1"
    )
    if gt_patient_ids:
        id_list = ",".join(str(r["id"]) for r in gt_patient_ids)
        result_ids = await db.fetch_all(
            f"SELECT id FROM screening_results WHERE patient_id IN ({id_list})"
        )
        if result_ids:
            rid_list = ",".join(str(r["id"]) for r in result_ids)
            await db.execute(
                f"DELETE FROM criterion_evaluations WHERE result_id IN ({rid_list})"
            )
        await db.execute(
            f"DELETE FROM screening_results WHERE patient_id IN ({id_list})"
        )
    await db.execute("DELETE FROM patients WHERE is_ground_truth = 1")

    # Load the protocol's REAL extracted rules so the ground truth is generated
    # relative to the flagship protocol's actual thresholds (Measure 2).
    rule_rows = await db.fetch_all(
        "SELECT * FROM criterion_rules WHERE protocol_id = ?", (request.protocol_id,)
    )
    flagship_rules = None
    if rule_rows:
        from app.models.protocol import CriterionRule, CriterionType
        flagship_rules = [
            CriterionRule(
                id=r["id"],
                protocol_id=r["protocol_id"],
                criterion_text=r["criterion_text"] or "",
                concept=r["concept"] or "",
                operator=r["operator"] or "presence",
                value=r["value"] or "",
                required=bool(r["required"]),
                criterion_type=CriterionType(r["criterion_type"]),
                snomed_code=r["snomed_code"],
                confidence=r["confidence"] or 0.0,
            )
            for r in rule_rows
        ]
    else:
        logger.warning(
            "Protocol {} has no criterion rules — ground truth will use reference thresholds",
            request.protocol_id,
        )

    # Generate patients (flagship-aware when rules are available)
    gt_patients = evaluator.build_ground_truth_set(request.protocol_id, rules=flagship_rules, count=request.count)

    eligible_count = ineligible_count = borderline_count = 0
    created = 0

    for gtp in gt_patients:
        parsed = fhir_parser.parse_bundle(gtp.fhir_bundle)
        fhir_str = json.dumps(gtp.fhir_bundle)
        pid = await db.execute(
            """INSERT INTO patients
               (patient_id, name, age, gender, fhir_json,
                is_ground_truth, ground_truth_verdict, ground_truth_reason,
                failure_mode, ground_truth_protocol_id)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                gtp.patient_id,
                parsed.name or gtp.patient_id,
                parsed.age,
                parsed.gender,
                fhir_str,
                1,
                gtp.ground_truth_verdict,
                gtp.ground_truth_reason,
                gtp.failure_mode,
                request.protocol_id,
            ),
        )
        created += 1
        if gtp.ground_truth_verdict == "ELIGIBLE":
            eligible_count += 1
        elif gtp.ground_truth_verdict == "INELIGIBLE":
            ineligible_count += 1
        else:
            borderline_count += 1

    logger.info(
        "✓ Ground truth set built: {} eligible, {} ineligible, {} borderline",
        eligible_count, ineligible_count, borderline_count
    )
    return {
        "patients_created": created,
        "eligible_count": eligible_count,
        "ineligible_count": ineligible_count,
        "borderline_count": borderline_count,
        "protocol_id": request.protocol_id,
    }


@router.post("/run")
async def run_evaluation(request: RunEvaluationRequest):
    """Run the full evaluation pipeline against ground truth patients."""
    await _ensure_gt_columns()

    proto = await db.fetch_one("SELECT id, title FROM protocols WHERE id = ?", (request.protocol_id,))
    if not proto:
        raise HTTPException(status_code=404, detail=f"Protocol {request.protocol_id} not found")

    # Fetch ground truth patients
    rows = await db.fetch_all(
        """SELECT patient_id, fhir_json, ground_truth_verdict, ground_truth_reason, failure_mode
           FROM patients
           WHERE is_ground_truth = 1 AND ground_truth_protocol_id = ?""",
        (request.protocol_id,),
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No ground truth patients found for this protocol. Run /build-ground-truth first.",
        )

    # Reconstruct GroundTruthPatient objects
    gt_patients = [
        GroundTruthPatient(
            patient_id=r["patient_id"],
            fhir_bundle=json.loads(r["fhir_json"]),
            ground_truth_verdict=r["ground_truth_verdict"],
            ground_truth_reason=r["ground_truth_reason"],
            failure_mode=r["failure_mode"],
        )
        for r in rows
    ]

    # Fetch protocol rules from DB (or use reference rules if none)
    rule_rows = await db.fetch_all(
        "SELECT * FROM criterion_rules WHERE protocol_id = ?", (request.protocol_id,)
    )

    if rule_rows:
        from app.models.protocol import CriterionRule, CriterionType
        rules = [
            CriterionRule(
                id=r["id"],
                protocol_id=r["protocol_id"],
                criterion_text=r["criterion_text"] or "",
                concept=r["concept"] or "",
                operator=r["operator"] or "presence",
                value=r["value"] or "",
                required=bool(r["required"]),
                criterion_type=CriterionType(r["criterion_type"]),
                snomed_code=r["snomed_code"],
                confidence=r["confidence"] or 0.0,
            )
            for r in rule_rows
        ]
    else:
        logger.warning("No criterion rules found for protocol {} — using reference T2DM rules", request.protocol_id)
        rules = REFERENCE_RULES

    logger.info("Running evaluation: {} patients × {} rules", len(gt_patients), len(rules))
    pairs = evaluator.run_evaluation(gt_patients, rules)
    metrics = evaluator.compute_metrics(pairs, protocol_title=proto.get("title", ""))

    # Generate reports
    html = report_generator.generate_html_report(
        metrics,
        protocol_title=proto["title"],
        eval_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    csv_data = evaluator.export_annotated_csv(gt_patients, pairs)

    metrics_json = metrics.model_dump_json()

    # Save to evaluation_runs
    await db.execute(
        """INSERT OR REPLACE INTO evaluation_runs
           (protocol_id, run_date, patient_count, sensitivity, specificity, ppv, npv,
            f1_score, accuracy, tp, tn, fp, fn, metrics_json, html_report, csv_data)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            request.protocol_id,
            datetime.utcnow().isoformat(),
            metrics.total_evaluated,
            metrics.sensitivity,
            metrics.specificity,
            metrics.ppv,
            metrics.npv,
            metrics.f1_score,
            metrics.accuracy,
            metrics.true_positives,
            metrics.true_negatives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics_json,
            html,
            csv_data,
        ),
    )

    return metrics.model_dump()


@router.get("/report/{protocol_id}")
async def get_report(protocol_id: int):
    """Return the latest AccuracyMetrics JSON for a protocol."""
    row = await db.fetch_one(
        "SELECT metrics_json FROM evaluation_runs WHERE protocol_id = ? ORDER BY id DESC LIMIT 1",
        (protocol_id,),
    )
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation run found for protocol {protocol_id}",
        )
    return json.loads(row["metrics_json"])


@router.get("/report/{protocol_id}/html", response_class=HTMLResponse)
async def get_html_report(protocol_id: int):
    """Return the full HTML accuracy report."""
    row = await db.fetch_one(
        "SELECT html_report FROM evaluation_runs WHERE protocol_id = ? ORDER BY id DESC LIMIT 1",
        (protocol_id,),
    )
    if not row or not row["html_report"]:
        raise HTTPException(status_code=404, detail="No HTML report found")
    return HTMLResponse(content=row["html_report"], media_type="text/html; charset=utf-8")


@router.get("/export/{protocol_id}/csv")
async def export_csv(protocol_id: int):
    """Return the 100-patient annotated CSV."""
    row = await db.fetch_one(
        "SELECT csv_data FROM evaluation_runs WHERE protocol_id = ? ORDER BY id DESC LIMIT 1",
        (protocol_id,),
    )
    if not row or not row["csv_data"]:
        raise HTTPException(status_code=404, detail="No CSV export found")

    def iterfile():
        yield row["csv_data"].encode()

    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="ground_truth_eval_{protocol_id}.csv"'
        },
    )
