import json
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from loguru import logger
from app import database as db
from app.models.patient import GeneratePatientsRequest, PatientData
from app.services.synthea_generator import synthea_generator
from app.services.fhir_parser import fhir_parser

router = APIRouter(prefix="/api/patients", tags=["patients"])


async def _parse_patient_from_db(row: dict) -> dict:
    fhir_json_str = row.get("fhir_json", "{}")
    try:
        fhir_bundle = json.loads(fhir_json_str) if fhir_json_str else {}
        parsed = fhir_parser.parse_bundle(fhir_bundle)
    except Exception:
        parsed = PatientData(
            patient_id=row.get("patient_id", ""),
            name=row.get("name", ""),
            age=row.get("age"),
            gender=None,
        )

    return {
        **row,
        "conditions": parsed.conditions,
        "medications": parsed.medications,
        "lab_results": [l.model_dump() for l in parsed.lab_results],
    }


@router.get("")
async def list_patients(skip: int = 0, limit: int = 20):
    rows = await db.fetch_all(
        """SELECT id, patient_id, name, age, gender, fhir_json, created_at,
                  is_ground_truth, ground_truth_verdict
           FROM patients
           ORDER BY created_at DESC LIMIT ? OFFSET ?""",
        (limit, skip),
    )
    logger.info("Listed {} patients (skip={} limit={})", len(rows), skip, limit)
    return [await _parse_patient_from_db(row) for row in rows]


@router.get("/count")
async def count_patients():
    row = await db.fetch_one("SELECT COUNT(*) as total FROM patients")
    return {"total": row["total"] if row else 0}


@router.get("/{patient_id}")
async def get_patient(patient_id: int):
    row = await db.fetch_one("SELECT * FROM patients WHERE id = ?", (patient_id,))
    if not row:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return await _parse_patient_from_db(row)


@router.post("/generate")
async def generate_patients(request: GeneratePatientsRequest):
    logger.info("Generating {} synthetic patients...", request.count)

    # Try real Synthea first
    source = "python_fallback"
    synthea_available = False
    bundles = []

    try:
        from app.services.synthea_runner import synthea_runner, SyntheaNotAvailableError, SyntheaGenerationError
        prereqs = synthea_runner.check_prerequisites()
        synthea_available = prereqs["can_run"]

        if synthea_available:
            logger.info("Using real MITRE Synthea for patient generation")
            bundles = synthea_runner.generate(count=request.count, seed=request.seed)
            source = "synthea"
        else:
            logger.warning("Synthea not available — using Python fallback generator")
            logger.warning("  Reason: {}", prereqs["reason"])
            logger.warning("  To enable real Synthea: see backend/synthea/SETUP.md")
    except (SyntheaNotAvailableError, SyntheaGenerationError) as e:
        logger.warning("Synthea failed: {} — falling back to Python generator", e)
    except Exception as e:
        logger.warning("Synthea check error: {} — using Python fallback", e)

    if not bundles:
        if source != "python_fallback":
            logger.warning(
                "Patient data will be functional but less clinically detailed than real Synthea"
            )
        bundles = synthea_generator.generate_patients(request.count, seed=request.seed)
        source = "python_fallback"

    created = []
    for bundle in bundles:
        try:
            parsed = fhir_parser.parse_bundle(bundle)
            fhir_str = json.dumps(bundle)

            existing = await db.fetch_one(
                "SELECT id FROM patients WHERE patient_id = ?", (parsed.patient_id,)
            )
            if existing:
                continue

            pid = await db.execute(
                "INSERT INTO patients (patient_id, name, age, gender, fhir_json) VALUES (?,?,?,?,?)",
                (parsed.patient_id, parsed.name, parsed.age, parsed.gender, fhir_str),
            )
            created.append(
                {
                    "id": pid,
                    "patient_id": parsed.patient_id,
                    "name": parsed.name,
                    "age": parsed.age,
                    "gender": parsed.gender,
                    "conditions": parsed.conditions,
                    "medications": parsed.medications,
                }
            )
        except Exception as e:
            logger.error("Failed to save synthetic patient: {}", e)

    logger.info("✓ Created {} patients in DB (source={})", len(created), source)
    return {
        "patients": created,
        "source": source,
        "synthea_available": synthea_available,
        "count": len(created),
    }


@router.post("/generate-500")
async def generate_500_patients():
    """Generate the canonical 500 synthetic patients via the real MITRE Synthea tool.

    This endpoint REQUIRES real Synthea — it does NOT fall back to the Python
    generator. If Synthea/Java is unavailable it returns HTTP 503 with setup
    instructions, per the spec (the 500-patient cohort must be real Synthea output).
    """
    from app.services.synthea_runner import (
        synthea_runner,
        SyntheaNotAvailableError,
        SyntheaGenerationError,
    )

    prereqs = synthea_runner.check_prerequisites()
    if not prereqs["can_run"]:
        logger.warning("generate-500 requested but Synthea unavailable: {}", prereqs["reason"])
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Synthea not available",
                "reason": prereqs["reason"],
                "setup_guide": "backend/synthea/SETUP.md",
                "instructions": synthea_runner.get_setup_instructions(),
            },
        )

    logger.info("=== GENERATING CANONICAL 500 PATIENTS VIA SYNTHEA ===")
    try:
        bundles = synthea_runner.generate(count=500, seed=42, module="diabetes*")
    except (SyntheaNotAvailableError, SyntheaGenerationError) as e:
        logger.error("Synthea generation failed: {}", e)
        raise HTTPException(status_code=503, detail={"error": "Synthea generation failed", "reason": str(e)})

    generated = 0
    failed: list[dict] = []
    for i, bundle in enumerate(bundles):
        try:
            parsed = fhir_parser.parse_bundle(bundle)
            existing = await db.fetch_one(
                "SELECT id FROM patients WHERE patient_id = ?", (parsed.patient_id,)
            )
            if existing:
                continue
            await db.execute(
                "INSERT INTO patients (patient_id, name, age, gender, fhir_json) VALUES (?,?,?,?,?)",
                (parsed.patient_id, parsed.name, parsed.age, parsed.gender, json.dumps(bundle)),
            )
            generated += 1
            if generated % 50 == 0:
                logger.info("[generate-500] inserted {} patients...", generated)
        except Exception as e:
            failed.append({"index": i, "error": str(e)})
            logger.error("Failed to save Synthea patient #{}: {}", i, e)

    logger.info("✓ generate-500 complete: {} inserted, {} failed", generated, len(failed))
    return {"generated": generated, "source": "synthea", "failed": failed}


@router.post("/upload")
async def upload_patient(bundle: dict):
    logger.info("Uploading FHIR bundle...")
    try:
        parsed = fhir_parser.parse_bundle(bundle)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FHIR bundle: {e}")

    fhir_str = json.dumps(bundle)
    existing = await db.fetch_one(
        "SELECT id FROM patients WHERE patient_id = ?", (parsed.patient_id,)
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"Patient {parsed.patient_id} already exists")

    pid = await db.execute(
        "INSERT INTO patients (patient_id, name, age, gender, fhir_json) VALUES (?,?,?,?,?)",
        (parsed.patient_id, parsed.name, parsed.age, parsed.gender, fhir_str),
    )
    logger.info("✓ Uploaded patient {} (DB id={})", parsed.patient_id, pid)
    return {"id": pid, "patient_id": parsed.patient_id, "name": parsed.name}


@router.post("/upload-bulk")
async def upload_patients_bulk(files: List[UploadFile] = File(...)):
    """Upload a folder of FHIR R4 bundle JSON files. Returns import summary."""
    imported, skipped, failed = 0, 0, 0
    errors: list[dict] = []

    for uf in files:
        fname = uf.filename or "unknown"
        if not fname.lower().endswith(".json"):
            skipped += 1
            continue
        try:
            raw = await uf.read()
            bundle = json.loads(raw)
        except Exception as e:
            failed += 1
            errors.append({"file": fname, "error": f"Invalid JSON: {e}"})
            continue

        try:
            parsed = fhir_parser.parse_bundle(bundle)
        except Exception as e:
            failed += 1
            errors.append({"file": fname, "error": f"Invalid FHIR bundle: {e}"})
            continue

        existing = await db.fetch_one(
            "SELECT id FROM patients WHERE patient_id = ?", (parsed.patient_id,)
        )
        if existing:
            skipped += 1
            continue

        try:
            await db.execute(
                "INSERT INTO patients (patient_id, name, age, gender, fhir_json) VALUES (?,?,?,?,?)",
                (parsed.patient_id, parsed.name, parsed.age, parsed.gender, json.dumps(bundle)),
            )
            imported += 1
        except Exception as e:
            failed += 1
            errors.append({"file": fname, "error": f"DB error: {e}"})

    logger.info("Bulk upload: imported={} skipped={} failed={}", imported, skipped, failed)
    return {"imported": imported, "skipped": skipped, "failed": failed, "errors": errors}


@router.delete("")
async def delete_all_patients():
    result = await db.fetch_one("SELECT COUNT(*) as total FROM patients")
    total = result["total"] if result else 0
    await db.execute("DELETE FROM screening_results")
    await db.execute("DELETE FROM patients")
    logger.info("Deleted all {} patients", total)
    return {"detail": f"Deleted {total} patients"}


@router.delete("/{patient_id}")
async def delete_patient(patient_id: int):
    row = await db.fetch_one("SELECT id FROM patients WHERE id = ?", (patient_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")
    await db.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    logger.info("Deleted patient {}", patient_id)
    return {"detail": f"Patient {patient_id} deleted"}
