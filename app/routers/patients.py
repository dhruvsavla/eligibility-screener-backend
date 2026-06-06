import json
from fastapi import APIRouter, HTTPException
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
        "SELECT id, patient_id, name, age, gender, created_at FROM patients ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, skip),
    )
    logger.info("Listed {} patients (skip={} limit={})", len(rows), skip, limit)
    return rows


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
    bundles = synthea_generator.generate_patients(request.count, seed=request.seed)

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

    logger.info("✓ Created {} patients in DB", len(created))
    return created


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


@router.delete("/{patient_id}")
async def delete_patient(patient_id: int):
    row = await db.fetch_one("SELECT id FROM patients WHERE id = ?", (patient_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")
    await db.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    logger.info("Deleted patient {}", patient_id)
    return {"detail": f"Patient {patient_id} deleted"}
