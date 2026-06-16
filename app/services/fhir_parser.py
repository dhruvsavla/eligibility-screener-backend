import json
from datetime import date, datetime
from loguru import logger
from app.models.patient import PatientData, LabResult

RELEVANT_LAB_PATTERNS = [
    "hba1c", "hemoglobin a1c", "glycated hemoglobin",
    "egfr", "glomerular filtration",
    "creatinine",
    "hemoglobin", "haemoglobin",
    "platelet",
    "alt", "alanine aminotransferase",
    "ast", "aspartate aminotransferase",
    "bilirubin",
    "wbc", "white blood cell", "leukocyte",
    # Additional labs from the synthetic patient generator
    "glucose",
    "cholesterol",
    "triglyceride",
    "low-density lipoprotein", "ldl",
    "high-density lipoprotein", "hdl",
    "body mass index", "bmi",
    "systolic blood pressure",
    "diastolic blood pressure",
    "heart rate",
    "sodium", "potassium",
    "uric acid",
    "international normalized ratio", "inr",
    "prothrombin",
    "thyroid stimulating hormone", "tsh",
    "brain natriuretic peptide", "bnp",
    "troponin",
    "ferritin", "iron",
]


def _calc_age(birth_date_str: str) -> int | None:
    try:
        bd = datetime.strptime(birth_date_str[:10], "%Y-%m-%d").date()
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except Exception:
        return None


KNOWN_LOINC_CODES = {
    "4548-4", "17856-6", "59261-8", "41995-2",       # HbA1c
    "33914-3", "62238-1", "98979-8", "48642-3", "48643-1",  # eGFR
    "2160-0", "38483-4", "14682-9", "59826-8",        # Creatinine
    "718-7", "59260-0", "20509-6",                    # Hemoglobin
    "6690-2", "49498-9", "26464-8",                   # WBC
    "777-3", "26515-7", "74775-1",                    # Platelets
    "1742-6", "1743-4", "76625-3",                    # ALT
    "1920-8", "1921-6",                               # AST
    "1975-2", "14629-0", "59828-4",                   # Bilirubin
    "2345-7", "14749-6", "2339-0",                    # Glucose
    "2951-2", "2947-0", "39791-9",                    # Sodium
    "2823-3", "6298-4", "39789-3",                    # Potassium
    "2093-3", "35200-5", "14647-2",                   # Cholesterol
    "13457-7", "18262-6", "2089-1",                   # LDL
    "2085-9", "14646-4",                              # HDL
    "2571-8", "12951-0",                              # Triglycerides
    "39156-5", "59574-4",                             # BMI
    "8480-6", "76534-7",                              # Systolic BP
    "8462-4", "76535-4",                              # Diastolic BP
}


def _is_relevant_lab(display: str) -> bool:
    d = display.lower()
    return any(pat in d for pat in RELEVANT_LAB_PATTERNS)


def _is_known_loinc(loinc_code: str | None) -> bool:
    return bool(loinc_code and loinc_code in KNOWN_LOINC_CODES)


class FHIRParser:
    def parse_bundle(self, fhir_json: dict) -> PatientData:
        if not fhir_json or not isinstance(fhir_json, dict):
            logger.warning("Invalid FHIR JSON — generating empty patient data")
            return PatientData(patient_id="unknown", name="Unknown Patient")

        entries = fhir_json.get("entry", [])
        if not entries:
            entries = []

        resources = []
        for entry in entries:
            res = entry.get("resource", entry) if isinstance(entry, dict) else {}
            resources.append(res)

        patient_id = "unknown"
        name = "Unknown"
        age = None
        gender = None
        conditions = []
        medications = []
        lab_results = []
        allergies = []

        for res in resources:
            rtype = res.get("resourceType", "")

            if rtype == "Patient":
                patient_id = res.get("id", "unknown")
                logger.info("Parsing FHIR bundle for patient {}", patient_id)
                gender = res.get("gender", None)
                birth_date = res.get("birthDate", None)
                if birth_date:
                    age = _calc_age(birth_date)
                names = res.get("name", [])
                if names:
                    n = names[0]
                    given = " ".join(n.get("given", []))
                    family = n.get("family", "")
                    name = f"{given} {family}".strip() or "Unknown"

            elif rtype == "Condition":
                status_obj = res.get("clinicalStatus", {})
                status_codings = status_obj.get("coding", [{}])
                status_code = status_codings[0].get("code", "active") if status_codings else "active"
                if status_code in ("inactive", "resolved", "remission"):
                    continue
                code_obj = res.get("code", {})
                display = (
                    code_obj.get("text")
                    or (code_obj.get("coding", [{}])[0].get("display") if code_obj.get("coding") else None)
                    or "Unknown condition"
                )
                if display:
                    conditions.append(display)

            elif rtype == "MedicationRequest":
                med_obj = res.get("medicationCodeableConcept") or res.get("medication", {})
                if isinstance(med_obj, dict):
                    display = (
                        med_obj.get("text")
                        or (med_obj.get("coding", [{}])[0].get("display") if med_obj.get("coding") else None)
                    )
                    if display:
                        medications.append(display)

            elif rtype == "Observation":
                code_obj = res.get("code", {})
                display = (
                    code_obj.get("text")
                    or (code_obj.get("coding", [{}])[0].get("display") if code_obj.get("coding") else None)
                    or ""
                )
                # Extract LOINC code from coding array
                loinc_code = None
                for coding in code_obj.get("coding", []):
                    if "loinc.org" in coding.get("system", ""):
                        loinc_code = coding.get("code")
                        break
                if not _is_relevant_lab(display) and not _is_known_loinc(loinc_code):
                    continue
                vq = res.get("valueQuantity", {})
                if vq and "value" in vq:
                    try:
                        lab_results.append(
                            LabResult(
                                name=display,
                                value=float(vq["value"]),
                                unit=vq.get("unit", ""),
                                date=res.get("effectiveDateTime", "")[:10] if res.get("effectiveDateTime") else "",
                                loinc_code=loinc_code,
                            )
                        )
                    except Exception:
                        pass

            elif rtype == "AllergyIntolerance":
                substance = res.get("code", {})
                display = (
                    substance.get("text")
                    or (substance.get("coding", [{}])[0].get("display") if substance.get("coding") else None)
                    or ""
                )
                if display:
                    allergies.append(display)

        logger.info("  Found {} conditions: {}", len(conditions), conditions[:5])
        logger.info("  Found {} medications: {}", len(medications), medications[:5])
        logger.info(
            "  Found {} relevant lab results: {}",
            len(lab_results),
            [f"{l.name}={l.value}{l.unit}" for l in lab_results],
        )

        return PatientData(
            patient_id=patient_id,
            name=name,
            age=age,
            gender=gender,
            conditions=conditions,
            medications=medications,
            lab_results=lab_results,
            allergies=allergies,
        )


fhir_parser = FHIRParser()
