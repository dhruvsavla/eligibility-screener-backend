"""
GroundTruthEvaluator — builds and evaluates a 100-patient annotated
ground truth set for measuring screening accuracy.

Ground truth generation strategy:
  For each patient in the set, we know the CORRECT verdict because we designed
  the patient to either definitively match or definitively not match the
  reference T2DM protocol criteria:

    INCLUSION:
      - Age 18-75
      - HbA1c >= 7.5% AND <= 11.0%
      - Diagnosis: Type 2 diabetes mellitus
      - eGFR >= 60 mL/min
    EXCLUSION:
      - Current insulin use
      - eGFR < 45 (CKD stage 3b+)
      - Active malignancy
      - Pregnancy

  Ground truth split:
    TRUE_ELIGIBLE  (40 patients): satisfy all inclusion, trigger no exclusion
    TRUE_INELIGIBLE (40 patients): violate at least one hard criterion
    BORDERLINE (20 patients): many criteria unverifiable → REVIEW_NEEDED
"""

import io
import json
import math
import random
import uuid
from datetime import date, timedelta
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel

from app.models.patient import PatientData
from app.models.protocol import CriterionRule, CriterionType
from app.models.screening import EvaluationStatus, VerdictType
from app.services.scoring_engine import ScoringEngine


# ---------------------------------------------------------------------------
# Reference protocol rules (hardcoded T2DM trial)
# ---------------------------------------------------------------------------

def _make_rule(cid: int, concept: str, operator: str, value: str, ctype: str) -> CriterionRule:
    return CriterionRule(
        id=cid,
        protocol_id=0,
        criterion_text=f"{concept} {operator} {value}".strip(),
        concept=concept,
        operator=operator,
        value=value,
        required=True,
        criterion_type=CriterionType(ctype),
        snomed_code=None,
        confidence=1.0,
    )


REFERENCE_RULES: list[CriterionRule] = [
    _make_rule(1, "Age",                    "between",  "18 75",  "inclusion"),
    _make_rule(2, "HbA1c",                  "between",  "7.5 11.0", "inclusion"),
    _make_rule(3, "Type 2 diabetes mellitus","presence", "",       "inclusion"),
    _make_rule(4, "eGFR",                   ">=",       "60",     "inclusion"),
    _make_rule(5, "Insulin",                "presence", "",       "exclusion"),
    _make_rule(6, "eGFR",                   "<",        "45",     "exclusion"),
    _make_rule(7, "Malignant neoplasm cancer","presence","",       "exclusion"),
    _make_rule(8, "Pregnancy",              "presence", "",       "exclusion"),
]


# ---------------------------------------------------------------------------
# Flagship-aware threshold derivation
#
# The 100-patient ground truth is generated relative to the FLAGSHIP protocol's
# ACTUAL extracted thresholds (read from its criterion_rules) rather than the
# hardcoded reference values. This makes the sensitivity number defensible:
# "85% on the real flagship protocol". If a recognizable threshold is missing,
# we fall back to the reference value and log a WARNING.
# ---------------------------------------------------------------------------

import re as _re

DEFAULT_THRESHOLDS = {
    "age_lo": 18.0, "age_hi": 75.0,
    "hba1c_lo": 7.5, "hba1c_hi": 11.0,
    "egfr_incl_min": 60.0, "egfr_excl_max": 45.0,
    "derived": False,
}


def _nums(s: str) -> list[float]:
    return [float(x) for x in _re.findall(r"\d+\.?\d*", str(s or ""))]


def derive_thresholds(rules: list[CriterionRule]) -> dict:
    """Read age / HbA1c / eGFR thresholds from a protocol's real extracted rules.

    Falls back to DEFAULT_THRESHOLDS values for any threshold not recognizable,
    and logs a WARNING when nothing usable is found.
    """
    thr = dict(DEFAULT_THRESHOLDS)
    found_any = False

    for r in rules:
        concept = (r.concept or "").lower()
        op = (r.operator or "").lower()
        ctype = r.criterion_type.value if hasattr(r.criterion_type, "value") else str(r.criterion_type)
        nums = _nums(r.value)

        if "age" in concept:
            if op in ("between", "range") and len(nums) >= 2:
                thr["age_lo"], thr["age_hi"] = nums[0], nums[1]; found_any = True
            elif op in (">=", ">") and nums:
                thr["age_lo"] = nums[0]; found_any = True
            elif op in ("<=", "<") and nums:
                thr["age_hi"] = nums[0]; found_any = True

        elif "hba1c" in concept or "a1c" in concept or "glycated" in concept:
            if op in ("between", "range") and len(nums) >= 2:
                thr["hba1c_lo"], thr["hba1c_hi"] = nums[0], nums[1]; found_any = True
            elif op in (">=", ">") and nums:
                thr["hba1c_lo"] = nums[0]; found_any = True
            elif op in ("<=", "<") and nums:
                thr["hba1c_hi"] = nums[0]; found_any = True

        elif "egfr" in concept or "glomerular" in concept or "gfr" in concept:
            if "inclu" in ctype and op in (">=", ">") and nums:
                thr["egfr_incl_min"] = nums[0]; found_any = True
            elif "exclu" in ctype and op in ("<=", "<") and nums:
                thr["egfr_excl_max"] = nums[0]; found_any = True
            elif op in (">=", ">") and nums:
                thr["egfr_incl_min"] = nums[0]; found_any = True
            elif op in ("<=", "<") and nums:
                thr["egfr_excl_max"] = nums[0]; found_any = True

    thr["derived"] = found_any
    # Track whether the protocol has a HbA1c LOWER bound (>= x) so that the
    # ground truth can generate the correct failure direction.  REWIND only
    # has an UPPER bound (<=9.5), so the failure mode is "too high", not "too low".
    thr["hba1c_has_lower_bound"] = False
    for r in rules:
        concept = (r.concept or "").lower()
        op = (r.operator or "").lower()
        if ("hba1c" in concept or "a1c" in concept or "glycated" in concept):
            if op in ("between", "range", ">=", ">"):
                thr["hba1c_has_lower_bound"] = True
                break

    if found_any:
        logger.info(
            "Ground truth thresholds derived from flagship rules: "
            "age [{:.0f}-{:.0f}], HbA1c [{}-{}], eGFR incl≥{} excl<{} "
            "(hba1c_lower_bound={})",
            thr["age_lo"], thr["age_hi"], thr["hba1c_lo"], thr["hba1c_hi"],
            thr["egfr_incl_min"], thr["egfr_excl_max"],
            thr["hba1c_has_lower_bound"],
        )
    else:
        logger.warning(
            "No recognizable age/HbA1c/eGFR thresholds in flagship rules — "
            "ground truth is using REFERENCE values, not flagship-derived values"
        )
    return thr


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class GroundTruthPatient(BaseModel):
    patient_id: str
    fhir_bundle: dict
    ground_truth_verdict: str   # ELIGIBLE | INELIGIBLE | REVIEW_NEEDED
    ground_truth_reason: str
    failure_mode: Optional[str] = None   # insulin_use | low_egfr | low_hba1c | malignancy | None
    annotator: str = "automated_design"


class EvaluationPair(BaseModel):
    patient_id: str
    ground_truth_verdict: str
    predicted_verdict: str
    predicted_score: int
    confidence_low: int
    confidence_high: int
    failure_mode: Optional[str] = None
    criterion_pass_count: int = 0
    criterion_fail_count: int = 0
    criterion_ambiguous_count: int = 0
    correct: bool = False
    age: Optional[int] = None
    hba1c: Optional[float] = None
    egfr: Optional[float] = None
    conditions: list[str] = []
    medications: list[str] = []


class AccuracyMetrics(BaseModel):
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1_score: float
    accuracy: float

    true_positives: int
    true_negatives: int
    false_positives: int        # total FPs (hard + soft) for backward compat
    false_positives_hard: int   # ELIGIBLE predicted, truly ineligible — coordinator won't catch
    false_positives_soft: int   # REVIEW_NEEDED predicted, truly ineligible — coordinator catches
    false_negatives: int        # INELIGIBLE predicted, truly eligible — critical misses

    failure_mode_accuracy: dict[str, Any]   # {mode: {total, correct, accuracy}}

    borderline_review_rate: float
    mean_confidence_width: float
    confidence_coverage: float

    meets_sensitivity_target: bool
    target_sensitivity: float = 0.85

    total_evaluated: int        # kept for backward compat
    total_patients: int         # alias for total_evaluated
    eligible_count: int
    ineligible_count: int
    borderline_count: int
    protocol_title: str = ""


# ---------------------------------------------------------------------------
# FHIR helpers (inline — avoids circular imports from synthea_generator)
# ---------------------------------------------------------------------------

def _obs(pid: str, display: str, loinc: str, value: float, unit: str, days_ago: int = 30) -> dict:
    obs_date = (date.today() - timedelta(days=days_ago)).isoformat()
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
        "code": {
            "coding": [{"system": "http://loinc.org", "code": loinc, "display": display}],
            "text": display,
        },
        "subject": {"reference": f"Patient/{pid}"},
        "effectiveDateTime": obs_date + "T08:00:00Z",
        "valueQuantity": {"value": round(value, 2), "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
    }


def _condition(pid: str, display: str, snomed: str, icd10: str) -> dict:
    return {
        "resourceType": "Condition",
        "id": str(uuid.uuid4()),
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
        "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
        "code": {
            "coding": [
                {"system": "http://snomed.info/sct", "code": snomed, "display": display},
                {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": icd10},
            ],
            "text": display,
        },
        "subject": {"reference": f"Patient/{pid}"},
        "onsetDateTime": (date.today() - timedelta(days=365 * 5)).isoformat(),
    }


def _medication(pid: str, display: str, rxnorm: str) -> dict:
    return {
        "resourceType": "MedicationRequest",
        "id": str(uuid.uuid4()),
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": rxnorm, "display": display}],
            "text": display,
        },
        "subject": {"reference": f"Patient/{pid}"},
        "authoredOn": (date.today() - timedelta(days=365)).isoformat(),
    }


def _patient_resource(pid: str, age: int, gender: str) -> dict:
    birth_year = date.today().year - age
    return {
        "resourceType": "Patient",
        "id": pid,
        "name": [{"use": "official", "family": "GroundTruth", "given": [pid]}],
        "gender": gender,
        "birthDate": f"{birth_year}-06-15",
    }


def _bundle(resources: list[dict]) -> dict:
    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": date.today().isoformat() + "T00:00:00Z",
        "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in resources],
    }


# ---------------------------------------------------------------------------
# Patient builders
# ---------------------------------------------------------------------------

def _build_eligible_patient(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age_lo = int(max(18, thr["age_lo"] + 1))
    age_hi = int(max(age_lo + 1, thr["age_hi"] - 1))
    age = rng.randint(age_lo, age_hi)
    gender = rng.choice(["male", "female"])
    # comfortably inside the inclusion HbA1c band
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, max(thr["hba1c_lo"] + 0.2, thr["hba1c_hi"] - 0.1)), 1)
    # comfortably above the eGFR inclusion minimum
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 2, thr["egfr_incl_min"] + 58), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _condition(pid, "Essential hypertension", "59621000", "I10"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
        _obs(pid, "Creatinine [Mass/volume] in Serum or Plasma", "2160-0", round(rng.uniform(0.7, 1.1), 2), "mg/dL"),
    ]
    logger.debug("  Patient {}: Age {}, HbA1c {}%, eGFR {} → ELIGIBLE (all criteria met)", pid, age, hba1c, int(egfr))
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="ELIGIBLE",
        ground_truth_reason=(
            f"Age {age} in [{int(thr['age_lo'])}-{int(thr['age_hi'])}], "
            f"HbA1c {hba1c}% in [{thr['hba1c_lo']}-{thr['hba1c_hi']}], "
            f"eGFR {int(egfr)} ≥ {int(thr['egfr_incl_min'])} — all inclusion met, no exclusions triggered"
        ),
        failure_mode=None,
    )


def _build_ineligible_insulin(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _medication(pid, "Insulin glargine 100 units/mL", "274783"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode A (insulin) — Age {}, insulin + T2DM → INELIGIBLE", pid, age)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason="Current insulin use — exclusion criterion triggered (insulin presence)",
        failure_mode="insulin_use",
    )


def _build_ineligible_egfr(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 5, max(int(thr["age_lo"]) + 6, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
    # below the exclusion eGFR ceiling → triggers exclusion AND fails inclusion min
    egfr = round(rng.uniform(20, max(21, thr["egfr_excl_max"] - 1)), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _condition(pid, "Chronic kidney disease stage 3", "433144002", "N18.3"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode B (eGFR) — Age {}, eGFR {} → INELIGIBLE", pid, age, int(egfr))
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason=(
            f"eGFR {int(egfr)} < {int(thr['egfr_excl_max'])} — fails inclusion "
            f"eGFR≥{int(thr['egfr_incl_min'])} and triggers exclusion eGFR<{int(thr['egfr_excl_max'])}"
        ),
        failure_mode="low_egfr",
    )


def _build_ineligible_hba1c(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    # below the inclusion HbA1c floor → fails inclusion (too well controlled)
    hba1c = round(rng.uniform(max(4.5, thr["hba1c_lo"] - 2.5), thr["hba1c_lo"] - 0.1), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode C (HbA1c) — Age {}, HbA1c {}% < {} → INELIGIBLE", pid, age, hba1c, thr["hba1c_lo"])
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason=f"HbA1c {hba1c}% < {thr['hba1c_lo']}% — fails inclusion threshold (too well controlled)",
        failure_mode="low_hba1c",
    )


def _build_ineligible_malignancy(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _condition(pid, "Malignant neoplasm cancer", "363346000", "C80.1"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode D (malignancy) — Age {}, cancer → INELIGIBLE", pid, age)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason="Active malignancy — exclusion criterion triggered",
        failure_mode="malignancy",
    )


def _build_borderline_patient(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    variant = rng.choice(["missing_labs", "boundary_hba1c", "boundary_age", "ckd_borderline"])
    mid_age = int((thr["age_lo"] + thr["age_hi"]) / 2)

    if variant == "missing_labs":
        age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
        # Only T2DM diagnosis, no labs → many AMBIGUOUS criteria → REVIEW_NEEDED
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        ]
        reason = f"Age {age}, T2DM confirmed — HbA1c and eGFR labs missing; unverifiable"
    elif variant == "boundary_hba1c":
        age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, mid_age))
        hba1c = thr["hba1c_lo"]  # exact boundary
        egfr = round(rng.uniform(thr["egfr_incl_min"] + 2, thr["egfr_incl_min"] + 30), 0)
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
            _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
            _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
        ]
        reason = f"HbA1c exactly {thr['hba1c_lo']}% (boundary), eGFR {int(egfr)} — borderline eligibility"
    elif variant == "boundary_age":
        hi = int(thr["age_hi"])
        age = rng.choice([hi - 1, hi, hi + 1])
        hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
        egfr = round(rng.uniform(thr["egfr_incl_min"] + 2, thr["egfr_incl_min"] + 30), 0)
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
            _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
            _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
        ]
        reason = f"Age {age} near age limit {hi} — borderline eligibility"
    else:  # ckd_borderline
        age = rng.randint(mid_age, max(mid_age + 1, int(thr["age_hi"]) - 5))
        hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
        # eGFR between exclusion ceiling and inclusion floor: fails ≥min inclusion but not <max exclusion
        lo_e = thr["egfr_excl_max"] + 1
        hi_e = max(lo_e + 1, thr["egfr_incl_min"] - 1)
        egfr = round(rng.uniform(lo_e, hi_e), 0)
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _condition(pid, "Chronic kidney disease stage 3", "433144002", "N18.3"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
            _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
            _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
        ]
        reason = (
            f"eGFR {int(egfr)} — borderline (below ≥{int(thr['egfr_incl_min'])} "
            f"inclusion but above <{int(thr['egfr_excl_max'])} exclusion)"
        )

    logger.debug("  Patient {} BORDERLINE ({}) → REVIEW_NEEDED", pid, variant)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="REVIEW_NEEDED",
        ground_truth_reason=reason,
        failure_mode=None,
    )


# ---------------------------------------------------------------------------
# Additional patient builders for failure modes present in real-world protocols
# (e.g. REWIND has no insulin/malignancy exclusion but does have uncontrolled
#  diabetes, pancreatitis, pregnancy, and a HbA1c upper-bound inclusion).
# ---------------------------------------------------------------------------

def _build_ineligible_high_hba1c(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    """Patient with HbA1c ABOVE the protocol's maximum threshold."""
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(thr["hba1c_hi"] + 0.5, thr["hba1c_hi"] + 3.0), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode HIGH_HBA1C — Age {}, HbA1c {}% > {} → INELIGIBLE",
                 pid, age, hba1c, thr["hba1c_hi"])
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason=(
            f"HbA1c {hba1c}% > {thr['hba1c_hi']}% — fails inclusion upper threshold "
            f"(too poorly controlled / uncontrolled diabetes)"
        ),
        failure_mode="high_hba1c",
    )


def _build_ineligible_uncontrolled_diabetes(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    """Patient with 'Uncontrolled type 2 diabetes' that triggers the absence exclusion."""
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        # This is what triggers "uncontrolled_diabetes absence" exclusion
        _condition(pid, "Uncontrolled type 2 diabetes mellitus", "44054006", "E11.65"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode UNCONTROLLED_DM — uncontrolled diabetes → INELIGIBLE", pid)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason="Uncontrolled diabetes — triggers exclusion criterion",
        failure_mode="uncontrolled_diabetes",
    )


def _build_ineligible_pancreatitis(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    """Patient with pancreatitis/organ disorder that triggers the compound exclusion."""
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _condition(pid, "Pancreatitis", "75694006", "K85.9"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode PANCREATITIS — organ disorder → INELIGIBLE", pid)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason="Pancreatitis — triggers organ disorder exclusion criterion",
        failure_mode="pancreatitis",
    )


def _build_ineligible_pregnancy(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    """Patient with current pregnancy that triggers the pregnancy exclusion."""
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, min(45, int(thr["age_hi"]) - 5))
    gender = "female"
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _condition(pid, "Pregnancy", "77386006", "Z34.90"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode PREGNANCY — current pregnancy → INELIGIBLE", pid)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason="Current pregnancy — triggers pregnancy exclusion criterion",
        failure_mode="pregnancy",
    )


def _build_ineligible_severe_hypoglycemia(index: int, rng: random.Random, thr: dict) -> GroundTruthPatient:
    """Patient with severe hypoglycemia history that triggers the hypoglycemia exclusion."""
    pid = f"GT-{index:03d}"
    age = rng.randint(int(thr["age_lo"]) + 2, max(int(thr["age_lo"]) + 3, int(thr["age_hi"]) - 5))
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(thr["hba1c_lo"] + 0.1, thr["hba1c_hi"] - 0.5), 1)
    egfr = round(rng.uniform(thr["egfr_incl_min"] + 5, thr["egfr_incl_min"] + 40), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _condition(pid, "Severe hypoglycemia", "421437000", "E11.641"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode SEVERE_HYPOGLYCEMIA — history of severe hypo → INELIGIBLE", pid)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason="Severe hypoglycemia in past year — triggers hypoglycemia exclusion criterion",
        failure_mode="severe_hypoglycemia",
    )


# ---------------------------------------------------------------------------
# Failure mode discovery — maps a protocol's REAL rules to testable modes
# ---------------------------------------------------------------------------

#: Failure modes that exist in REFERENCE_RULES (the hardcoded diabetes trial).
REFERENCE_FAILURE_MODES = ["insulin_use", "low_egfr", "low_hba1c", "malignancy"]


def discover_failure_modes(rules: list[CriterionRule]) -> list[str]:
    """
    Inspect a protocol's real extracted rules and return the ordered list of
    failure mode names that have testable ground-truth patients in this module.

    Only modes that correspond to an actual rule in *this* protocol are included.
    If fewer than 2 testable modes are found the caller should fall back to
    REFERENCE_RULES / REFERENCE_FAILURE_MODES.
    """
    modes: list[str] = []
    seen: set[str] = set()

    def _add(m: str):
        if m not in seen:
            modes.append(m)
            seen.add(m)

    for r in rules:
        ctype = r.criterion_type.value if hasattr(r.criterion_type, "value") else str(r.criterion_type)
        concept = (r.concept or "").lower().replace("_", " ")
        op = (r.operator or "").lower()

        if "inclu" in ctype:
            # HbA1c inclusion rules — determine the direction of the failure
            if any(k in concept for k in ("hba1c", "a1c", "glycated")):
                if op in ("<=", "<"):          # upper-bound only (e.g. REWIND ≤9.5)
                    _add("high_hba1c")
                elif op in (">=", ">"):        # lower-bound (poorly-controlled required)
                    _add("low_hba1c")
                elif op in ("between", "range"):
                    _add("low_hba1c")          # below the floor
                    _add("high_hba1c")         # above the ceiling
            # eGFR inclusion (low eGFR → fails inclusion min)
            if any(k in concept for k in ("egfr", "glomerular", "gfr")):
                _add("low_egfr")

        elif "exclu" in ctype:
            if any(k in concept for k in ("insulin",)):
                _add("insulin_use")
            if any(k in concept for k in ("malign", "cancer", "neoplasm", "tumor")):
                _add("malignancy")
            # For eGFR/kidney exclusion require specific eGFR terms — do NOT trigger
            # on compound concepts like "pancreatitis_hepatic_renal_thyroid_disorder"
            # that merely mention "renal" among other organs.
            if any(k in concept for k in ("egfr", "glomerular", "creatinine")):
                _add("low_egfr")
            elif "kidney" in concept and "pancreatit" not in concept:
                _add("low_egfr")
            # Compound concept names: pancreatitis_hepatic_renal_thyroid_disorder
            if "pancreatit" in concept:
                _add("pancreatitis")
            if any(k in concept for k in ("pregnan",)):
                _add("pregnancy")
            if any(k in concept for k in ("hypoglycemia", "hypoglycaemia")):
                _add("severe_hypoglycemia")
            if any(k in concept for k in ("uncontrol",)):
                _add("uncontrolled_diabetes")

    if modes:
        logger.info("Discovered {} testable failure modes from protocol rules: {}",
                    len(modes), modes)
    else:
        logger.warning("No testable failure modes discovered — will use REFERENCE_FAILURE_MODES")
    return modes


# Maps a failure mode name to its patient-builder function.
_MODE_BUILDER = {
    "insulin_use":           _build_ineligible_insulin,
    "low_egfr":              _build_ineligible_egfr,
    "low_hba1c":             _build_ineligible_hba1c,
    "high_hba1c":            _build_ineligible_high_hba1c,
    "malignancy":            _build_ineligible_malignancy,
    "uncontrolled_diabetes": _build_ineligible_uncontrolled_diabetes,
    "pancreatitis":          _build_ineligible_pancreatitis,
    "pregnancy":             _build_ineligible_pregnancy,
    "severe_hypoglycemia":   _build_ineligible_severe_hypoglycemia,
}


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class GroundTruthEvaluator:

    def build_ground_truth_set(
        self, protocol_id: int, rules: Optional[list[CriterionRule]] = None
    ) -> list[GroundTruthPatient]:
        """Create 100 deterministic patients with known correct verdicts.

        When `rules` are provided the failure modes for the 40 INELIGIBLE patients
        are DERIVED from the protocol's actual extracted rules, so the evaluation
        tests criteria that genuinely exist in the protocol (not hardcoded modes
        that may not apply).  If the protocol has no recognizable testable modes
        (unlikely) we fall back to the reference T2DM failure modes.
        """
        rng = random.Random(42)
        patients: list[GroundTruthPatient] = []

        thr = derive_thresholds(rules) if rules else dict(DEFAULT_THRESHOLDS)
        logger.info(
            "Building 100-patient ground truth evaluation set (thresholds {})...",
            "derived from flagship rules" if thr.get("derived") else "from reference values",
        )

        # Discover which failure modes actually have corresponding rules
        if rules:
            active_modes = discover_failure_modes(rules)
            if not active_modes:
                logger.warning(
                    "No testable failure modes found in flagship rules — "
                    "falling back to REFERENCE_FAILURE_MODES: {}",
                    REFERENCE_FAILURE_MODES,
                )
                active_modes = list(REFERENCE_FAILURE_MODES)
            elif len(active_modes) < 2:
                logger.warning(
                    "Only {} testable failure mode(s) found — "
                    "augmenting with REFERENCE_FAILURE_MODES for evaluation completeness",
                    len(active_modes),
                )
                for m in REFERENCE_FAILURE_MODES:
                    if m not in active_modes:
                        active_modes.append(m)
        else:
            active_modes = list(REFERENCE_FAILURE_MODES)

        # Log which modes are being tested
        for m in active_modes:
            if m in _MODE_BUILDER:
                logger.info("  Failure mode '{}' ← has patient builder", m)
            else:
                logger.warning("  Failure mode '{}' ← NO patient builder — skipping", m)
        active_modes = [m for m in active_modes if m in _MODE_BUILDER]

        # 40 TRUE_ELIGIBLE
        logger.info("Creating 40 TRUE_ELIGIBLE patients...")
        for i in range(1, 41):
            patients.append(_build_eligible_patient(i, rng, thr))
        logger.info("  ✓ {} eligible patients created", 40)

        # 40 TRUE_INELIGIBLE — distributed evenly across active failure modes
        logger.info(
            "Creating 40 TRUE_INELIGIBLE patients across {} mode(s): {}",
            len(active_modes), active_modes,
        )
        ineligible_per_mode = 40 // len(active_modes)
        remainder = 40 % len(active_modes)
        patient_idx = 41
        for mode_rank, mode in enumerate(active_modes):
            count = ineligible_per_mode + (1 if mode_rank < remainder else 0)
            builder = _MODE_BUILDER[mode]
            for _ in range(count):
                patients.append(builder(patient_idx, rng, thr))
                patient_idx += 1
            logger.info("  ✓ {} '{}' patients created", count, mode)
        logger.info("  ✓ {} ineligible patients total", patient_idx - 41)

        # 20 BORDERLINE
        logger.info("Creating 20 BORDERLINE patients...")
        for i in range(81, 101):
            patients.append(_build_borderline_patient(i, rng, thr))
        logger.info("  ✓ {} borderline patients created", 20)

        logger.info(
            "✓ Ground truth set complete: 40 eligible, {} ineligible, 20 borderline",
            patient_idx - 41,
        )
        return patients

    def run_evaluation(
        self,
        ground_truth_patients: list[GroundTruthPatient],
        rules: list[CriterionRule],
    ) -> list[EvaluationPair]:
        """Score each patient against protocol rules and compare to ground truth."""
        from app.services.fhir_parser import fhir_parser

        engine = ScoringEngine()
        results: list[EvaluationPair] = []
        correct = 0
        wrong = 0

        for gt_patient in ground_truth_patients:
            try:
                parsed = fhir_parser.parse_bundle(gt_patient.fhir_bundle)
                scoring_result = engine.evaluate_patient(parsed, rules)
            except Exception as e:
                logger.error("Error scoring patient {}: {}", gt_patient.patient_id, e)
                continue

            predicted = scoring_result.overall_verdict.value
            actual = gt_patient.ground_truth_verdict
            is_correct = predicted == actual

            pass_count = sum(1 for ev in scoring_result.evaluations if ev.status == EvaluationStatus.PASS)
            fail_count = sum(1 for ev in scoring_result.evaluations if ev.status == EvaluationStatus.FAIL)
            amb_count  = sum(1 for ev in scoring_result.evaluations if ev.status == EvaluationStatus.AMBIGUOUS)

            # Extract HbA1c and eGFR from parsed labs
            hba1c_val = next(
                (l.value for l in parsed.lab_results
                 if "hemoglobin a1c" in l.name.lower() or "hba1c" in l.name.lower()),
                None,
            )
            egfr_val = next(
                (l.value for l in parsed.lab_results
                 if "glomerular filtration" in l.name.lower() or "egfr" in l.name.lower()),
                None,
            )

            if is_correct:
                correct += 1
                logger.info(
                    "Evaluating {} ({})... predicted {} ✓ CORRECT",
                    gt_patient.patient_id, actual, predicted,
                )
            else:
                wrong += 1
                logger.warning(
                    "Evaluating {} ({})... predicted {} ✗ WRONG",
                    gt_patient.patient_id, actual, predicted,
                )
                logger.warning(
                    "  Expected {}, got {} — mode: {}",
                    actual, predicted, gt_patient.failure_mode or "n/a",
                )

            results.append(EvaluationPair(
                patient_id=gt_patient.patient_id,
                ground_truth_verdict=actual,
                predicted_verdict=predicted,
                predicted_score=scoring_result.fit_score,
                confidence_low=scoring_result.confidence_low,
                confidence_high=scoring_result.confidence_high,
                failure_mode=gt_patient.failure_mode,
                criterion_pass_count=pass_count,
                criterion_fail_count=fail_count,
                criterion_ambiguous_count=amb_count,
                correct=is_correct,
                age=parsed.age,
                hba1c=hba1c_val,
                egfr=egfr_val,
                conditions=parsed.conditions,
                medications=parsed.medications,
            ))

        logger.info(
            "Evaluation complete: {}/{} correct ({:.0f}%)",
            correct, len(results), 100 * correct / max(len(results), 1)
        )
        return results

    def compute_metrics(
        self, results: list[EvaluationPair], protocol_title: str = ""
    ) -> AccuracyMetrics:
        """
        Option B sensitivity: REVIEW_NEEDED counts as TRUE POSITIVE.
        Only INELIGIBLE predicted for a truly eligible patient is a FALSE NEGATIVE.
        """
        eligible_gt   = [r for r in results if r.ground_truth_verdict == "ELIGIBLE"]
        ineligible_gt = [r for r in results if r.ground_truth_verdict == "INELIGIBLE"]
        borderline_gt = [r for r in results if r.ground_truth_verdict == "REVIEW_NEEDED"]

        # TP = truly eligible predicted ELIGIBLE or REVIEW_NEEDED
        tp      = len([r for r in eligible_gt if r.predicted_verdict in ("ELIGIBLE", "REVIEW_NEEDED")])
        fn      = len([r for r in eligible_gt if r.predicted_verdict == "INELIGIBLE"])
        tn      = len([r for r in ineligible_gt if r.predicted_verdict == "INELIGIBLE"])
        fp_hard = len([r for r in ineligible_gt if r.predicted_verdict == "ELIGIBLE"])
        fp_soft = len([r for r in ineligible_gt if r.predicted_verdict == "REVIEW_NEEDED"])
        fp      = fp_hard + fp_soft

        sensitivity = tp / len(eligible_gt)   if eligible_gt   else 0.0
        specificity = tn / len(ineligible_gt) if ineligible_gt else 0.0
        ppv         = tp / (tp + fp)          if (tp + fp) > 0 else 0.0
        npv         = tn / (tn + fn)          if (tn + fn) > 0 else 0.0
        f1          = (2 * ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
        accuracy    = (tp + tn) / len(results) if results else 0.0

        # False negatives detail
        fn_patients = [r for r in eligible_gt if r.predicted_verdict == "INELIGIBLE"]
        if fn_patients:
            logger.error(
                "FALSE NEGATIVES: {}/{} truly eligible patients predicted INELIGIBLE — "
                "these are the cases most urgently needing investigation",
                len(fn_patients), len(eligible_gt)
            )
            for fn_r in fn_patients:
                logger.error(
                    "  ✗ FN Patient {}: score={} | fail={} excl_triggered={} "
                    "incl_ambig={} excl_ambig={}",
                    fn_r.patient_id, fn_r.predicted_score,
                    fn_r.criterion_fail_count, 0,
                    fn_r.criterion_ambiguous_count, 0,
                )

        # Failure mode breakdown — computed dynamically from whatever modes are
        # present in the evaluation pairs (not a hardcoded list of 4 modes).
        failure_mode_accuracy: dict[str, Any] = {}
        present_modes = {r.failure_mode for r in ineligible_gt if r.failure_mode}
        for mode in sorted(present_modes):
            mode_patients = [r for r in ineligible_gt if r.failure_mode == mode]
            correct_mode = [r for r in mode_patients if r.predicted_verdict == "INELIGIBLE"]
            acc_val = len(correct_mode) / len(mode_patients)
            failure_mode_accuracy[mode] = {
                "total":    len(mode_patients),
                "correct":  len(correct_mode),
                "accuracy": round(acc_val, 4),
            }

        # Borderline: what fraction got REVIEW_NEEDED
        borderline_review_rate = 0.0
        if borderline_gt:
            borderline_review_rate = sum(
                1 for r in borderline_gt if r.predicted_verdict == "REVIEW_NEEDED"
            ) / len(borderline_gt)

        # Confidence band calibration
        widths = [r.confidence_high - r.confidence_low for r in results]
        mean_confidence_width = sum(widths) / max(len(widths), 1)

        coverage_hits = sum(
            1 for r in results
            if r.confidence_low <= r.predicted_score <= r.confidence_high
        )
        confidence_coverage = coverage_hits / max(len(results), 1)

        meets_target = sensitivity >= 0.85

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS — {} patients | {}", len(results), protocol_title)
        logger.info("=" * 60)
        logger.info(
            "Sensitivity (Option B): {:.1%}  [TARGET ≥85%] {}",
            sensitivity, "✓ MEETS TARGET" if meets_target else "✗ BELOW TARGET"
        )
        logger.info("Specificity:            {:.1%}", specificity)
        logger.info("PPV (precision):        {:.1%}", ppv)
        logger.info("NPV:                    {:.1%}", npv)
        logger.info("F1 Score:               {:.1%}", f1)
        logger.info("Overall Accuracy:       {:.1%}", accuracy)
        logger.info("-" * 60)
        logger.info("True Positives  (TP): {:3d}/{} (eligible predicted ELIGIBLE or REVIEW_NEEDED)",
                    tp, len(eligible_gt))
        logger.info("False Negatives (FN): {:3d}/{} (eligible predicted INELIGIBLE — CRITICAL)",
                    fn, len(eligible_gt))
        logger.info("True Negatives  (TN): {:3d}/{} (ineligible correctly rejected)",
                    tn, len(ineligible_gt))
        logger.info("False Pos Hard  (FP): {:3d}/{} (said ELIGIBLE, truly not — bad)",
                    fp_hard, len(ineligible_gt))
        logger.info("False Pos Soft  (FP): {:3d}/{} (REVIEW_NEEDED, truly not — coordinator catches)",
                    fp_soft, len(ineligible_gt))
        logger.info("-" * 60)
        for mode, stats in failure_mode_accuracy.items():
            logger.info(
                "  {:20s}: {}/{} correct ({:.0%}){}",
                mode, stats["correct"], stats["total"], stats["accuracy"],
                " ← investigate" if stats["accuracy"] < 0.80 else "",
            )
        logger.info("-" * 60)
        logger.info(
            "Borderline REVIEW rate: {:.1%} ({}/{} correctly sent to review)",
            borderline_review_rate,
            int(borderline_review_rate * len(borderline_gt)) if borderline_gt else 0,
            len(borderline_gt),
        )
        logger.info("Mean confidence band:   ±{:.1f} pts", mean_confidence_width / 2)
        logger.info("=" * 60)

        return AccuracyMetrics(
            sensitivity=round(sensitivity, 4),
            specificity=round(specificity, 4),
            ppv=round(ppv, 4),
            npv=round(npv, 4),
            f1_score=round(f1, 4),
            accuracy=round(accuracy, 4),
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_positives_hard=fp_hard,
            false_positives_soft=fp_soft,
            false_negatives=fn,
            failure_mode_accuracy=failure_mode_accuracy,
            borderline_review_rate=round(borderline_review_rate, 4),
            mean_confidence_width=round(mean_confidence_width, 2),
            confidence_coverage=round(confidence_coverage, 4),
            meets_sensitivity_target=meets_target,
            total_evaluated=len(results),
            total_patients=len(results),
            eligible_count=len(eligible_gt),
            ineligible_count=len(ineligible_gt),
            borderline_count=len(borderline_gt),
            protocol_title=protocol_title,
        )

    def export_annotated_csv(
        self,
        ground_truth_patients: list[GroundTruthPatient],
        results: list[EvaluationPair],
    ) -> str:
        """Return CSV string of the 100-patient annotated dataset."""
        result_map = {r.patient_id: r for r in results}
        lines = [
            "patient_id,age,gender,hba1c,egfr,conditions,medications,"
            "ground_truth_verdict,ground_truth_reason,failure_mode,"
            "predicted_verdict,predicted_score,confidence_low,confidence_high,"
            "correct,criterion_pass_count,criterion_fail_count,criterion_ambiguous_count"
        ]

        for gt in ground_truth_patients:
            r = result_map.get(gt.patient_id)
            if not r:
                continue

            def _esc(s: str) -> str:
                return '"' + str(s).replace('"', '""') + '"'

            conditions_str = _esc("|".join(r.conditions))
            medications_str = _esc("|".join(r.medications))
            reason_str = _esc(gt.ground_truth_reason)

            lines.append(
                ",".join([
                    gt.patient_id,
                    str(r.age or ""),
                    "",  # gender not stored in EvaluationPair
                    str(round(r.hba1c, 1)) if r.hba1c else "",
                    str(round(r.egfr, 0)) if r.egfr else "",
                    conditions_str,
                    medications_str,
                    gt.ground_truth_verdict,
                    reason_str,
                    gt.failure_mode or "",
                    r.predicted_verdict,
                    str(r.predicted_score),
                    str(r.confidence_low),
                    str(r.confidence_high),
                    str(r.correct).lower(),
                    str(r.criterion_pass_count),
                    str(r.criterion_fail_count),
                    str(r.criterion_ambiguous_count),
                ])
            )

        return "\n".join(lines)


evaluator = GroundTruthEvaluator()
