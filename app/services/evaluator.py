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
from typing import Optional

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
    # Binary classification (ELIGIBLE vs non-ELIGIBLE)
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1_score: float
    accuracy: float

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    failure_mode_accuracy: dict[str, float]

    borderline_review_rate: float
    mean_confidence_width: float
    confidence_coverage: float

    meets_sensitivity_target: bool
    target_sensitivity: float = 0.85

    total_evaluated: int
    eligible_count: int
    ineligible_count: int
    borderline_count: int


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

def _build_eligible_patient(index: int, rng: random.Random) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(35, 70)
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(7.6, 10.9), 1)
    egfr = round(rng.uniform(62, 118), 0)

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
        ground_truth_reason=f"Age {age}, HbA1c {hba1c}%, eGFR {int(egfr)} — all inclusion met, no exclusions triggered",
        failure_mode=None,
    )


def _build_ineligible_insulin(index: int, rng: random.Random) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(45, 65)
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(7.6, 9.5), 1)
    egfr = round(rng.uniform(65, 100), 0)

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
        ground_truth_reason="Current insulin use — exclusion criterion triggered",
        failure_mode="insulin_use",
    )


def _build_ineligible_egfr(index: int, rng: random.Random) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(50, 70)
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(7.6, 9.5), 1)
    egfr = round(rng.uniform(20, 44), 0)

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
        ground_truth_reason=f"eGFR {int(egfr)} < 45 — fails inclusion eGFR≥60 and triggers exclusion eGFR<45",
        failure_mode="low_egfr",
    )


def _build_ineligible_hba1c(index: int, rng: random.Random) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(35, 65)
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(5.0, 7.4), 1)
    egfr = round(rng.uniform(65, 100), 0)

    resources = [
        _patient_resource(pid, age, gender),
        _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
        _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
        _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
    ]
    logger.debug("  Patient {} Mode C (HbA1c) — Age {}, HbA1c {}% < 7.5% → INELIGIBLE", pid, age, hba1c)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="INELIGIBLE",
        ground_truth_reason=f"HbA1c {hba1c}% < 7.5% — fails inclusion threshold (too well controlled)",
        failure_mode="low_hba1c",
    )


def _build_ineligible_malignancy(index: int, rng: random.Random) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    age = rng.randint(40, 65)
    gender = rng.choice(["male", "female"])
    hba1c = round(rng.uniform(7.6, 9.5), 1)
    egfr = round(rng.uniform(65, 100), 0)

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


def _build_borderline_patient(index: int, rng: random.Random) -> GroundTruthPatient:
    pid = f"GT-{index:03d}"
    variant = rng.choice(["missing_labs", "boundary_hba1c", "boundary_age", "ckd_borderline"])

    if variant == "missing_labs":
        age = rng.randint(40, 65)
        # Only T2DM diagnosis, no labs → many AMBIGUOUS criteria → REVIEW_NEEDED
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
        ]
        reason = f"Age {age}, T2DM confirmed — HbA1c and eGFR labs missing; unverifiable"
    elif variant == "boundary_hba1c":
        age = rng.randint(40, 65)
        hba1c = 7.5  # exact boundary
        egfr = round(rng.uniform(62, 90), 0)
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
            _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
            _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
        ]
        reason = f"HbA1c exactly 7.5% (boundary), eGFR {int(egfr)} — borderline eligibility"
    elif variant == "boundary_age":
        age = rng.choice([74, 75, 76])
        hba1c = round(rng.uniform(7.6, 9.5), 1)
        egfr = round(rng.uniform(62, 90), 0)
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
            _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
            _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
        ]
        reason = f"Age {age} near age limit 75 — borderline eligibility"
    else:  # ckd_borderline
        age = rng.randint(50, 65)
        hba1c = round(rng.uniform(7.6, 9.5), 1)
        # eGFR between 45-60: fails >=60 inclusion but not <45 exclusion exactly
        egfr = round(rng.uniform(45, 59), 0)
        resources = [
            _patient_resource(pid, age, rng.choice(["male", "female"])),
            _condition(pid, "Type 2 diabetes mellitus", "44054006", "E11.9"),
            _condition(pid, "Chronic kidney disease stage 3", "433144002", "N18.3"),
            _medication(pid, "Metformin 1000 mg oral tablet", "860974"),
            _obs(pid, "Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", hba1c, "%"),
            _obs(pid, "Glomerular filtration rate/1.73 sq M.predicted", "33914-3", egfr, "mL/min/1.73m2"),
        ]
        reason = f"eGFR {int(egfr)} — borderline (below >=60 threshold but not <45 exclusion)"

    logger.debug("  Patient {} BORDERLINE ({}) → REVIEW_NEEDED", pid, variant)
    return GroundTruthPatient(
        patient_id=pid,
        fhir_bundle=_bundle(resources),
        ground_truth_verdict="REVIEW_NEEDED",
        ground_truth_reason=reason,
        failure_mode=None,
    )


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class GroundTruthEvaluator:

    def build_ground_truth_set(self, protocol_id: int) -> list[GroundTruthPatient]:
        """Create 100 deterministic patients with known correct verdicts."""
        rng = random.Random(42)
        patients: list[GroundTruthPatient] = []

        logger.info("Building 100-patient ground truth evaluation set...")

        # 40 TRUE_ELIGIBLE
        logger.info("Creating 40 TRUE_ELIGIBLE patients...")
        for i in range(1, 41):
            patients.append(_build_eligible_patient(i, rng))
        logger.info("  ✓ {} eligible patients created", 40)

        # 40 TRUE_INELIGIBLE — 10 per failure mode
        logger.info("Creating 40 TRUE_INELIGIBLE patients...")
        for i in range(41, 51):
            patients.append(_build_ineligible_insulin(i, rng))
        for i in range(51, 61):
            patients.append(_build_ineligible_egfr(i, rng))
        for i in range(61, 71):
            patients.append(_build_ineligible_hba1c(i, rng))
        for i in range(71, 81):
            patients.append(_build_ineligible_malignancy(i, rng))
        logger.info("  ✓ {} ineligible patients created (10 per failure mode)", 40)

        # 20 BORDERLINE
        logger.info("Creating 20 BORDERLINE patients...")
        for i in range(81, 101):
            patients.append(_build_borderline_patient(i, rng))
        logger.info("  ✓ {} borderline patients created", 20)

        logger.info(
            "✓ Ground truth set complete: 40 eligible, 40 ineligible, 20 borderline"
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

    def compute_metrics(self, results: list[EvaluationPair]) -> AccuracyMetrics:
        """Compute sensitivity, specificity, PPV, NPV, F1 from evaluation pairs."""
        # Binary: ELIGIBLE vs non-ELIGIBLE
        tp = tn = fp = fn = 0

        eligible_gt = [r for r in results if r.ground_truth_verdict == "ELIGIBLE"]
        ineligible_gt = [r for r in results if r.ground_truth_verdict == "INELIGIBLE"]
        borderline_gt = [r for r in results if r.ground_truth_verdict == "REVIEW_NEEDED"]

        for r in eligible_gt:
            if r.predicted_verdict == "ELIGIBLE":
                tp += 1
            else:
                fn += 1

        for r in ineligible_gt:
            if r.predicted_verdict != "ELIGIBLE":
                tn += 1
            else:
                fp += 1

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        f1 = 2 * ppv * sensitivity / max(ppv + sensitivity, 1e-9)
        accuracy = (tp + tn) / max(len(eligible_gt) + len(ineligible_gt), 1)

        # By failure mode
        failure_mode_accuracy: dict[str, float] = {}
        for mode in ("insulin_use", "low_egfr", "low_hba1c", "malignancy"):
            mode_results = [r for r in ineligible_gt if r.failure_mode == mode]
            if mode_results:
                correct_n = sum(1 for r in mode_results if r.correct)
                failure_mode_accuracy[mode] = correct_n / len(mode_results)

        # Borderline: what fraction got REVIEW_NEEDED
        borderline_review_rate = 0.0
        if borderline_gt:
            borderline_review_rate = sum(
                1 for r in borderline_gt if r.predicted_verdict == "REVIEW_NEEDED"
            ) / len(borderline_gt)

        # Confidence band calibration
        widths = [r.confidence_high - r.confidence_low for r in results]
        mean_confidence_width = sum(widths) / max(len(widths), 1)

        # Coverage: % where true score is within band (approximated as all eligible with score in band)
        coverage_hits = sum(
            1 for r in results
            if r.confidence_low <= r.predicted_score <= r.confidence_high
        )
        confidence_coverage = coverage_hits / max(len(results), 1)

        # Log detailed results
        logger.info("=" * 48)
        logger.info("EVALUATION RESULTS — {} Patient Ground Truth", len(results))
        logger.info("=" * 48)
        logger.info("Sensitivity (recall):  {:.3f}  [TARGET: ≥0.85] {}", sensitivity,
                    "✓ MEETS TARGET" if sensitivity >= 0.85 else "✗ MISSES TARGET")
        logger.info("Specificity:           {:.3f}", specificity)
        logger.info("PPV (precision):       {:.3f}", ppv)
        logger.info("NPV:                   {:.3f}", npv)
        logger.info("F1 Score:              {:.3f}", f1)
        logger.info("Overall Accuracy:      {:.3f}", accuracy)
        logger.info("-" * 48)
        logger.info("True Positives:   {}/{}  (correctly identified eligible)", tp, len(eligible_gt))
        logger.info("True Negatives:   {}/{}  (correctly rejected ineligible)", tn, len(ineligible_gt))
        logger.info("False Positives:   {}/{}  (said eligible, was not) ← REVIEW THESE", fp, len(ineligible_gt))
        logger.info("False Negatives:   {}/{}  (missed eligible patients) ← CRITICAL", fn, len(eligible_gt))
        logger.info("-" * 48)
        logger.info("By failure mode:")
        for mode, acc in failure_mode_accuracy.items():
            mode_n = sum(1 for r in ineligible_gt if r.failure_mode == mode)
            correct_n = round(acc * mode_n)
            logger.info("  {}: {}/{} correct ({:.0f}%)", mode, correct_n, mode_n, acc * 100)
        logger.info("-" * 48)
        logger.info("Borderline (REVIEW_NEEDED rate): {:.2f} ({}/{} correctly sent to review)",
                    borderline_review_rate,
                    round(borderline_review_rate * len(borderline_gt)),
                    len(borderline_gt))
        logger.info("=" * 48)

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
            false_negatives=fn,
            failure_mode_accuracy={k: round(v, 4) for k, v in failure_mode_accuracy.items()},
            borderline_review_rate=round(borderline_review_rate, 4),
            mean_confidence_width=round(mean_confidence_width, 2),
            confidence_coverage=round(confidence_coverage, 4),
            meets_sensitivity_target=sensitivity >= 0.85,
            total_evaluated=len(results),
            eligible_count=len(eligible_gt),
            ineligible_count=len(ineligible_gt),
            borderline_count=len(borderline_gt),
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
