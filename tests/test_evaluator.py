"""Tests for evaluator.py"""

import pytest
from app.services.evaluator import (
    GroundTruthEvaluator,
    EvaluationPair,
    AccuracyMetrics,
    REFERENCE_RULES,
)


@pytest.fixture
def ev():
    return GroundTruthEvaluator()


# Test 1: build_ground_truth_set returns exactly 100 patients
def test_build_ground_truth_count(ev):
    patients = ev.build_ground_truth_set(protocol_id=1)
    assert len(patients) == 100


# Test 2: TRUE_ELIGIBLE patients have correct FHIR structure
def test_eligible_patients_have_hba1c_and_egfr(ev):
    patients = ev.build_ground_truth_set(protocol_id=1)
    eligible = [p for p in patients if p.ground_truth_verdict == "ELIGIBLE"]
    assert len(eligible) == 40

    for p in eligible:
        entries = p.fhir_bundle.get("entry", [])
        resources = [e["resource"] for e in entries]
        resource_types = [r["resourceType"] for r in resources]
        assert "Patient" in resource_types
        assert "Observation" in resource_types
        assert "Condition" in resource_types

        obs_displays = [
            coding["display"]
            for r in resources if r["resourceType"] == "Observation"
            for coding in r.get("code", {}).get("coding", [])
        ]
        hba1c_found = any("Hemoglobin A1c" in d for d in obs_displays)
        egfr_found = any("Glomerular filtration" in d for d in obs_displays)
        assert hba1c_found, f"Patient {p.patient_id} missing HbA1c"
        assert egfr_found, f"Patient {p.patient_id} missing eGFR"


# Test 3: TRUE_INELIGIBLE patients have correct failure modes
def test_ineligible_failure_modes(ev):
    patients = ev.build_ground_truth_set(protocol_id=1)
    ineligible = [p for p in patients if p.ground_truth_verdict == "INELIGIBLE"]
    assert len(ineligible) == 40

    modes = [p.failure_mode for p in ineligible]
    assert modes.count("insulin_use") == 10
    assert modes.count("low_egfr") == 10
    assert modes.count("low_hba1c") == 10
    assert modes.count("malignancy") == 10


# Test 4: compute_metrics with all correct → sensitivity=1.0, specificity=1.0
def test_compute_metrics_perfect(ev):
    pairs = []
    for i in range(40):
        pairs.append(EvaluationPair(
            patient_id=f"GT-{i:03d}", ground_truth_verdict="ELIGIBLE",
            predicted_verdict="ELIGIBLE", predicted_score=90,
            confidence_low=80, confidence_high=95, correct=True,
        ))
    for i in range(40, 80):
        pairs.append(EvaluationPair(
            patient_id=f"GT-{i:03d}", ground_truth_verdict="INELIGIBLE",
            predicted_verdict="INELIGIBLE", predicted_score=30,
            confidence_low=20, confidence_high=40, correct=True,
            failure_mode="low_egfr",
        ))

    metrics = ev.compute_metrics(pairs)
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.true_positives == 40
    assert metrics.true_negatives == 40
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


# Test 5: compute_metrics with all wrong → sensitivity=0.0
def test_compute_metrics_all_wrong(ev):
    pairs = []
    for i in range(40):
        pairs.append(EvaluationPair(
            patient_id=f"GT-{i:03d}", ground_truth_verdict="ELIGIBLE",
            predicted_verdict="INELIGIBLE", predicted_score=30,
            confidence_low=20, confidence_high=40, correct=False,
        ))
    for i in range(40, 80):
        pairs.append(EvaluationPair(
            patient_id=f"GT-{i:03d}", ground_truth_verdict="INELIGIBLE",
            predicted_verdict="ELIGIBLE", predicted_score=80,
            confidence_low=70, confidence_high=90, correct=False,
            failure_mode="low_egfr",
        ))

    metrics = ev.compute_metrics(pairs)
    assert metrics.sensitivity == 0.0
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 40
    assert metrics.false_positives == 40


# Test 6: False negative detection counted correctly
def test_false_negative_detection(ev):
    pairs = [
        EvaluationPair(
            patient_id="GT-001", ground_truth_verdict="ELIGIBLE",
            predicted_verdict="ELIGIBLE", predicted_score=85,
            confidence_low=75, confidence_high=90, correct=True,
        ),
        EvaluationPair(
            patient_id="GT-002", ground_truth_verdict="ELIGIBLE",
            predicted_verdict="INELIGIBLE", predicted_score=35,
            confidence_low=25, confidence_high=40, correct=False,
        ),
        EvaluationPair(
            patient_id="GT-003", ground_truth_verdict="INELIGIBLE",
            predicted_verdict="INELIGIBLE", predicted_score=20,
            confidence_low=10, confidence_high=30, correct=True,
            failure_mode="low_egfr",
        ),
    ]
    metrics = ev.compute_metrics(pairs)
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1
    assert metrics.true_negatives == 1


# Test 7: meets_sensitivity_target correct boundaries
def test_sensitivity_target_boundary(ev):
    def _make_metrics(tp: int, fn: int) -> AccuracyMetrics:
        pairs_e = [
            EvaluationPair(
                patient_id=f"GT-{i:03d}", ground_truth_verdict="ELIGIBLE",
                predicted_verdict="ELIGIBLE" if i < tp else "INELIGIBLE",
                predicted_score=85 if i < tp else 35,
                confidence_low=75, confidence_high=90,
                correct=i < tp,
            )
            for i in range(tp + fn)
        ]
        return ev.compute_metrics(pairs_e)

    metrics_low = _make_metrics(tp=84, fn=16)   # 84% → below target
    assert metrics_low.meets_sensitivity_target is False

    metrics_high = _make_metrics(tp=85, fn=15)   # 85% → meets target
    assert metrics_high.meets_sensitivity_target is True


# Test 8: export_annotated_csv returns valid CSV with required columns
def test_export_annotated_csv(ev):
    patients = ev.build_ground_truth_set(protocol_id=1)
    pairs = [
        EvaluationPair(
            patient_id=p.patient_id,
            ground_truth_verdict=p.ground_truth_verdict,
            predicted_verdict=p.ground_truth_verdict,  # perfect predictions
            predicted_score=85,
            confidence_low=75,
            confidence_high=90,
            failure_mode=p.failure_mode,
            correct=True,
            age=50,
            hba1c=8.2,
            egfr=75.0,
        )
        for p in patients
    ]
    csv = ev.export_annotated_csv(patients, pairs)
    lines = csv.strip().split("\n")
    header = lines[0]

    required_cols = [
        "patient_id", "age", "ground_truth_verdict", "predicted_verdict",
        "predicted_score", "correct", "failure_mode",
        "criterion_pass_count", "criterion_fail_count", "criterion_ambiguous_count",
    ]
    for col in required_cols:
        assert col in header, f"Missing column: {col}"

    assert len(lines) == 101  # header + 100 patients
