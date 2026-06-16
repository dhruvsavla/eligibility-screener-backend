import pytest
from app.services.scoring_engine import scoring_engine
from app.models.patient import PatientData, LabResult
from app.models.protocol import CriterionRule, CriterionType
from app.models.screening import VerdictType, EvaluationStatus


def make_rule(
    id: int,
    concept: str,
    operator: str,
    value: str,
    criterion_type: str = "inclusion",
    required: bool = True,
) -> CriterionRule:
    return CriterionRule(
        id=id,
        protocol_id=1,
        criterion_text=f"{concept} {operator} {value}",
        concept=concept,
        operator=operator,
        value=value,
        required=required,
        criterion_type=CriterionType(criterion_type),
        confidence=0.95,
    )


def make_patient(
    age: int = 50,
    conditions: list[str] = None,
    medications: list[str] = None,
    labs: list[LabResult] = None,
) -> PatientData:
    return PatientData(
        patient_id="PT-TEST",
        name="Test Patient",
        age=age,
        gender="male",
        conditions=conditions or [],
        medications=medications or [],
        lab_results=labs or [],
    )


class TestScoringEngine:
    def test_all_pass_returns_eligible(self):
        rules = [
            make_rule(1, "age", "between", "18-65"),
            make_rule(2, "HbA1c", ">=", "7.5%"),
        ]
        patient = make_patient(
            age=45,
            labs=[LabResult(name="HbA1c", value=8.5, unit="%")],
        )
        result = scoring_engine.evaluate_patient(patient, rules)
        assert result.fit_score >= 75
        assert result.overall_verdict == VerdictType.ELIGIBLE

    def test_mandatory_inclusion_fail_returns_ineligible(self):
        rules = [
            make_rule(1, "age", "between", "18-40"),
            make_rule(2, "HbA1c", ">=", "7.5%"),
        ]
        patient = make_patient(
            age=65,
            labs=[LabResult(name="HbA1c", value=8.5, unit="%")],
        )
        result = scoring_engine.evaluate_patient(patient, rules)
        assert result.fit_score <= 40 or result.overall_verdict == VerdictType.INELIGIBLE

    def test_all_ambiguous_returns_review_needed_with_wide_band(self):
        rules = [
            make_rule(1, "eGFR", ">=", "60"),
            make_rule(2, "creatinine", "<", "1.5 mg/dL"),
            make_rule(3, "platelet", ">", "100"),
        ]
        patient = make_patient(age=50)
        result = scoring_engine.evaluate_patient(patient, rules)
        assert result.overall_verdict == VerdictType.REVIEW_NEEDED
        band_width = result.confidence_high - result.confidence_low
        assert band_width >= 8

    def test_exclusion_criterion_fires_reduces_score(self):
        rules = [
            make_rule(1, "insulin", "absence", "insulin", criterion_type="exclusion", required=False),
        ]
        patient = make_patient(medications=["Insulin glargine 10u"])
        result_with = scoring_engine.evaluate_patient(patient, rules)

        rules2 = [
            make_rule(1, "insulin", "absence", "insulin", criterion_type="exclusion", required=False),
        ]
        patient_without = make_patient(medications=["Metformin 1000mg"])
        result_without = scoring_engine.evaluate_patient(patient_without, rules2)

        assert result_with.fit_score < result_without.fit_score

    def test_score_clamped_between_0_and_100(self):
        rules = [make_rule(i, f"concept_{i}", "presence", "x") for i in range(10)]
        patient = make_patient()
        result = scoring_engine.evaluate_patient(patient, rules)
        assert 0 <= result.fit_score <= 100
        assert 0 <= result.confidence_low <= 100
        assert 0 <= result.confidence_high <= 100

    def test_evaluations_count_matches_rules(self):
        rules = [
            make_rule(1, "age", ">=", "18"),
            make_rule(2, "HbA1c", ">=", "7.5"),
            make_rule(3, "eGFR", ">=", "60"),
        ]
        patient = make_patient(age=30, labs=[LabResult(name="HbA1c", value=9.0, unit="%")])
        result = scoring_engine.evaluate_patient(patient, rules)
        assert len(result.evaluations) == 3

    def test_hierarchy_match_via_concept_subsumes(self, monkeypatch):
        """A patient coded with a specific diagnosis should match a rule written
        with the parent concept via SNOMED-CT hierarchy (concept_subsumes)."""
        from app.services import scoring_engine as se

        # Patient has the SPECIFIC condition; rule asks for the GENERAL parent.
        rule = make_rule(1, "Hypertensive disorder", "presence", "", "inclusion")
        patient = make_patient(age=50, conditions=["Essential hypertension"])

        # Force FAISS expansion off (no weak terms) so only hierarchy can match.
        monkeypatch.setattr(se.snomed_matcher, "find_best_match", lambda q, top_k=3: [])
        # Make the hierarchy say essential hypertension IS-A hypertensive disorder.
        def fake_subsumes(rule_concept, patient_concept):
            return (rule_concept.lower() == "hypertensive disorder"
                    and patient_concept.lower() == "essential hypertension")
        monkeypatch.setattr(se.snomed_matcher, "concept_subsumes", fake_subsumes)

        found, item, score = se.scoring_engine._concept_in_list(
            "Hypertensive disorder", patient.conditions
        )
        assert found is True
        assert item == "Essential hypertension"
        assert score == 1.0
