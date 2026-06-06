import pytest
from app.services.fhir_parser import fhir_parser

SAMPLE_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "PT-001",
                "name": [{"use": "official", "family": "Smith", "given": ["John"]}],
                "gender": "male",
                "birthDate": "1975-03-15",
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{"code": "active"}]
                },
                "code": {
                    "coding": [{"display": "Type 2 diabetes mellitus"}],
                    "text": "Type 2 diabetes mellitus",
                },
                "subject": {"reference": "Patient/PT-001"},
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{"code": "active"}]
                },
                "code": {"text": "Essential hypertension"},
                "subject": {"reference": "Patient/PT-001"},
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [{"display": "Metformin 1000mg"}],
                    "text": "Metformin 1000mg",
                },
                "subject": {"reference": "Patient/PT-001"},
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [{"display": "HbA1c"}],
                    "text": "HbA1c",
                },
                "valueQuantity": {"value": 8.5, "unit": "%"},
                "effectiveDateTime": "2024-01-15T10:00:00Z",
                "subject": {"reference": "Patient/PT-001"},
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {"text": "eGFR"},
                "valueQuantity": {"value": 75.0, "unit": "mL/min/1.73m2"},
                "subject": {"reference": "Patient/PT-001"},
            }
        },
    ],
}


class TestFHIRParser:
    def test_patient_demographics(self):
        result = fhir_parser.parse_bundle(SAMPLE_BUNDLE)
        assert result.patient_id == "PT-001"
        assert result.name == "John Smith"
        assert result.gender == "male"
        assert result.age is not None
        assert result.age > 0

    def test_conditions_extracted(self):
        result = fhir_parser.parse_bundle(SAMPLE_BUNDLE)
        assert len(result.conditions) == 2
        assert any("diabetes" in c.lower() for c in result.conditions)
        assert any("hypertension" in c.lower() for c in result.conditions)

    def test_medications_extracted(self):
        result = fhir_parser.parse_bundle(SAMPLE_BUNDLE)
        assert len(result.medications) == 1
        assert any("metformin" in m.lower() for m in result.medications)

    def test_lab_results_extracted(self):
        result = fhir_parser.parse_bundle(SAMPLE_BUNDLE)
        assert len(result.lab_results) >= 2
        names_lower = [l.name.lower() for l in result.lab_results]
        assert any("hba1c" in n for n in names_lower)
        assert any("egfr" in n for n in names_lower)

    def test_lab_values_correct(self):
        result = fhir_parser.parse_bundle(SAMPLE_BUNDLE)
        hba1c = next((l for l in result.lab_results if "hba1c" in l.name.lower()), None)
        assert hba1c is not None
        assert hba1c.value == 8.5
        assert hba1c.unit == "%"

    def test_empty_bundle_does_not_crash(self):
        result = fhir_parser.parse_bundle({})
        assert result is not None
        assert result.conditions == []
        assert result.medications == []
        assert result.lab_results == []

    def test_missing_optional_fields_graceful(self):
        minimal = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "PT-999"}}
            ],
        }
        result = fhir_parser.parse_bundle(minimal)
        assert result.patient_id == "PT-999"
        assert result.age is None
        assert result.gender is None

    def test_resolved_conditions_excluded(self):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "clinicalStatus": {"coding": [{"code": "resolved"}]},
                        "code": {"text": "Old condition"},
                    }
                }
            ],
        }
        result = fhir_parser.parse_bundle(bundle)
        assert len(result.conditions) == 0
