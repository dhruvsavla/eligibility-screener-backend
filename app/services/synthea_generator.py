"""
FALLBACK Python Synthetic Patient Generator
===========================================
This module generates synthetic FHIR R4 patient bundles entirely in Python.
It is used when the real MITRE Synthea Java tool is not available.

Differences from real Synthea:
- Smaller condition/medication vocabulary (~20 conditions vs thousands)
- No longitudinal care history (no Encounter timeline)
- Lab values are randomized within realistic ranges, not physiologically modeled
- Designed for functional testing of the eligibility screening pipeline

To use real Synthea instead: see backend/synthea/SETUP.md

Improvements over a basic generator:
- Proper SNOMED CT codes for conditions
- Proper LOINC codes for observations
- RxNorm codes for medications
- Age-stratified condition prevalence
- Correlated comorbidities (e.g. DM → likely hypertension + hyperlipidemia)
- Lab values correlated with diagnoses (diabetic → elevated HbA1c)
- Longitudinal lab history (3–5 observations over 2 years)
- Vitals: BP, BMI, weight, height, heart rate
- Condition onset dates (realistic duration before today)
- Medication start dates aligned with diagnosis
- AllergyIntolerance resources
"""

import json
import math
import random
import uuid
from datetime import date, timedelta
from loguru import logger

# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------
FIRST_NAMES_M = [
    "James", "Michael", "Robert", "David", "William", "Richard", "Charles",
    "Thomas", "Joseph", "Daniel", "Mark", "Paul", "Steven", "Andrew", "Kenneth",
    "Joshua", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy", "Jason",
    "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric", "Jonathan",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Dorothy", "Kimberly", "Emily", "Donna", "Michelle", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore",
    "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee", "Gonzalez", "Harris",
    "Clark", "Lewis", "Robinson", "Walker", "Perez", "Hall", "Young",
]

# ---------------------------------------------------------------------------
# Condition definitions with SNOMED CT codes and realistic onset ranges
# ---------------------------------------------------------------------------
CONDITIONS = {
    "type2_diabetes": {
        "display": "Type 2 diabetes mellitus",
        "snomed": "44054006",
        "icd10": "E11.9",
        "onset_years": (1, 20),
        "age_min": 30,
    },
    "hypertension": {
        "display": "Essential hypertension",
        "snomed": "59621000",
        "icd10": "I10",
        "onset_years": (1, 25),
        "age_min": 25,
    },
    "hyperlipidemia": {
        "display": "Hyperlipidemia",
        "snomed": "55822004",
        "icd10": "E78.5",
        "onset_years": (1, 20),
        "age_min": 30,
    },
    "ckd_stage3": {
        "display": "Chronic kidney disease stage 3",
        "snomed": "433144002",
        "icd10": "N18.3",
        "onset_years": (1, 10),
        "age_min": 45,
    },
    "obesity": {
        "display": "Obesity",
        "snomed": "414916001",
        "icd10": "E66.9",
        "onset_years": (2, 30),
        "age_min": 18,
    },
    "cad": {
        "display": "Coronary artery disease",
        "snomed": "53741008",
        "icd10": "I25.10",
        "onset_years": (1, 15),
        "age_min": 45,
    },
    "heart_failure": {
        "display": "Heart failure",
        "snomed": "84114007",
        "icd10": "I50.9",
        "onset_years": (1, 8),
        "age_min": 50,
    },
    "afib": {
        "display": "Atrial fibrillation",
        "snomed": "49436004",
        "icd10": "I48.91",
        "onset_years": (1, 10),
        "age_min": 50,
    },
    "asthma": {
        "display": "Asthma",
        "snomed": "195967001",
        "icd10": "J45.909",
        "onset_years": (5, 40),
        "age_min": 5,
    },
    "copd": {
        "display": "Chronic obstructive pulmonary disease",
        "snomed": "13645005",
        "icd10": "J44.1",
        "onset_years": (1, 15),
        "age_min": 45,
    },
    "hypothyroidism": {
        "display": "Hypothyroidism",
        "snomed": "40930008",
        "icd10": "E03.9",
        "onset_years": (1, 20),
        "age_min": 30,
    },
    "depression": {
        "display": "Major depressive disorder",
        "snomed": "35489007",
        "icd10": "F32.9",
        "onset_years": (1, 30),
        "age_min": 18,
    },
    "anxiety": {
        "display": "Generalized anxiety disorder",
        "snomed": "21897009",
        "icd10": "F41.1",
        "onset_years": (1, 25),
        "age_min": 18,
    },
    "osteoarthritis": {
        "display": "Osteoarthritis",
        "snomed": "396275006",
        "icd10": "M19.90",
        "onset_years": (1, 20),
        "age_min": 45,
    },
    "gerd": {
        "display": "Gastroesophageal reflux disease",
        "snomed": "235595009",
        "icd10": "K21.0",
        "onset_years": (1, 20),
        "age_min": 25,
    },
    "anemia": {
        "display": "Anemia",
        "snomed": "271737000",
        "icd10": "D64.9",
        "onset_years": (1, 10),
        "age_min": 18,
    },
    "sleep_apnea": {
        "display": "Obstructive sleep apnea",
        "snomed": "78275009",
        "icd10": "G47.33",
        "onset_years": (1, 15),
        "age_min": 30,
    },
    "neuropathy": {
        "display": "Peripheral neuropathy",
        "snomed": "302226006",
        "icd10": "G60.9",
        "onset_years": (1, 10),
        "age_min": 40,
    },
    "ckd_stage4": {
        "display": "Chronic kidney disease stage 4",
        "snomed": "431855005",
        "icd10": "N18.4",
        "onset_years": (1, 5),
        "age_min": 50,
    },
    "esrd": {
        "display": "End-stage renal disease",
        "snomed": "46177005",
        "icd10": "N18.6",
        "onset_years": (1, 5),
        "age_min": 50,
    },
    "breast_cancer": {
        "display": "Breast cancer",
        "snomed": "254837009",
        "icd10": "C50.919",
        "onset_years": (1, 10),
        "age_min": 35,
    },
    "lung_cancer": {
        "display": "Non-small cell lung cancer",
        "snomed": "254637007",
        "icd10": "C34.90",
        "onset_years": (1, 5),
        "age_min": 45,
    },
    "ra": {
        "display": "Rheumatoid arthritis",
        "snomed": "69896004",
        "icd10": "M06.9",
        "onset_years": (1, 20),
        "age_min": 30,
    },
    "hepatitis_c": {
        "display": "Chronic hepatitis C",
        "snomed": "128302006",
        "icd10": "B18.2",
        "onset_years": (5, 30),
        "age_min": 25,
    },
    "hiv": {
        "display": "Human immunodeficiency virus infection",
        "snomed": "86406008",
        "icd10": "B20",
        "onset_years": (1, 20),
        "age_min": 18,
    },
    "mi_history": {
        "display": "History of myocardial infarction",
        "snomed": "22298006",
        "icd10": "I25.2",
        "onset_years": (1, 15),
        "age_min": 40,
    },
    "stroke_history": {
        "display": "History of stroke",
        "snomed": "230690007",
        "icd10": "Z86.73",
        "onset_years": (1, 10),
        "age_min": 45,
    },
    "osteoporosis": {
        "display": "Osteoporosis",
        "snomed": "64859006",
        "icd10": "M81.0",
        "onset_years": (1, 15),
        "age_min": 55,
    },
    "back_pain": {
        "display": "Chronic low back pain",
        "snomed": "279039007",
        "icd10": "M54.5",
        "onset_years": (1, 20),
        "age_min": 25,
    },
}

# ---------------------------------------------------------------------------
# Comorbidity clusters — realistic co-occurrence patterns
# ---------------------------------------------------------------------------
COMORBIDITY_CLUSTERS = {
    "metabolic_syndrome": {
        "core": ["type2_diabetes"],
        "likely": ["hypertension", "hyperlipidemia", "obesity"],
        "possible": ["neuropathy", "sleep_apnea", "gerd", "ckd_stage3"],
    },
    "cardiovascular": {
        "core": ["cad"],
        "likely": ["hypertension", "hyperlipidemia"],
        "possible": ["heart_failure", "afib", "mi_history", "type2_diabetes"],
    },
    "renal": {
        "core": ["ckd_stage3"],
        "likely": ["hypertension", "anemia"],
        "possible": ["type2_diabetes", "ckd_stage4"],
    },
    "pulmonary": {
        "core": ["copd"],
        "likely": ["hypertension"],
        "possible": ["asthma", "sleep_apnea", "heart_failure"],
    },
    "psychiatric": {
        "core": ["depression"],
        "likely": ["anxiety"],
        "possible": ["sleep_apnea", "back_pain", "gerd"],
    },
    "general_elderly": {
        "core": ["hypertension"],
        "likely": ["hyperlipidemia", "osteoarthritis"],
        "possible": ["hypothyroidism", "osteoporosis", "back_pain", "gerd"],
    },
}

# ---------------------------------------------------------------------------
# Medication definitions with RxNorm codes
# ---------------------------------------------------------------------------
MEDICATIONS = {
    "metformin":       {"display": "Metformin 1000 mg oral tablet", "rxnorm": "860974",  "dose": "1000 mg", "route": "oral", "frequency": "twice daily"},
    "metformin_500":   {"display": "Metformin 500 mg oral tablet",  "rxnorm": "860971",  "dose": "500 mg",  "route": "oral", "frequency": "twice daily"},
    "glipizide":       {"display": "Glipizide 5 mg oral tablet",    "rxnorm": "310489",  "dose": "5 mg",    "route": "oral", "frequency": "once daily"},
    "sitagliptin":     {"display": "Sitagliptin 100 mg oral tablet","rxnorm": "593411",  "dose": "100 mg",  "route": "oral", "frequency": "once daily"},
    "insulin_glargine":{"display": "Insulin glargine 100 units/mL", "rxnorm": "274783",  "dose": "20 units","route": "subcutaneous", "frequency": "once daily at bedtime"},
    "insulin_lispro":  {"display": "Insulin lispro 100 units/mL",   "rxnorm": "1160696", "dose": "varies",  "route": "subcutaneous", "frequency": "with meals"},
    "lisinopril":      {"display": "Lisinopril 10 mg oral tablet",  "rxnorm": "29046",   "dose": "10 mg",   "route": "oral", "frequency": "once daily"},
    "lisinopril_20":   {"display": "Lisinopril 20 mg oral tablet",  "rxnorm": "104375",  "dose": "20 mg",   "route": "oral", "frequency": "once daily"},
    "amlodipine":      {"display": "Amlodipine 5 mg oral tablet",   "rxnorm": "17767",   "dose": "5 mg",    "route": "oral", "frequency": "once daily"},
    "losartan":        {"display": "Losartan 50 mg oral tablet",    "rxnorm": "52175",   "dose": "50 mg",   "route": "oral", "frequency": "once daily"},
    "metoprolol":      {"display": "Metoprolol succinate 25 mg",    "rxnorm": "866426",  "dose": "25 mg",   "route": "oral", "frequency": "once daily"},
    "carvedilol":      {"display": "Carvedilol 6.25 mg oral tablet","rxnorm": "20352",   "dose": "6.25 mg", "route": "oral", "frequency": "twice daily"},
    "atorvastatin":    {"display": "Atorvastatin 40 mg oral tablet","rxnorm": "617310",  "dose": "40 mg",   "route": "oral", "frequency": "once daily at bedtime"},
    "rosuvastatin":    {"display": "Rosuvastatin 20 mg oral tablet","rxnorm": "301542",  "dose": "20 mg",   "route": "oral", "frequency": "once daily"},
    "furosemide":      {"display": "Furosemide 40 mg oral tablet",  "rxnorm": "202991",  "dose": "40 mg",   "route": "oral", "frequency": "once daily"},
    "spironolactone":  {"display": "Spironolactone 25 mg oral tablet","rxnorm":"9997",   "dose": "25 mg",   "route": "oral", "frequency": "once daily"},
    "aspirin":         {"display": "Aspirin 81 mg oral tablet",     "rxnorm": "1191",    "dose": "81 mg",   "route": "oral", "frequency": "once daily"},
    "warfarin":        {"display": "Warfarin 5 mg oral tablet",     "rxnorm": "11289",   "dose": "5 mg",    "route": "oral", "frequency": "once daily"},
    "apixaban":        {"display": "Apixaban 5 mg oral tablet",     "rxnorm": "1364435", "dose": "5 mg",    "route": "oral", "frequency": "twice daily"},
    "omeprazole":      {"display": "Omeprazole 20 mg oral capsule", "rxnorm": "40790",   "dose": "20 mg",   "route": "oral", "frequency": "once daily before breakfast"},
    "levothyroxine":   {"display": "Levothyroxine 50 mcg oral tablet","rxnorm":"10582",  "dose": "50 mcg",  "route": "oral", "frequency": "once daily on empty stomach"},
    "sertraline":      {"display": "Sertraline 50 mg oral tablet",  "rxnorm": "36437",   "dose": "50 mg",   "route": "oral", "frequency": "once daily"},
    "escitalopram":    {"display": "Escitalopram 10 mg oral tablet","rxnorm": "596926",  "dose": "10 mg",   "route": "oral", "frequency": "once daily"},
    "gabapentin":      {"display": "Gabapentin 300 mg oral capsule","rxnorm": "310431",  "dose": "300 mg",  "route": "oral", "frequency": "three times daily"},
    "pregabalin":      {"display": "Pregabalin 75 mg oral capsule", "rxnorm": "187832",  "dose": "75 mg",   "route": "oral", "frequency": "twice daily"},
    "albuterol":       {"display": "Albuterol 90 mcg/actuation inhaler","rxnorm":"745752","dose":"2 puffs","route": "inhalation", "frequency": "as needed"},
    "tiotropium":      {"display": "Tiotropium 18 mcg inhalation",  "rxnorm": "704459",  "dose": "18 mcg",  "route": "inhalation", "frequency": "once daily"},
    "fluticasone":     {"display": "Fluticasone 250 mcg/salmeterol 50 mcg","rxnorm":"896790","dose":"1 puff","route":"inhalation","frequency":"twice daily"},
    "allopurinol":     {"display": "Allopurinol 300 mg oral tablet","rxnorm": "6940",    "dose": "300 mg",  "route": "oral", "frequency": "once daily"},
    "alendronate":     {"display": "Alendronate 70 mg oral tablet", "rxnorm": "41493",   "dose": "70 mg",   "route": "oral", "frequency": "once weekly"},
    "hydroxychloroquine":{"display":"Hydroxychloroquine 200 mg oral tablet","rxnorm":"5521","dose":"200 mg","route":"oral","frequency":"twice daily"},
}

CONDITION_MEDICATIONS = {
    "type2_diabetes":   ["metformin", "glipizide", "sitagliptin"],
    "hypertension":     ["lisinopril", "amlodipine", "losartan", "metoprolol"],
    "hyperlipidemia":   ["atorvastatin", "rosuvastatin"],
    "ckd_stage3":       ["lisinopril", "furosemide"],
    "ckd_stage4":       ["lisinopril_20", "furosemide", "spironolactone"],
    "obesity":          ["metformin_500"],
    "cad":              ["aspirin", "atorvastatin", "metoprolol"],
    "heart_failure":    ["furosemide", "carvedilol", "lisinopril", "spironolactone"],
    "afib":             ["apixaban", "metoprolol"],
    "mi_history":       ["aspirin", "atorvastatin", "metoprolol"],
    "asthma":           ["albuterol", "fluticasone"],
    "copd":             ["tiotropium", "albuterol"],
    "hypothyroidism":   ["levothyroxine"],
    "depression":       ["sertraline", "escitalopram"],
    "anxiety":          ["escitalopram", "sertraline"],
    "gerd":             ["omeprazole"],
    "neuropathy":       ["gabapentin", "pregabalin"],
    "ra":               ["hydroxychloroquine"],
    "osteoporosis":     ["alendronate"],
}

INSULIN_CONDITIONS = {"type2_diabetes"}  # added only in severe/ineligible cases

# ---------------------------------------------------------------------------
# Lab observation definitions with LOINC codes
# ---------------------------------------------------------------------------
OBSERVATIONS = {
    "hba1c": {
        "display": "Hemoglobin A1c/Hemoglobin.total in Blood",
        "loinc": "4548-4",
        "unit": "%",
        "normal": (4.8, 5.7),
        "prediabetes": (5.7, 6.5),
        "diabetes": (6.5, 13.0),
    },
    "egfr": {
        "display": "Glomerular filtration rate/1.73 sq M.predicted",
        "loinc": "33914-3",
        "unit": "mL/min/1.73m2",
        "normal": (60, 120),
        "ckd3": (30, 59),
        "ckd4": (15, 29),
        "esrd": (5, 14),
    },
    "creatinine": {
        "display": "Creatinine [Mass/volume] in Serum or Plasma",
        "loinc": "2160-0",
        "unit": "mg/dL",
        "normal_m": (0.74, 1.18),
        "normal_f": (0.53, 1.02),
        "elevated": (1.5, 5.0),
    },
    "hemoglobin": {
        "display": "Hemoglobin [Mass/volume] in Blood",
        "loinc": "718-7",
        "unit": "g/dL",
        "normal_m": (13.5, 17.5),
        "normal_f": (12.0, 15.5),
        "anemia": (7.0, 11.9),
    },
    "wbc": {
        "display": "Leukocytes [#/volume] in Blood by Automated count",
        "loinc": "6690-2",
        "unit": "10*3/uL",
        "normal": (4.5, 11.0),
    },
    "platelets": {
        "display": "Platelets [#/volume] in Blood by Automated count",
        "loinc": "777-3",
        "unit": "10*3/uL",
        "normal": (150, 400),
    },
    "glucose": {
        "display": "Glucose [Mass/volume] in Blood",
        "loinc": "2345-7",
        "unit": "mg/dL",
        "normal": (70, 99),
        "diabetes": (126, 350),
    },
    "alt": {
        "display": "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
        "loinc": "1742-6",
        "unit": "U/L",
        "normal": (7, 40),
        "elevated": (41, 200),
    },
    "ast": {
        "display": "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
        "loinc": "1920-8",
        "unit": "U/L",
        "normal": (10, 40),
        "elevated": (41, 200),
    },
    "ldl": {
        "display": "Cholesterol in LDL [Mass/volume] in Serum or Plasma",
        "loinc": "18262-6",
        "unit": "mg/dL",
        "optimal": (50, 100),
        "borderline": (100, 160),
        "high": (160, 250),
    },
    "hdl": {
        "display": "Cholesterol in HDL [Mass/volume] in Serum or Plasma",
        "loinc": "2085-9",
        "unit": "mg/dL",
        "low": (20, 40),
        "normal": (40, 80),
    },
    "sodium": {
        "display": "Sodium [Moles/volume] in Serum or Plasma",
        "loinc": "2951-2",
        "unit": "mmol/L",
        "normal": (136, 145),
    },
    "potassium": {
        "display": "Potassium [Moles/volume] in Serum or Plasma",
        "loinc": "2823-3",
        "unit": "mmol/L",
        "normal": (3.5, 5.0),
    },
    "systolic_bp": {
        "display": "Systolic blood pressure",
        "loinc": "8480-6",
        "unit": "mmHg",
        "normal": (100, 120),
        "elevated": (121, 139),
        "hypertension": (140, 180),
    },
    "diastolic_bp": {
        "display": "Diastolic blood pressure",
        "loinc": "8462-4",
        "unit": "mmHg",
        "normal": (60, 80),
        "elevated": (81, 89),
        "hypertension": (90, 110),
    },
    "bmi": {
        "display": "Body mass index (BMI) [Ratio]",
        "loinc": "39156-5",
        "unit": "kg/m2",
        "normal": (18.5, 24.9),
        "overweight": (25.0, 29.9),
        "obese": (30.0, 45.0),
    },
    "heart_rate": {
        "display": "Heart rate",
        "loinc": "8867-4",
        "unit": "/min",
        "normal": (55, 90),
        "elevated": (91, 120),
    },
}

COMMON_ALLERGIES = [
    {"display": "Penicillin", "snomed": "372687004"},
    {"display": "Sulfonamides", "snomed": "387406002"},
    {"display": "Aspirin", "snomed": "387458008"},
    {"display": "Latex", "snomed": "1003755004"},
    {"display": "Codeine", "snomed": "387494000"},
    {"display": "Ibuprofen", "snomed": "387207008"},
    {"display": "Contrast dye", "snomed": "406845001"},
]


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _rand_date_before(years_ago_min: float, years_ago_max: float) -> str:
    days_min = int(years_ago_min * 365)
    days_max = int(years_ago_max * 365)
    days = random.randint(days_min, days_max)
    return (date.today() - timedelta(days=days)).isoformat()


def _make_patient_resource(pid: str, given: str, family: str, birth_date: str, gender: str) -> dict:
    return {
        "resourceType": "Patient",
        "id": pid,
        "meta": {"profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"]},
        "name": [{"use": "official", "family": family, "given": [given]}],
        "gender": gender,
        "birthDate": birth_date,
        "address": [{"use": "home", "country": "US", "state": random.choice(["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"])}],
        "communication": [{"language": {"coding": [{"system": "urn:ietf:bcp:47", "code": "en-US"}]}, "preferred": True}],
    }


def _make_condition(patient_id: str, cond_key: str, onset_date: str) -> dict:
    c = CONDITIONS[cond_key]
    return {
        "resourceType": "Condition",
        "id": str(uuid.uuid4()),
        "clinicalStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active", "display": "Active"}]
        },
        "verificationStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]
        },
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
        "code": {
            "coding": [
                {"system": "http://snomed.info/sct", "code": c["snomed"], "display": c["display"]},
                {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": c["icd10"]},
            ],
            "text": c["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "onsetDateTime": onset_date,
        "recordedDate": onset_date,
    }


def _make_medication_request(patient_id: str, med_key: str, start_date: str) -> dict:
    m = MEDICATIONS[med_key]
    return {
        "resourceType": "MedicationRequest",
        "id": str(uuid.uuid4()),
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": m["rxnorm"], "display": m["display"]}],
            "text": m["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "authoredOn": start_date,
        "dosageInstruction": [{
            "text": f"{m['dose']} {m['frequency']}",
            "route": {"coding": [{"display": m["route"]}]},
        }],
    }


def _make_observation(patient_id: str, obs_key: str, value: float, obs_date: str) -> dict:
    o = OBSERVATIONS[obs_key]
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
        "code": {
            "coding": [{"system": "http://loinc.org", "code": o["loinc"], "display": o["display"]}],
            "text": o["display"],
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": obs_date + "T08:00:00Z",
        "valueQuantity": {
            "value": round(value, 2),
            "unit": o["unit"],
            "system": "http://unitsofmeasure.org",
            "code": o["unit"],
        },
    }


def _make_allergy(patient_id: str, allergy: dict) -> dict:
    return {
        "resourceType": "AllergyIntolerance",
        "id": str(uuid.uuid4()),
        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
        "type": "allergy",
        "category": ["medication"],
        "criticality": random.choice(["low", "high"]),
        "code": {
            "coding": [{"system": "http://snomed.info/sct", "code": allergy["snomed"], "display": allergy["display"]}],
            "text": allergy["display"],
        },
        "patient": {"reference": f"Patient/{patient_id}"},
        "recordedDate": _rand_date_before(1, 20),
    }


def _build_bundle(resources: list[dict]) -> dict:
    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": date.today().isoformat() + "T00:00:00Z",
        "meta": {
            "tag": [
                {
                    "code": "synthetic-python",
                    "display": "Generated by Python fallback (not Synthea)",
                }
            ]
        },
        "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in resources],
    }


# ---------------------------------------------------------------------------
# Lab value generator — correlated with conditions
# ---------------------------------------------------------------------------

def _generate_labs(cond_keys: set[str], gender: str, age: int, profile: str) -> list[tuple[str, float, str]]:
    """Returns list of (obs_key, value, date_str) tuples for longitudinal history."""
    results = []

    # Generate 3–5 time points over the past 24 months
    timepoints = sorted(random.sample(range(30, 730), k=random.randint(3, 5)))
    most_recent = timepoints[-1]

    for days_ago in timepoints:
        obs_date = (date.today() - timedelta(days=days_ago)).isoformat()

        # HbA1c
        if "type2_diabetes" in cond_keys:
            if profile == "ineligible":
                base = random.uniform(9.0, 13.0)
            elif profile == "eligible":
                base = random.uniform(7.5, 9.5)
            else:
                base = random.uniform(6.5, 10.0)
            # slight trend toward control over time (more recent = slightly lower)
            trend = (days_ago / 730) * random.uniform(0, 1.5)
            results.append(("hba1c", base + trend, obs_date))
        else:
            results.append(("hba1c", random.uniform(4.8, 5.9), obs_date))

        # eGFR — correlated with CKD stage
        if "esrd" in cond_keys:
            results.append(("egfr", random.uniform(5, 14), obs_date))
        elif "ckd_stage4" in cond_keys:
            results.append(("egfr", random.uniform(15, 29), obs_date))
        elif "ckd_stage3" in cond_keys:
            results.append(("egfr", random.uniform(30, 59), obs_date))
        else:
            results.append(("egfr", random.uniform(65, 120), obs_date))

        # Creatinine — correlated with eGFR
        if "ckd_stage4" in cond_keys or "esrd" in cond_keys:
            cr = random.uniform(2.0, 6.0)
        elif "ckd_stage3" in cond_keys:
            cr = random.uniform(1.3, 2.5)
        else:
            lo, hi = (0.74, 1.18) if gender == "male" else (0.53, 1.02)
            cr = random.uniform(lo, hi)
        results.append(("creatinine", cr, obs_date))

        # Hemoglobin — lower with CKD/anemia
        if "anemia" in cond_keys or "ckd_stage4" in cond_keys or "esrd" in cond_keys:
            hgb = random.uniform(7.5, 11.5)
        elif "ckd_stage3" in cond_keys:
            hgb = random.uniform(10.0, 13.0)
        else:
            lo, hi = (13.5, 17.5) if gender == "male" else (12.0, 15.5)
            hgb = random.uniform(lo, hi)
        results.append(("hemoglobin", hgb, obs_date))

        # WBC — normal unless infection context
        results.append(("wbc", random.uniform(4.5, 10.5), obs_date))

        # Platelets
        results.append(("platelets", random.uniform(150, 400), obs_date))

        # Glucose
        if "type2_diabetes" in cond_keys:
            results.append(("glucose", random.uniform(110, 280), obs_date))
        else:
            results.append(("glucose", random.uniform(72, 99), obs_date))

        # LDL — hyperlipidemia or statin-treated
        if "hyperlipidemia" in cond_keys and "atorvastatin" not in cond_keys:
            results.append(("ldl", random.uniform(120, 220), obs_date))
        else:
            results.append(("ldl", random.uniform(55, 110), obs_date))

        # HDL
        results.append(("hdl", random.uniform(35, 75), obs_date))

        # Liver enzymes
        results.append(("alt", random.uniform(10, 45), obs_date))
        results.append(("ast", random.uniform(12, 40), obs_date))

        # Electrolytes
        results.append(("sodium", random.uniform(137, 144), obs_date))
        results.append(("potassium", random.uniform(3.6, 4.8), obs_date))

    # Vitals — use most recent timepoint date
    vitals_date = (date.today() - timedelta(days=most_recent)).isoformat()

    if "hypertension" in cond_keys and profile == "ineligible":
        systolic = random.uniform(150, 185)
        diastolic = random.uniform(92, 110)
    elif "hypertension" in cond_keys:
        systolic = random.uniform(130, 155)
        diastolic = random.uniform(82, 95)
    else:
        systolic = random.uniform(100, 128)
        diastolic = random.uniform(62, 82)
    results.append(("systolic_bp", systolic, vitals_date))
    results.append(("diastolic_bp", diastolic, vitals_date))

    if "obesity" in cond_keys:
        bmi = random.uniform(30.5, 48.0)
    elif age > 50:
        bmi = random.uniform(22.0, 32.0)
    else:
        bmi = random.uniform(19.0, 27.0)
    results.append(("bmi", bmi, vitals_date))

    if "afib" in cond_keys or "heart_failure" in cond_keys:
        results.append(("heart_rate", random.uniform(85, 115), vitals_date))
    else:
        results.append(("heart_rate", random.uniform(58, 88), vitals_date))

    return results


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def _select_conditions(profile: str, age: int) -> set[str]:
    eligible_age_conds = {k for k, v in CONDITIONS.items() if age >= v.get("age_min", 0)}

    if profile == "eligible":
        # Metabolic syndrome core — designed to pass most DM trial inclusion criteria
        conds = {"type2_diabetes", "hypertension", "hyperlipidemia"}
        if age > 50 and random.random() < 0.4:
            conds.add("obesity")
        if age > 55 and random.random() < 0.25:
            conds.add("neuropathy")
        if random.random() < 0.3:
            conds.add("gerd")

    elif profile == "ineligible":
        # Designed to fail common DM trial exclusion criteria
        base = random.choice(["renal", "insulin_user"])
        if base == "renal":
            conds = {"type2_diabetes", "ckd_stage3", "hypertension", "anemia"}
            if random.random() < 0.4:
                conds.add("ckd_stage4")
        else:
            conds = {"type2_diabetes", "hypertension", "heart_failure"}
        if random.random() < 0.3:
            conds.add("stroke_history")

    else:  # ambiguous
        cluster_name = random.choice(list(COMORBIDITY_CLUSTERS.keys()))
        cluster = COMORBIDITY_CLUSTERS[cluster_name]
        conds = set()
        for c in cluster["core"]:
            if c in eligible_age_conds:
                conds.add(c)
        for c in cluster["likely"]:
            if c in eligible_age_conds and random.random() < 0.7:
                conds.add(c)
        for c in cluster["possible"]:
            if c in eligible_age_conds and random.random() < 0.35:
                conds.add(c)
        # add 1–2 random independent conditions
        extras = list(eligible_age_conds - conds)
        if extras:
            for c in random.sample(extras, min(2, len(extras))):
                if random.random() < 0.3:
                    conds.add(c)

    return conds & eligible_age_conds


def _select_medications(cond_keys: set[str], profile: str) -> list[str]:
    meds: set[str] = set()

    for cond in cond_keys:
        options = CONDITION_MEDICATIONS.get(cond, [])
        if options:
            # pick 1–2 medications per condition
            chosen = random.sample(options, min(len(options), random.randint(1, 2)))
            meds.update(chosen)

    # Ineligible patients may be on insulin
    if profile == "ineligible" and "type2_diabetes" in cond_keys:
        meds.add("insulin_glargine")
        if random.random() < 0.4:
            meds.add("insulin_lispro")
        meds.discard("glipizide")  # usually not combined with insulin in simple regimens

    # Aspirin for cardiovascular conditions
    if any(c in cond_keys for c in ["cad", "mi_history", "stroke_history"]) and "aspirin" not in meds:
        meds.add("aspirin")

    return list(meds)


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SyntheticPatientGenerator:
    def generate_patients(self, count: int, seed: int = 42) -> list[dict]:
        random.seed(seed)
        logger.info("Generating {} realistic synthetic FHIR R4 patient bundles...", count)

        bundles = []
        eligible_target = math.floor(count * 0.30)
        ineligible_target = math.floor(count * 0.30)
        eligible_count = ineligible_count = 0

        for i in range(count):
            if i % 10 == 0 and i > 0:
                logger.info("  Generated {}/{} patients...", i, count)

            # Assign profile
            if eligible_count < eligible_target:
                profile = "eligible"
                eligible_count += 1
            elif ineligible_count < ineligible_target:
                profile = "ineligible"
                ineligible_count += 1
            else:
                profile = "ambiguous"

            # Demographics
            gender = random.choice(["male", "female"])
            given = random.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
            family = random.choice(LAST_NAMES)

            # Age stratified: 20% young (18-40), 50% middle (41-65), 30% elderly (66-85)
            age_roll = random.random()
            if age_roll < 0.20:
                age = random.randint(18, 40)
            elif age_roll < 0.70:
                age = random.randint(41, 65)
            else:
                age = random.randint(66, 85)

            birth_year = date.today().year - age
            birth_date = f"{birth_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            pid = f"PT-{i + 1:04d}"

            # Conditions
            cond_keys = _select_conditions(profile, age)
            if not cond_keys:
                cond_keys = {"hypertension"}

            # Medications
            med_keys = _select_medications(cond_keys, profile)

            # Labs
            labs = _generate_labs(cond_keys, gender, age, profile)

            # Build FHIR resources
            resources = [_make_patient_resource(pid, given, family, birth_date, gender)]

            for ck in cond_keys:
                c_def = CONDITIONS[ck]
                onset_min, onset_max = c_def["onset_years"]
                onset_years = random.uniform(onset_min, min(onset_max, age - c_def.get("age_min", 0) + 1))
                onset_date = _rand_date_before(max(0.5, onset_years * 0.8), onset_years)
                resources.append(_make_condition(pid, ck, onset_date))

            for mk in med_keys:
                start_date = _rand_date_before(0.5, 5)
                resources.append(_make_medication_request(pid, mk, start_date))

            for obs_key, value, obs_date in labs:
                resources.append(_make_observation(pid, obs_key, value, obs_date))

            # Allergies — 30% of patients have at least one
            if random.random() < 0.30:
                n_allergies = random.randint(1, 2)
                for allergy in random.sample(COMMON_ALLERGIES, n_allergies):
                    resources.append(_make_allergy(pid, allergy))

            bundles.append(_build_bundle(resources))

        ambiguous_count = count - eligible_count - ineligible_count
        logger.info(
            "✓ Generation complete — {} eligible, {} ineligible, {} ambiguous | avg resources/patient: {}",
            eligible_count, ineligible_count, ambiguous_count,
            round(sum(len(b["entry"]) for b in bundles) / count, 1),
        )
        return bundles


synthea_generator = SyntheticPatientGenerator()
