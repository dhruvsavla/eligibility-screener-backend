# Setting Up Real Synthea (MITRE)

The eligibility screener supports both real Synthea and a Python fallback generator.
Real Synthea produces more clinically accurate patient data. The Python fallback
works identically for development and testing.

## Prerequisites
- Java 11 or higher: https://adoptium.net/

## Download Synthea

```bash
mkdir -p backend/synthea
cd backend/synthea
curl -L -o synthea.jar \
  https://github.com/synthetichealth/synthea/releases/latest/download/synthea-with-dependencies.jar
```

## Verify

```bash
java -jar backend/synthea/synthea.jar --help
```

## First run (generates 10 patients in Massachusetts with diabetes focus)

```bash
java -jar backend/synthea/synthea.jar \
  -p 10 --seed 42 \
  --exporter.fhir.export=true \
  -m diabetes \
  Massachusetts
```

Output FHIR R4 bundles appear in: `backend/synthea/output/fhir/`

## Notes

- First run downloads modules (~30s). Subsequent runs are faster.
- Use `-p 500` to generate 500 patients (takes ~2-5 minutes).
- Add `-m diabetes` to bias toward diabetic patients matching your trials.
- The `--seed` flag makes generation reproducible.
- Patient bundles are placed in `output/fhir/*.json`, one file per patient.
- Generated bundles are full longitudinal records including Encounters, Procedures,
  and Immunizations — significantly more detailed than the Python fallback.

## If Synthea is not set up

The application automatically falls back to the Python generator (`synthea_generator.py`).
All downstream components (FHIR parser, scoring engine) work identically with both.
A banner in the Patients page indicates which generator was used.

## Differences from Python fallback

| Feature                | Real Synthea          | Python Fallback         |
|------------------------|-----------------------|-------------------------|
| Condition vocabulary   | Thousands of modules  | ~25 conditions          |
| Longitudinal care      | Full Encounter history | No encounters           |
| Physiological modeling | Yes (correlated labs) | Statistical ranges only |
| Medications            | Evidence-based        | Condition-mapped        |
| Generation time        | 2-5 min / 500 pts     | <1s / 500 pts           |
| Setup required         | Java + JAR download   | None (built-in)         |
