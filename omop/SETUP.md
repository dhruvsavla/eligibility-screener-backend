# Setting Up OMOP Vocabulary (Athena)

The OMOP vocabulary gives the eligibility screener access to 5M+ standardized
medical concepts across SNOMED, LOINC, RxNorm, and ICD-10.

Without OMOP: the app uses a hardcoded list of 80 SNOMED concepts (works for demos).
With OMOP: concept matching covers the full clinical vocabulary.

## Download Steps

1. Create a free account at https://athena.ohdsi.org
2. Click **Download** in the top navigation
3. Select these vocabulary IDs:
   - **1**  — SNOMED CT
   - **17** — RxNorm
   - **21** — LOINC
   - **34** — ICD10CM
4. Click **Download Vocabularies** — you will receive an email with a download link
5. Unzip the downloaded archive

## Install

```bash
mkdir -p backend/omop

# Copy these files from the unzipped archive:
cp /path/to/download/CONCEPT.csv backend/omop/
cp /path/to/download/CONCEPT_SYNONYM.csv backend/omop/
cp /path/to/download/CONCEPT_RELATIONSHIP.csv backend/omop/   # optional but recommended
```

## Verify

```bash
wc -l backend/omop/CONCEPT.csv
# Should show ~5-6 million rows
```

## Restart Backend

```bash
uvicorn app.main:app --reload
```

On startup you will see:
```
INFO  | === OMOP VOCABULARY CHECK ===
INFO  | CONCEPT.csv: ✓ found (498.2 MB)
INFO  | ✓ Loaded 200000 OMOP standard concepts (...)
```

## Notes

- Files are large: CONCEPT.csv ~500 MB, total ~2 GB unzipped
- The app loads a filtered subset into memory (~200K standard concepts, ~500 MB RAM)
- Loading takes ~15-30 seconds on first startup; the index is held in memory
- Without these files, the app uses the built-in 80-concept SNOMED fallback
- The fallback is visible in the health endpoint:
  `GET /api/health` → `components.concept_matcher.mode`

## What OMOP Improves

| Scenario                         | Without OMOP           | With OMOP                     |
|----------------------------------|------------------------|-------------------------------|
| "Glomerulonephritis" concept     | No match               | SNOMED 36171008               |
| "Imatinib" drug match            | No match               | RxNorm 282388                 |
| LOINC lab code resolution        | ~18 common labs        | Full LOINC (~90K entries)     |
| ICD-10 → SNOMED bridging         | Manual only            | CONCEPT_RELATIONSHIP mapping  |
