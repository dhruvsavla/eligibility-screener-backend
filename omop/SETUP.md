# OMOP Vocabulary Setup (Athena)

The screener uses the OMOP vocabulary for SNOMED-CT concept matching AND hierarchy
traversal (so specific patient diagnoses match general protocol criteria).

Without OMOP: the app uses a hardcoded list of 72 SNOMED concepts (works for demos),
and hierarchy traversal is disabled.
With OMOP: concept matching covers the full clinical vocabulary, and the SNOMED-CT
parent/child hierarchy lets a patient coded "essential hypertension" match a rule
written for "hypertensive disorder".

## Download
1. Create a free account at https://athena.ohdsi.org
2. Click "Download" → select vocabularies:
   - 1 (SNOMED) — required, includes the concept hierarchy
   - 17 (RxNorm) — drug concepts
   - 21 (LOINC) — lab concepts
3. Submit. You receive an email with a download link (can take minutes to hours).
4. Unzip the archive.

## Install
```bash
mkdir -p backend/omop
cp /path/to/download/CONCEPT.csv backend/omop/
cp /path/to/download/CONCEPT_ANCESTOR.csv backend/omop/   # REQUIRED for hierarchy
cp /path/to/download/CONCEPT_SYNONYM.csv backend/omop/    # optional
```

## Verify
```bash
wc -l backend/omop/CONCEPT.csv          # ~5-6 million rows
wc -l backend/omop/CONCEPT_ANCESTOR.csv # tens of millions of rows
```

CONCEPT_ANCESTOR.csv is large but essential — it encodes the SNOMED-CT hierarchy
(parent/child relationships). Without it, hierarchy traversal is disabled and the
matcher falls back to flat semantic matching.

## Restart Backend
```bash
uvicorn app.main:app --reload --port 8000
```

On startup you will see:
```
INFO  | === OMOP VOCABULARY CHECK (backend/omop) ===
INFO  |   CONCEPT.csv: ✓ found (498.2 MB)
INFO  |   CONCEPT_ANCESTOR.csv: ✓ found (1203.4 MB)
INFO  | ✓ Loaded 200000 OMOP standard concepts
INFO  | ✓ Loaded N hierarchy edges (M concepts have ancestors)
```

Confirm via the health endpoint:
`GET /api/health` → `components.omop.available` and `components.concept_matcher.hierarchy_active`.

## Notes
- Files are large: CONCEPT.csv ~500 MB, CONCEPT_ANCESTOR.csv ~1 GB+
- The app loads a filtered subset into memory (up to 200K standard concepts)
- Loading takes ~15-30 seconds on first startup; the index is held in memory

## Without OMOP
The app uses a 72-concept hardcoded SNOMED fallback. Hierarchy traversal is
unavailable in fallback mode. The system still functions for demos.
