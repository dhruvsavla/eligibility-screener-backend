# Automated Eligibility Screener — Backend

FastAPI backend for automated clinical trial eligibility screening. Fetches trials from ClinicalTrials.gov, extracts structured criteria via GPT-4o, parses FHIR R4 patient bundles, and scores each patient against trial criteria using a SNOMED CT–indexed semantic matching engine.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o (criteria extraction + rationale) |
| NLP | scispaCy / spaCy NER |
| Semantic search | FAISS + sentence-transformers (all-MiniLM-L6-v2) |
| Patient data | FHIR R4 Bundle (synthetic generator included) |
| Ontology | SNOMED CT (72 indexed concepts), LOINC, RxNorm |
| Database | SQLite via aiosqlite |
| Runtime | Python 3.11+ |

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install spaCy model
#    Try the medical model first; fall back to the general one if the URL is unavailable
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz \
  || python -m spacy download en_core_web_sm

# 4. Configure environment
cp .env.example .env
# Open .env and set your OPENAI_API_KEY

# 5. Start the server
uvicorn app.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key (GPT-4o access required) |
| `DATABASE_URL` | No | `sqlite:///./eligibility.db` | SQLite path |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `CORS_ORIGINS` | No | `["http://localhost:5173"]` | Allowed CORS origins as JSON array |

> **Never commit `.env`** — it is listed in `.gitignore`. Use `.env.example` as the template.

## Project Structure

```
app/
  config.py              # Pydantic settings — reads .env
  database.py            # SQLite schema + async query helpers
  main.py                # FastAPI app, CORS, request logging, lifespan
  models/                # Pydantic data models
  routers/
    health.py
    patients.py
    protocols.py
    screening.py
  services/
    concept_matcher.py       # FAISS SNOMED index builder + searcher
    criteria_extractor.py    # GPT-4o structured criteria extraction
    fhir_parser.py           # FHIR R4 Bundle → PatientData
    ner_service.py           # scispaCy / spaCy NER
    rationale_generator.py   # GPT-4o plain-English rationale cards
    scoring_engine.py        # Rule evaluator, ULN resolution, fit scorer
    synthea_generator.py     # Realistic synthetic FHIR R4 patient generator
    trials_fetcher.py        # ClinicalTrials.gov API v2 client
tests/
  test_criteria_extractor.py
  test_fhir_parser.py
  test_scoring_engine.py
```

## Scoring Algorithm

- Base score: **100/100**
- Confirmed inclusion criterion **FAIL**: −20 pts
- Confirmed exclusion criterion **triggered**: −25 pts
- **AMBIGUOUS** (no patient data available): no score penalty — only widens the confidence band
- Thresholds with `Nx ULN` (e.g. "1.5× ULN") are resolved against standard reference ranges
- **ELIGIBLE**: score ≥ 75 and <50% ambiguous criteria
- **REVIEW_NEEDED**: score ≥ 40 but many unverifiable criteria
- **INELIGIBLE**: any confirmed inclusion FAIL, or score < 40

## Running Tests

```bash
pytest tests/ -v
```

## Requirements

- Python 3.11+
- OpenAI API key with GPT-4o access
- ~2 GB RAM (sentence-transformers + FAISS index)
- Internet access (ClinicalTrials.gov API + OpenAI)
