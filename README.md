# Automated Eligibility Screener — Backend

FastAPI backend for automated clinical trial eligibility screening. Fetches trials
from ClinicalTrials.gov, extracts structured criteria via a **LangChain document
agent** powered by **Claude Sonnet** (with **scispaCy** clinical NER), parses FHIR R4
patient bundles, and scores each patient against trial criteria using an
**OMOP/SNOMED-CT**–indexed semantic matching engine with hierarchy traversal.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM | Anthropic **Claude Sonnet** (`claude-sonnet-4-6`) — criteria extraction + rationale |
| Agent | **LangChain** document agent (`langchain-anthropic`, 4 tools) |
| Clinical NER | **scispaCy** (`en_core_sci_sm`, fallback `en_core_web_sm`) |
| Semantic search | FAISS + sentence-transformers (all-MiniLM-L6-v2) |
| Vocabulary | **OMOP** (Athena) with **SNOMED-CT hierarchy** (CONCEPT_ANCESTOR), LOINC, RxNorm |
| Patient data | **Synthea** (MITRE) real FHIR R4 + Python fallback |
| Database | SQLite via aiosqlite |
| Runtime | Python 3.11+ |

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install scispaCy clinical NER model (required for clinical entity recognition)
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
#    Fallback if the above fails:
python -m spacy download en_core_web_sm

# 4. Configure environment
cp .env.example .env
# Open .env and set your ANTHROPIC_API_KEY

# 5. (Optional but recommended) Download Synthea + OMOP — see SETUP.md files below

# 6. Start the server
uvicorn app.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

## Required Tools & Setup Guides

| Tool | Purpose | Required | Guide |
|---|---|---|---|
| Anthropic API key | Claude Sonnet LLM | **Yes** | set `ANTHROPIC_API_KEY` in `.env` |
| scispaCy model | clinical NER | Yes (falls back to en_core_web_sm) | command above |
| Synthea JAR + Java 11+ | 500 real patients | for `/generate-500` | [synthea/SETUP.md](synthea/SETUP.md) |
| OMOP CONCEPT.csv + CONCEPT_ANCESTOR.csv | SNOMED hierarchy | for hierarchy matching | [omop/SETUP.md](omop/SETUP.md) |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | — | Anthropic API key (Claude Sonnet access) |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Claude model id |
| `DATABASE_URL` | No | `sqlite:///./eligibility.db` | SQLite path |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `CORS_ORIGINS` | No | `["http://localhost:5173"]` | Allowed CORS origins as JSON array |
| `SYNTHEA_JAR_PATH` | No | `backend/synthea/synthea.jar` | Path to the Synthea JAR |
| `OMOP_DIR` | No | `backend/omop` | Directory holding OMOP CSV files |

> **Never commit `.env`** — it is listed in `.gitignore`. Use `.env.example` as the template.

## Two Accuracy Measures

- **Measure 1 — Criteria Extraction Accuracy:** hand-label gold rules on the
  Annotation page, then `POST /api/protocols/{id}/extraction-accuracy` compares
  Claude's extraction against the gold standard (precision / recall / F1).
- **Measure 2 — Screening Sensitivity:** `POST /api/evaluation/build-ground-truth`
  then `POST /api/evaluation/run` scores 100 ground-truth patients built against the
  flagship protocol's real thresholds (Option B sensitivity, target ≥85%).

## First-Use Sequence

```
a. POST /api/protocols/seed-all          # seeds 10 protocols, flags diabetes flagship
b. Annotation page → flagship → hand-label gold rules → Save
c. POST /api/protocols/{flagship}/extraction-accuracy   # MEASURE 1
d. POST /api/patients/generate-500       # real Synthea
e. POST /api/evaluation/build-ground-truth {protocol_id: flagship}
f. POST /api/evaluation/run {protocol_id: flagship}      # MEASURE 2
g. GET  /api/evaluation/report/{flagship}/html
   Verify all components green: GET /api/health
```

## Project Structure

```
app/
  config.py              # Pydantic settings — reads .env
  database.py            # SQLite schema + async query helpers
  main.py                # FastAPI app, CORS, request logging, lifespan
  data/seed_protocols.py # 10-protocol seeder + flagship flag
  models/                # Pydantic data models
  routers/
    health.py            # 8-component system status
    patients.py          # incl. /generate-500 (real Synthea)
    protocols.py         # incl. /seed-all, gold annotations, extraction-accuracy
    screening.py
    evaluation.py
  services/
    llm_client.py            # shared Claude Sonnet client (all LLM calls route here)
    langchain_agent.py       # LangChain document agent (4 tools)
    ner_service.py           # scispaCy clinical NER
    criteria_extractor.py    # Claude Sonnet structured criteria extraction
    concept_matcher.py       # OMOP/FAISS matcher + SNOMED-CT hierarchy
    omop_vocabulary.py       # OMOP loader + is_a() hierarchy traversal
    extraction_evaluator.py  # Measure 1: extraction precision/recall/F1
    evaluator.py             # Measure 2: flagship-aware ground truth + metrics
    fhir_parser.py           # FHIR R4 Bundle → PatientData
    rationale_generator.py   # Claude Sonnet plain-English rationale cards
    scoring_engine.py        # Rule evaluator, ULN resolution, fit scorer
    synthea_runner.py        # Real MITRE Synthea wrapper
    synthea_generator.py     # Python fallback FHIR R4 generator
    trials_fetcher.py        # ClinicalTrials.gov API v2 client
tests/                       # pytest suite (75 tests)
```

## Scoring Algorithm

- Base score: **100/100**
- Confirmed inclusion criterion **FAIL**: −20 pts; inclusion **AMBIGUOUS**: −5 pts
- Exclusion **triggered**: −25 pts; exclusion **AMBIGUOUS**: −3 pts
- Concept matching: substring **OR** FAISS ≥ 0.72 **OR** SNOMED-CT hierarchy is-a
- Asymmetric confidence band widens with unverifiable criteria
- **ELIGIBLE**: score ≥ 80, inclusion pass-rate ≥ 60%, no ambiguous exclusions
- **REVIEW_NEEDED**: default (counts as TP under Option B sensitivity)
- **INELIGIBLE**: confirmed inclusion FAIL, exclusion triggered, or score < 30

## Running Tests

```bash
pytest tests/ -v
```

## Requirements

- Python 3.11+
- Anthropic API key with Claude Sonnet access
- Java 11+ (for real Synthea)
- ~2 GB RAM (sentence-transformers + FAISS index)
- Internet access (ClinicalTrials.gov API + Anthropic API)
