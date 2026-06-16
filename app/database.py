import aiosqlite
from loguru import logger
from app.config import settings

DB_PATH = settings.DATABASE_URL.replace("sqlite:///", "")

CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS protocols (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nct_id TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL,
        condition TEXT,
        phase TEXT,
        sponsor TEXT,
        raw_criteria_text TEXT,
        agent_trace TEXT,
        is_flagship INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS criterion_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        protocol_id INTEGER NOT NULL REFERENCES protocols(id) ON DELETE CASCADE,
        criterion_text TEXT,
        concept TEXT,
        operator TEXT,
        value TEXT,
        required BOOLEAN DEFAULT TRUE,
        criterion_type TEXT CHECK(criterion_type IN ('inclusion', 'exclusion')),
        snomed_code TEXT,
        confidence REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE NOT NULL,
        name TEXT,
        age INTEGER,
        gender TEXT,
        fhir_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS screening_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL REFERENCES patients(id),
        protocol_id INTEGER NOT NULL REFERENCES protocols(id),
        fit_score INTEGER DEFAULT 0,
        confidence_low INTEGER DEFAULT 0,
        confidence_high INTEGER DEFAULT 100,
        overall_verdict TEXT CHECK(overall_verdict IN ('ELIGIBLE', 'INELIGIBLE', 'REVIEW_NEEDED')),
        rationale_summary TEXT,
        score_breakdown_json TEXT,
        override_verdict TEXT,
        override_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS criterion_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id INTEGER NOT NULL REFERENCES screening_results(id) ON DELETE CASCADE,
        criterion_id INTEGER REFERENCES criterion_rules(id),
        criterion_text TEXT,
        concept TEXT,
        criterion_type TEXT,
        status TEXT CHECK(status IN ('PASS', 'FAIL', 'AMBIGUOUS')),
        explanation TEXT,
        data_found TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gold_annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        protocol_id INTEGER NOT NULL,
        criterion_text TEXT NOT NULL,
        concept TEXT NOT NULL,
        operator TEXT NOT NULL,
        value TEXT,
        required INTEGER NOT NULL,
        criterion_type TEXT NOT NULL,
        annotator TEXT DEFAULT 'human',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (protocol_id) REFERENCES protocols(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS extraction_accuracy_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        protocol_id INTEGER NOT NULL,
        run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        gold_count INTEGER,
        extracted_count INTEGER,
        matched_count INTEGER,
        precision REAL,
        recall REAL,
        f1 REAL,
        details_json TEXT,
        FOREIGN KEY (protocol_id) REFERENCES protocols(id) ON DELETE CASCADE
    )
    """,
]


async def get_db():
    return await aiosqlite.connect(DB_PATH)


async def init_db():
    logger.info("Initializing database at {}", DB_PATH)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        for sql in CREATE_TABLES_SQL:
            await db.execute(sql)
        # Migration: add score_breakdown_json column if it doesn't exist
        try:
            await db.execute(
                "ALTER TABLE screening_results ADD COLUMN score_breakdown_json TEXT"
            )
            logger.info("Migration: added score_breakdown_json column to screening_results")
        except Exception:
            pass  # Column already exists
        # Migration: add is_ground_truth / ground_truth columns if missing
        for col_sql in [
            "ALTER TABLE patients ADD COLUMN is_ground_truth INTEGER DEFAULT 0",
            "ALTER TABLE patients ADD COLUMN ground_truth_verdict TEXT",
            "ALTER TABLE patients ADD COLUMN ground_truth_protocol_id INTEGER",
            # Protocol columns for the LangChain agent trace + flagship flag
            "ALTER TABLE protocols ADD COLUMN agent_trace TEXT",
            "ALTER TABLE protocols ADD COLUMN is_flagship INTEGER DEFAULT 0",
        ]:
            try:
                await db.execute(col_sql)
            except Exception:
                pass
        await db.commit()
    logger.info("✓ Database initialized — all tables ready")


async def fetch_one(query: str, params: tuple = ()):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        async with db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def fetch_all(query: str, params: tuple = ()):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def execute(query: str, params: tuple = ()):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(query, params)
        await db.commit()
        return cursor.lastrowid
