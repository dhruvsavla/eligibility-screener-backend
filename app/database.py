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
