"""
OMOP vocabulary loader with SNOMED-CT hierarchy support.

Loads from Athena-downloaded OMOP CSV files (backend/omop/):
  CONCEPT.csv               — all concepts (concept_id, name, vocabulary, domain, code)
  CONCEPT_ANCESTOR.csv      — hierarchy (ancestor_concept_id, descendant_concept_id, levels)
  CONCEPT_SYNONYM.csv       — synonyms (optional)
  CONCEPT_RELATIONSHIP.csv  — maps-to/is-a relationships (optional)

The CONCEPT_ANCESTOR table is what provides SNOMED-CT hierarchy: it lets us
determine that "Essential hypertension" IS-A "Hypertensive disorder", so a rule
about "hypertensive disorder" correctly matches a patient coded with the more
specific "essential hypertension".
"""
import csv
import sys
from pathlib import Path
from loguru import logger
from app.config import settings


def _resolve_omop_dir() -> Path:
    """Resolve the OMOP directory regardless of the process working directory.

    Tries, in order: the configured path as-is, that path relative to CWD, and
    the canonical backend/omop directory relative to this source file.
    """
    candidates = [
        Path(settings.OMOP_DIR),
        Path.cwd() / settings.OMOP_DIR,
        Path(__file__).resolve().parent.parent.parent / "omop",
    ]
    for c in candidates:
        if (c / "CONCEPT.csv").exists():
            return c
    # default to the canonical location even if missing (status reporting handles it)
    return Path(__file__).resolve().parent.parent.parent / "omop"


class OMOPFilesNotFoundError(Exception):
    """OMOP vocabulary files are missing from backend/omop/."""


class OMOPLoadError(Exception):
    """Failed to parse OMOP files."""


class OMOPVocabulary:
    def __init__(self):
        self.omop_dir = _resolve_omop_dir()
        self.concepts: dict[int, dict] = {}          # concept_id → concept
        self.name_to_id: dict[str, int] = {}         # lowercased name → concept_id
        self.code_to_id: dict[str, int] = {}         # SNOMED code → concept_id
        self.ancestors: dict[int, set[int]] = {}     # descendant_id → set of ancestor_ids
        self.descendants: dict[int, set[int]] = {}   # ancestor_id → set of descendant_ids
        self.synonyms: dict[int, list[str]] = {}
        self.loaded = False
        self.status = self.check_files()

    def check_files(self) -> dict:
        required = ["CONCEPT.csv"]
        hierarchy = ["CONCEPT_ANCESTOR.csv"]
        optional = ["CONCEPT_SYNONYM.csv", "CONCEPT_RELATIONSHIP.csv"]
        found, missing = [], []
        for f in required + hierarchy + optional:
            (found if (self.omop_dir / f).exists() else missing).append(f)
        omop_available = (self.omop_dir / "CONCEPT.csv").exists()
        hierarchy_available = (self.omop_dir / "CONCEPT_ANCESTOR.csv").exists()
        logger.info("=== OMOP VOCABULARY CHECK ({}) ===", self.omop_dir)
        for f in found:
            size = (self.omop_dir / f).stat().st_size / 1e6
            logger.info("  {}: ✓ found ({:.1f} MB)", f, size)
        for f in missing:
            logger.warning("  {}: ✗ not found", f)
        if not omop_available:
            logger.warning("OMOP not available — concept matcher will use hardcoded fallback")
            logger.warning("To enable: see backend/omop/SETUP.md")
        elif not hierarchy_available:
            logger.warning("OMOP loaded but CONCEPT_ANCESTOR.csv missing — hierarchy traversal disabled")
        return {
            "omop_available": omop_available,
            "hierarchy_available": hierarchy_available,
            "files_found": found, "files_missing": missing,
            "reason": "OK" if omop_available else f"CONCEPT.csv not found in {self.omop_dir}",
        }

    def load(self, vocabularies=("SNOMED", "LOINC", "RxNorm"),
             domains=("Condition", "Drug", "Measurement", "Observation"),
             max_concepts: int = 200_000):
        if not self.status["omop_available"]:
            logger.warning("Skipping OMOP load — files not present")
            return
        if self.loaded:
            return
        # OMOP concept files have very large fields; raise the csv limit.
        try:
            csv.field_size_limit(sys.maxsize)
        except (OverflowError, ValueError):
            csv.field_size_limit(2**31 - 1)

        logger.info("Loading OMOP CONCEPT.csv (filter: vocab={}, domains={})...",
                    vocabularies, domains)
        concept_path = self.omop_dir / "CONCEPT.csv"
        n = 0
        with open(concept_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if row.get("standard_concept") != "S":
                    continue
                if row.get("vocabulary_id") not in vocabularies:
                    continue
                if row.get("domain_id") not in domains:
                    continue
                try:
                    cid = int(row["concept_id"])
                except (ValueError, KeyError):
                    continue
                name = row["concept_name"]
                self.concepts[cid] = {
                    "concept_id": cid, "name": name,
                    "vocabulary_id": row["vocabulary_id"],
                    "domain_id": row["domain_id"],
                    "concept_code": row["concept_code"],
                }
                self.name_to_id[name.lower()] = cid
                self.code_to_id[row["concept_code"]] = cid
                n += 1
                if n >= max_concepts:
                    logger.warning("Reached max_concepts cap ({}) — stopping CONCEPT load", max_concepts)
                    break
        logger.info("✓ Loaded {} OMOP standard concepts", n)

        # Load hierarchy
        if self.status["hierarchy_available"]:
            self._load_ancestors()

        # Load synonyms (optional)
        syn_path = self.omop_dir / "CONCEPT_SYNONYM.csv"
        if syn_path.exists():
            self._load_synonyms(syn_path)

        self.loaded = True

    def _load_ancestors(self):
        logger.info("Loading SNOMED-CT hierarchy from CONCEPT_ANCESTOR.csv...")
        path = self.omop_dir / "CONCEPT_ANCESTOR.csv"
        n = 0
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                try:
                    anc = int(row["ancestor_concept_id"])
                    desc = int(row["descendant_concept_id"])
                except (ValueError, KeyError):
                    continue
                # only keep relationships where both ends are in our loaded concept set
                if anc in self.concepts and desc in self.concepts:
                    self.ancestors.setdefault(desc, set()).add(anc)
                    self.descendants.setdefault(anc, set()).add(desc)
                    n += 1
        logger.info("✓ Loaded {} hierarchy edges ({} concepts have ancestors)",
                    n, len(self.ancestors))

    def _load_synonyms(self, path):
        logger.info("Loading concept synonyms...")
        try:
            csv.field_size_limit(sys.maxsize)
        except (OverflowError, ValueError):
            csv.field_size_limit(2**31 - 1)
        n = 0
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                try:
                    cid = int(row["concept_id"])
                except (ValueError, KeyError):
                    continue
                if cid in self.concepts:
                    syn = row.get("concept_synonym_name", "")
                    if syn:
                        self.synonyms.setdefault(cid, []).append(syn)
                        self.name_to_id[syn.lower()] = cid
                        n += 1
        logger.info("✓ Loaded {} synonyms", n)

    def is_a(self, specific_concept: str, general_concept: str) -> bool:
        """
        SNOMED-CT hierarchy check: does specific_concept descend from general_concept?
        e.g. is_a("essential hypertension", "hypertensive disorder") → True
        Used by the concept matcher so a patient coded with a specific condition
        matches a rule written with the parent (more general) concept.
        """
        spec_id = self.name_to_id.get(specific_concept.lower())
        gen_id = self.name_to_id.get(general_concept.lower())
        if spec_id is None or gen_id is None:
            return False
        if spec_id == gen_id:
            return True
        return gen_id in self.ancestors.get(spec_id, set())

    def get_descendants(self, concept: str) -> list[str]:
        """Return all more-specific concepts under this one (for query expansion)."""
        cid = self.name_to_id.get(concept.lower())
        if cid is None:
            return []
        return [self.concepts[d]["name"] for d in self.descendants.get(cid, set())
                if d in self.concepts]

    def get_concept_names(self) -> list[str]:
        """All concept names (used to build the FAISS index when OMOP is active)."""
        return [c["name"] for c in self.concepts.values()]

    def get_concept_count(self) -> int | None:
        return len(self.concepts) if self.concepts else None

    def get_status(self) -> dict:
        return {**self.status, "concept_count": len(self.concepts),
                "hierarchy_edges": sum(len(v) for v in self.ancestors.values()),
                "loaded": self.loaded}

    def get_setup_instructions(self) -> str:
        setup_path = self.omop_dir / "SETUP.md"
        if setup_path.exists():
            return setup_path.read_text()
        return "See backend/omop/SETUP.md for OMOP vocabulary download instructions."


_omop: OMOPVocabulary | None = None


def get_omop() -> OMOPVocabulary:
    global _omop
    if _omop is None:
        _omop = OMOPVocabulary()
    return _omop
