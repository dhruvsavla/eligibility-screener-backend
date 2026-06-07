"""
OMOPVocabulary — loads and queries the OMOP Common Data Model vocabulary
downloaded from https://athena.ohdsi.org

The OMOP vocabulary provides:
- 5M+ standardized medical concepts across SNOMED, LOINC, RxNorm, ICD-10
- Concept relationships (is-a, maps-to, subsumes)
- Synonym resolution across coding systems

Required files (download from https://athena.ohdsi.org → Download → select vocabularies):
  backend/omop/CONCEPT.csv          (~500MB) — all concepts
  backend/omop/CONCEPT_SYNONYM.csv  (~200MB) — alternate names
  backend/omop/CONCEPT_RELATIONSHIP.csv (~1GB) — concept mappings (optional)

Vocabulary IDs to select on Athena:
  1   - SNOMED
  17  - RxNorm
  21  - LOINC
  34  - ICD10CM
  85  - RxNorm Extension
"""

import difflib
import os
from loguru import logger

OMOP_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "omop")
)
CONCEPT_PATH = os.path.join(OMOP_DIR, "CONCEPT.csv")
SYNONYM_PATH = os.path.join(OMOP_DIR, "CONCEPT_SYNONYM.csv")
RELATIONSHIP_PATH = os.path.join(OMOP_DIR, "CONCEPT_RELATIONSHIP.csv")


class OMOPFilesNotFoundError(Exception):
    """OMOP vocabulary files are missing from backend/omop/."""


class OMOPLoadError(Exception):
    """pandas failed to parse OMOP files."""


class OMOPVocabulary:

    def __init__(self):
        self._concepts = None       # pd.DataFrame after load()
        self._synonym_map: dict[str, list[str]] = {}  # concept_id → [synonym, ...]
        self._loaded = False

    def check_files(self) -> dict:
        """Return a status dict without loading data."""
        logger.info("=== OMOP VOCABULARY CHECK ===")
        files_to_check = {
            "CONCEPT.csv": CONCEPT_PATH,
            "CONCEPT_SYNONYM.csv": SYNONYM_PATH,
            "CONCEPT_RELATIONSHIP.csv": RELATIONSHIP_PATH,
        }
        files_found = []
        files_missing = []

        for name, path in files_to_check.items():
            if os.path.isfile(path):
                size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
                logger.info("  {}: ✓ found ({} MB)", name, size_mb)
                files_found.append(name)
            else:
                if name != "CONCEPT_RELATIONSHIP.csv":  # optional
                    logger.warning("  {}: ✗ not found", name)
                else:
                    logger.info("  {}: ✗ not found (optional)", name)
                files_missing.append(name)

        required_present = "CONCEPT.csv" in files_found
        omop_available = required_present

        if omop_available:
            if "CONCEPT_RELATIONSHIP.csv" in files_missing:
                logger.warning(
                    "⚠ OMOP partially available — concept matching will work but relationship "
                    "traversal disabled. Download CONCEPT_RELATIONSHIP.csv from athena.ohdsi.org"
                )
            reason = "CONCEPT.csv found — OMOP vocabulary available"
        else:
            logger.warning("⚠ OMOP vocabulary files not found at {}/", OMOP_DIR)
            logger.warning("  Falling back to hardcoded SNOMED concept list (80 concepts)")
            logger.warning("  To enable full OMOP (5M+ concepts): see backend/omop/SETUP.md")
            reason = f"Required file CONCEPT.csv not found at {OMOP_DIR}"

        return {
            "omop_available": omop_available,
            "files_found": files_found,
            "files_missing": files_missing,
            "concept_count": None,
            "reason": reason,
        }

    def load(self, max_concepts: int = 200_000) -> None:
        """Read and filter CONCEPT.csv into memory."""
        try:
            import pandas as pd
        except ImportError:
            raise OMOPLoadError("pandas is required for OMOP vocabulary — pip install pandas")

        if self._loaded:
            return

        logger.info(
            "Loading OMOP vocabulary (filtering to Condition/Drug/Measurement/Observation "
            "domains, standard concepts only)..."
        )

        try:
            df = pd.read_csv(
                CONCEPT_PATH,
                sep="\t",
                usecols=["concept_id", "concept_name", "vocabulary_id", "domain_id",
                          "concept_code", "standard_concept"],
                dtype=str,
                low_memory=False,
            )
        except Exception as e:
            raise OMOPLoadError(f"Failed to read CONCEPT.csv: {e}") from e

        # Filter: standard concepts in relevant vocabularies + domains
        df = df[df["standard_concept"] == "S"]
        df = df[df["vocabulary_id"].isin(["SNOMED", "LOINC", "RxNorm"])]
        df = df[df["domain_id"].isin(["Condition", "Drug", "Measurement", "Observation"])]

        if len(df) > max_concepts:
            df = df.sample(n=max_concepts, random_state=42)

        snomed_n = (df["vocabulary_id"] == "SNOMED").sum()
        loinc_n = (df["vocabulary_id"] == "LOINC").sum()
        rxnorm_n = (df["vocabulary_id"] == "RxNorm").sum()
        logger.info(
            "✓ Loaded {} OMOP standard concepts ({} SNOMED, {} LOINC, {} RxNorm)",
            len(df), snomed_n, loinc_n, rxnorm_n,
        )

        self._concepts = df.reset_index(drop=True)

        # Load synonyms if available
        if os.path.isfile(SYNONYM_PATH):
            try:
                syn_df = pd.read_csv(
                    SYNONYM_PATH,
                    sep="\t",
                    usecols=["concept_id", "concept_synonym_name"],
                    dtype=str,
                    low_memory=False,
                )
                for row in syn_df.itertuples():
                    cid = str(row.concept_id)
                    name = str(row.concept_synonym_name)
                    self._synonym_map.setdefault(cid, []).append(name.lower())
                logger.info("✓ Loaded {} concept synonyms", len(syn_df))
            except Exception as e:
                logger.warning("⚠ Could not load CONCEPT_SYNONYM.csv: {}", e)

        self._loaded = True

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search concepts by exact, prefix, then fuzzy match."""
        if self._concepts is None or not self._loaded:
            return []

        q = query.lower().strip()
        results = []

        concept_names_lower = self._concepts["concept_name"].str.lower()

        # Exact match
        mask_exact = concept_names_lower == q
        for _, row in self._concepts[mask_exact].head(top_k).iterrows():
            results.append({**row.to_dict(), "match_type": "exact", "score": 1.0})

        # Prefix match
        if len(results) < top_k:
            mask_prefix = concept_names_lower.str.startswith(q)
            for _, row in self._concepts[mask_prefix].head(top_k - len(results)).iterrows():
                cid = str(row["concept_id"])
                if not any(r["concept_id"] == cid for r in results):
                    results.append({**row.to_dict(), "match_type": "prefix", "score": 0.9})

        # Fuzzy match via difflib
        if len(results) < top_k:
            sample = self._concepts.sample(
                n=min(5000, len(self._concepts)), random_state=0
            )
            names = sample["concept_name"].str.lower().tolist()
            close = difflib.get_close_matches(q, names, n=top_k, cutoff=0.6)
            for match in close:
                mask = concept_names_lower == match
                for _, row in self._concepts[mask].head(1).iterrows():
                    cid = str(row["concept_id"])
                    if not any(r["concept_id"] == cid for r in results):
                        score = difflib.SequenceMatcher(None, q, match).ratio()
                        results.append({**row.to_dict(), "match_type": "fuzzy", "score": score})

        results = results[:top_k]

        if results:
            top = results[0]
            logger.debug(
                "OMOP search: '{}' → top match: '{}' ({}:{}) [{}]",
                query,
                top["concept_name"],
                top["vocabulary_id"],
                top["concept_code"],
                top["match_type"],
            )

        return results

    def get_concept_count(self) -> int | None:
        if self._concepts is None:
            return None
        return len(self._concepts)

    def get_setup_instructions(self) -> str:
        setup_path = os.path.join(OMOP_DIR, "SETUP.md")
        if os.path.isfile(setup_path):
            with open(setup_path) as f:
                return f.read()
        return "See backend/omop/SETUP.md for OMOP vocabulary download instructions."
