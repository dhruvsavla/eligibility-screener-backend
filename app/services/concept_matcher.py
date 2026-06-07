"""
Unified ConceptMatcher — uses OMOP vocabulary when available,
falls back to the hardcoded SNOMED list when OMOP files are absent.

The singleton `snomed_matcher` is kept for backward compatibility with
all existing callers (scoring_engine, protocols router, health router).
"""

import time
import numpy as np
from loguru import logger
from typing import Optional

from app.services.omop_vocabulary import OMOPVocabulary

SNOMED_CONCEPTS = [
    {"code": "44054006", "term": "Type 2 diabetes mellitus"},
    {"code": "73211009", "term": "Diabetes mellitus"},
    {"code": "44054006", "term": "HbA1c glycated hemoglobin"},
    {"code": "14183003", "term": "Chronic kidney disease"},
    {"code": "59621000", "term": "Essential hypertension"},
    {"code": "40930008", "term": "Hypothyroidism"},
    {"code": "34093004", "term": "Hyperthyroidism"},
    {"code": "237599002", "term": "Insulin resistance"},
    {"code": "22298006", "term": "Myocardial infarction"},
    {"code": "84114007", "term": "Heart failure"},
    {"code": "49436004", "term": "Atrial fibrillation"},
    {"code": "53741008", "term": "Coronary artery disease"},
    {"code": "230690007", "term": "Stroke cerebrovascular accident"},
    {"code": "128053003", "term": "Deep vein thrombosis"},
    {"code": "59282003", "term": "Pulmonary embolism"},
    {"code": "363346000", "term": "Malignant neoplasm cancer"},
    {"code": "254837009", "term": "Breast cancer"},
    {"code": "254637007", "term": "Non-small cell lung cancer"},
    {"code": "93761005", "term": "Colon cancer"},
    {"code": "372095001", "term": "Pancreatic cancer"},
    {"code": "363418001", "term": "Pancreatic malignant neoplasm"},
    {"code": "404080003", "term": "Prostate cancer"},
    {"code": "285432005", "term": "Lymphoma"},
    {"code": "91861009", "term": "Acute myeloid leukemia"},
    {"code": "372567009", "term": "Metformin"},
    {"code": "412231004", "term": "Insulin"},
    {"code": "372756006", "term": "Warfarin anticoagulant"},
    {"code": "387207008", "term": "Ibuprofen NSAID"},
    {"code": "372687004", "term": "Amoxicillin antibiotic"},
    {"code": "108490001", "term": "Chemotherapy"},
    {"code": "108290001", "term": "Radiation therapy"},
    {"code": "416608005", "term": "Immunotherapy"},
    {"code": "43396009", "term": "eGFR glomerular filtration rate"},
    {"code": "250745003", "term": "Creatinine level"},
    {"code": "365755008", "term": "Hemoglobin level"},
    {"code": "415068001", "term": "Platelet count"},
    {"code": "413587002", "term": "White blood cell count"},
    {"code": "102737005", "term": "ALT liver enzyme"},
    {"code": "45896001", "term": "AST liver enzyme"},
    {"code": "17234004", "term": "Bilirubin"},
    {"code": "36048009", "term": "Potassium electrolyte"},
    {"code": "39972003", "term": "Sodium electrolyte"},
    {"code": "444301002", "term": "Age years"},
    {"code": "248152002", "term": "Female sex"},
    {"code": "248153007", "term": "Male sex"},
    {"code": "77386006", "term": "Pregnancy"},
    {"code": "169631005", "term": "Breastfeeding lactation"},
    {"code": "415068001", "term": "BMI body mass index"},
    {"code": "56018004", "term": "Smoking tobacco use"},
    {"code": "228273003", "term": "Alcohol use"},
    {"code": "195967001", "term": "Asthma"},
    {"code": "13645005", "term": "COPD chronic obstructive pulmonary disease"},
    {"code": "233726005", "term": "Pulmonary fibrosis"},
    {"code": "84757009", "term": "Epilepsy seizure disorder"},
    {"code": "49049000", "term": "Parkinson disease"},
    {"code": "26929004", "term": "Alzheimer disease dementia"},
    {"code": "35489007", "term": "Depression"},
    {"code": "69322001", "term": "Schizophrenia"},
    {"code": "19943007", "term": "Liver cirrhosis"},
    {"code": "235856003", "term": "Hepatitis"},
    {"code": "128302006", "term": "Chronic hepatitis C"},
    {"code": "50711007", "term": "Hepatitis C"},
    {"code": "66071002", "term": "Hepatitis B"},
    {"code": "69896004", "term": "Rheumatoid arthritis"},
    {"code": "200936003", "term": "Lupus systemic lupus erythematosus"},
    {"code": "24526004", "term": "Inflammatory bowel disease"},
    {"code": "9014002",  "term": "Psoriasis"},
    {"code": "86406008", "term": "HIV AIDS"},
    {"code": "40122008", "term": "Pneumonia"},
    {"code": "14189004", "term": "Active tuberculosis"},
    {"code": "236425005", "term": "End stage renal disease dialysis"},
    {"code": "161665007", "term": "Kidney transplant"},
]


class ConceptMatcher:
    """
    Unified concept matcher: uses OMOP when available, FAISS+SNOMED fallback otherwise.
    Exposes find_best_match() and build_index() for backward compatibility.
    """
    _instance: Optional["ConceptMatcher"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._index = None
            cls._instance._model = None
            cls._instance._embeddings = None
            cls._instance._omop = None
            cls._instance._using_omop = False
        return cls._instance

    def _init_omop(self):
        if self._omop is not None:
            return
        self._omop = OMOPVocabulary()
        status = self._omop.check_files()
        if status["omop_available"]:
            try:
                self._omop.load()
                self._using_omop = True
                n = self._omop.get_concept_count() or 0
                logger.info("✓ ConceptMatcher using OMOP vocabulary ({} concepts)", n)
            except Exception as e:
                logger.warning("⚠ OMOP load failed: {} — falling back to SNOMED FAISS", e)
                self._using_omop = False
        else:
            logger.warning(
                "ConceptMatcher using hardcoded SNOMED fallback (80 concepts): {}",
                status["reason"]
            )
            self._using_omop = False

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model all-MiniLM-L6-v2...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def build_index(self):
        """Build FAISS index. Also initializes OMOP if files are present."""
        import faiss

        self._init_omop()

        logger.info("Building FAISS SNOMED index with {} concepts...", len(SNOMED_CONCEPTS))
        start = time.time()
        model = self._get_model()
        terms = [c["term"] for c in SNOMED_CONCEPTS]
        self._embeddings = model.encode(terms, normalize_embeddings=True)
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(np.array(self._embeddings, dtype="float32"))
        elapsed = int((time.time() - start) * 1000)
        logger.info(
            "✓ SNOMED index built in {}ms ({} concepts indexed)", elapsed, len(SNOMED_CONCEPTS)
        )

    def find_best_match(self, query: str, top_k: int = 3) -> list[dict]:
        # Try OMOP first
        if self._using_omop and self._omop:
            omop_results = self._omop.search(query, top_k)
            if omop_results:
                return [
                    {
                        "code": r["concept_code"],
                        "term": r["concept_name"],
                        "vocabulary": r.get("vocabulary_id", "OMOP"),
                        "score": r["score"],
                    }
                    for r in omop_results
                ]

        # FAISS fallback
        if self._index is None:
            logger.warning("FAISS index not built — building on-demand")
            self.build_index()

        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.array(q_emb, dtype="float32"), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            concept = SNOMED_CONCEPTS[idx]
            results.append(
                {
                    "code": concept["code"],
                    "term": concept["term"],
                    "score": float(score),
                }
            )
            logger.debug(
                "  SNOMED match: '{}' → '{}' (code: {}, score: {:.3f})",
                query, concept["term"], concept["code"], score,
            )

        return results

    def get_status(self) -> dict:
        self._init_omop()
        concept_count = (
            self._omop.get_concept_count()
            if self._using_omop and self._omop
            else len(SNOMED_CONCEPTS)
        )
        return {
            "mode": "omop" if self._using_omop else "fallback",
            "concept_count": concept_count,
            "omop_available": self._using_omop,
        }


# Singleton — backward compatible name
snomed_matcher = ConceptMatcher()
