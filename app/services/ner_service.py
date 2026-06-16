"""
Clinical NER using scispaCy. Extracts medical entities (diseases, chemicals/drugs,
procedures) from free-text eligibility criteria BEFORE Claude extraction.

Role in the pipeline: scispaCy pre-annotates the raw criteria text with detected
clinical entities. These entities are passed to Claude as hints, improving
extraction precision and giving the LangChain agent concrete spans to reason about.
It also feeds the concept-matcher: every detected entity gets SNOMED-mapped.
"""
from typing import Optional
from loguru import logger

# When the full scispaCy model (en_core_sci_sm) is loaded, keep only clinically
# meaningful entity labels. en_core_sci_sm tags spans with the generic "ENTITY"
# label, so this set is a no-op there but filters the en_ner_* models if present.
KEEP_LABELS = {"DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT", "ORGANISM", "ENTITY"}


class ClinicalNERService:
    PRIMARY_MODEL = "en_core_sci_sm"
    FALLBACK_MODEL = "en_core_web_sm"

    _instance: Optional["ClinicalNERService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nlp = None
            cls._instance._model_name = ""
        return cls._instance

    # Backward-compatible attribute used by health.py / main.py
    @property
    def nlp(self):
        return self._nlp

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_model(self):
        """Idempotent model load. Tries scispaCy first, falls back to en_core_web_sm."""
        if self._nlp is not None:
            return
        self._load_model()

    def _load_model(self):
        import spacy
        try:
            self._nlp = spacy.load(self.PRIMARY_MODEL)
            self._model_name = self.PRIMARY_MODEL
            logger.info("✓ Loaded scispaCy model: {}", self.PRIMARY_MODEL)
        except OSError:
            logger.warning(
                "scispaCy model {} not found — falling back to {}",
                self.PRIMARY_MODEL, self.FALLBACK_MODEL,
            )
            try:
                self._nlp = spacy.load(self.FALLBACK_MODEL)
                self._model_name = self.FALLBACK_MODEL
                logger.warning(
                    "✓ Loaded fallback model: {} (install scispaCy for best results — see README)",
                    self.FALLBACK_MODEL,
                )
            except OSError:
                logger.error(
                    "No spaCy model available. Run: python -m spacy download en_core_web_sm"
                )
                self._nlp = None
                self._model_name = "none"

    def extract_entities(self, text: str) -> list[dict]:
        """
        Returns list of {text, label, start, end} for detected clinical entities.
        For en_core_sci_sm, entities are clinical spans (no fine-grained labels in sm).
        """
        if self._nlp is None:
            self.load_model()
        if self._nlp is None:
            logger.warning("NER model not loaded — skipping entity extraction")
            return []
        if not text:
            return []

        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            label = ent.label_ or "ENTITY"
            # For the fine-grained ner models, restrict to clinical labels.
            if self._model_name == self.PRIMARY_MODEL and ent.label_ and ent.label_ not in KEEP_LABELS:
                continue
            entities.append(
                {"text": ent.text, "label": label,
                 "start": ent.start_char, "end": ent.end_char}
            )
        logger.info(
            "scispaCy ({}) found {} clinical entities in {} chars of criteria text",
            self._model_name or "none", len(entities), len(text),
        )
        logger.debug("Entities: {}", [e["text"] for e in entities[:30]])
        return entities

    def get_status(self) -> dict:
        return {
            "loaded": self._nlp is not None,
            "model": self._model_name or "not loaded",
            "is_scispacy": self._model_name == self.PRIMARY_MODEL,
        }


_ner_service: Optional[ClinicalNERService] = None


def get_ner_service() -> ClinicalNERService:
    global _ner_service
    if _ner_service is None:
        _ner_service = ClinicalNERService()
    return _ner_service


# Singleton — backward compatible name used by main.py and health.py
ner_service = get_ner_service()
