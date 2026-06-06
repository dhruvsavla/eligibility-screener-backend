from loguru import logger
from typing import Optional

KEEP_LABELS = {"DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT", "ORGANISM"}


class ClinicalNERService:
    _instance: Optional["ClinicalNERService"] = None
    _nlp = None
    _model_name: str = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load("en_core_sci_sm")
            self._model_name = "en_core_sci_sm"
            logger.info("Loaded spaCy model: {}", self._model_name)
        except OSError:
            logger.warning(
                "en_core_sci_sm not found, falling back to en_core_web_sm"
            )
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                self._model_name = "en_core_web_sm"
                logger.info("Loaded spaCy model: {}", self._model_name)
            except OSError:
                logger.error(
                    "Neither en_core_sci_sm nor en_core_web_sm found — NER disabled"
                )
                self._nlp = None
                self._model_name = "none"

    def extract_entities(self, text: str) -> list[dict]:
        if self._nlp is None:
            logger.warning("NER model not loaded — skipping entity extraction")
            return []
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            if self._model_name == "en_core_sci_sm" and ent.label_ not in KEEP_LABELS:
                continue
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
        entity_list = [e["text"] for e in entities[:10]]
        logger.info("NER found {} entities in criteria text: {}", len(entities), entity_list)
        return entities


ner_service = ClinicalNERService()
