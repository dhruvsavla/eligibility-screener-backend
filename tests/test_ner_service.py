"""Tests for the scispaCy clinical NER service."""
import pytest
from app.services.ner_service import ClinicalNERService, get_ner_service


@pytest.fixture
def ner():
    svc = get_ner_service()
    svc.load_model()
    return svc


def test_singleton_identity():
    assert get_ner_service() is get_ner_service()


def test_get_status_shape(ner):
    status = ner.get_status()
    assert set(status.keys()) == {"loaded", "model", "is_scispacy"}
    assert isinstance(status["loaded"], bool)


def test_extract_entities_returns_spans(ner):
    if not ner.get_status()["loaded"]:
        pytest.skip("No spaCy model installed in this environment")
    text = "Patients with type 2 diabetes mellitus taking metformin and insulin."
    ents = ner.extract_entities(text)
    assert isinstance(ents, list)
    for e in ents:
        assert set(["text", "label", "start", "end"]).issubset(e.keys())
        assert text[e["start"]:e["end"]] == e["text"]


def test_extract_entities_empty_text(ner):
    assert ner.extract_entities("") == []


def test_no_model_returns_empty(monkeypatch):
    svc = ClinicalNERService()
    monkeypatch.setattr(svc, "_nlp", None)
    # Force load to leave _nlp None (simulate no model available)
    monkeypatch.setattr(svc, "load_model", lambda: None)
    assert svc.extract_entities("type 2 diabetes") == []
