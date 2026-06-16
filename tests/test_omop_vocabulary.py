"""Tests for omop_vocabulary.py (OMOP + SNOMED-CT hierarchy) and ConceptMatcher."""
import pytest
from app.services.omop_vocabulary import OMOPVocabulary


@pytest.fixture
def vocab():
    v = OMOPVocabulary()
    # Reset to a clean, manually-populated state for hierarchy tests.
    v.concepts = {}
    v.name_to_id = {}
    v.code_to_id = {}
    v.ancestors = {}
    v.descendants = {}
    v.loaded = False
    return v


def _add(v, cid, name, code="0"):
    v.concepts[cid] = {"concept_id": cid, "name": name, "vocabulary_id": "SNOMED",
                       "domain_id": "Condition", "concept_code": code}
    v.name_to_id[name.lower()] = cid
    v.code_to_id[code] = cid


def _link(v, ancestor_id, descendant_id):
    v.ancestors.setdefault(descendant_id, set()).add(ancestor_id)
    v.descendants.setdefault(ancestor_id, set()).add(descendant_id)


# ── check_files ──────────────────────────────────────────────────────────────

def test_check_files_returns_status_dict(vocab):
    status = vocab.check_files()
    assert "omop_available" in status
    assert "hierarchy_available" in status
    assert isinstance(status["files_found"], list)


# ── is_a hierarchy ───────────────────────────────────────────────────────────

def test_is_a_true_for_descendant(vocab):
    _add(vocab, 1, "Hypertensive disorder")
    _add(vocab, 2, "Essential hypertension")
    _link(vocab, ancestor_id=1, descendant_id=2)
    assert vocab.is_a("essential hypertension", "hypertensive disorder") is True


def test_is_a_false_for_unrelated(vocab):
    _add(vocab, 1, "Hypertensive disorder")
    _add(vocab, 3, "Type 2 diabetes mellitus")
    assert vocab.is_a("type 2 diabetes mellitus", "hypertensive disorder") is False


def test_is_a_same_concept_true(vocab):
    _add(vocab, 1, "Hypertensive disorder")
    assert vocab.is_a("hypertensive disorder", "hypertensive disorder") is True


def test_is_a_unknown_concept_false(vocab):
    _add(vocab, 1, "Hypertensive disorder")
    assert vocab.is_a("nonexistent", "hypertensive disorder") is False


def test_get_descendants(vocab):
    _add(vocab, 1, "Hypertensive disorder")
    _add(vocab, 2, "Essential hypertension")
    _add(vocab, 4, "Secondary hypertension")
    _link(vocab, 1, 2)
    _link(vocab, 1, 4)
    descendants = set(vocab.get_descendants("hypertensive disorder"))
    assert descendants == {"Essential hypertension", "Secondary hypertension"}


# ── ConceptMatcher.concept_subsumes ──────────────────────────────────────────

def test_concept_matcher_fallback_status():
    from app.services.concept_matcher import snomed_matcher
    status = snomed_matcher.get_status()
    assert status["mode"] in ("omop", "fallback")
    assert "concept_count" in status
    assert "hierarchy_active" in status


def test_concept_subsumes_via_hierarchy(monkeypatch):
    """When OMOP is active, concept_subsumes delegates to omop.is_a."""
    from app.services.concept_matcher import snomed_matcher

    fake_omop = OMOPVocabulary()
    fake_omop.concepts = {}
    fake_omop.name_to_id = {}
    fake_omop.ancestors = {}
    fake_omop.descendants = {}
    _add(fake_omop, 1, "Hypertensive disorder")
    _add(fake_omop, 2, "Essential hypertension")
    _link(fake_omop, 1, 2)

    monkeypatch.setattr(snomed_matcher, "_omop", fake_omop)
    monkeypatch.setattr(snomed_matcher, "_using_omop", True)

    assert snomed_matcher.concept_subsumes("hypertensive disorder", "essential hypertension") is True
    assert snomed_matcher.concept_subsumes("hypertensive disorder", "type 2 diabetes") is False


def test_concept_subsumes_false_in_fallback_mode(monkeypatch):
    from app.services.concept_matcher import snomed_matcher
    monkeypatch.setattr(snomed_matcher, "_using_omop", False)
    assert snomed_matcher.concept_subsumes("anything", "anything else") is False
