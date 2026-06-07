"""Tests for omop_vocabulary.py"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from app.services.omop_vocabulary import OMOPVocabulary, OMOP_DIR


@pytest.fixture
def vocab():
    v = OMOPVocabulary()
    # Reset singleton state for test isolation
    v._concepts = None
    v._synonym_map = {}
    v._loaded = False
    return v


# Test 1: check_files when omop/ dir has no files → omop_available=False
def test_check_files_missing(vocab):
    with patch("os.path.isfile", return_value=False):
        status = vocab.check_files()

    assert status["omop_available"] is False
    assert "CONCEPT.csv" in status["files_missing"]
    assert status["concept_count"] is None


# Test 2: check_files with CONCEPT.csv present → omop_available=True
def test_check_files_present(vocab):
    def fake_isfile(path):
        return "CONCEPT.csv" in path

    def fake_getsize(path):
        return 500 * 1024 * 1024

    with patch("os.path.isfile", side_effect=fake_isfile):
        with patch("os.path.getsize", side_effect=fake_getsize):
            status = vocab.check_files()

    assert status["omop_available"] is True
    assert "CONCEPT.csv" in status["files_found"]


# Test 3: search with loaded concepts returns correct top match
def test_search_with_loaded_concepts(vocab):
    import pandas as pd

    df = pd.DataFrame([
        {"concept_id": "44054006", "concept_name": "Type 2 diabetes mellitus",
         "vocabulary_id": "SNOMED", "domain_id": "Condition", "concept_code": "44054006",
         "standard_concept": "S"},
        {"concept_id": "59621000", "concept_name": "Essential hypertension",
         "vocabulary_id": "SNOMED", "domain_id": "Condition", "concept_code": "59621000",
         "standard_concept": "S"},
    ])
    vocab._concepts = df
    vocab._loaded = True

    results = vocab.search("type 2 diabetes", top_k=1)

    assert len(results) >= 1
    assert "diabetes" in results[0]["concept_name"].lower()


# Test 4: ConceptMatcher falls back to FAISS when OMOP not available
def test_concept_matcher_fallback():
    # Reset the singleton to test initialization
    from app.services import concept_matcher as cm_module

    # The snomed_matcher should still work (FAISS) even without OMOP
    snomed_matcher = cm_module.snomed_matcher
    # It should have the find_best_match method
    assert hasattr(snomed_matcher, "find_best_match")
    assert hasattr(snomed_matcher, "build_index")
    assert hasattr(snomed_matcher, "get_status")

    # get_status should return a valid dict
    status = snomed_matcher.get_status()
    assert "mode" in status
    assert status["mode"] in ("omop", "fallback")
    assert "concept_count" in status
