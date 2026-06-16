"""Tests for the extraction accuracy evaluator (Measure 1)."""
from app.services.extraction_evaluator import ExtractionEvaluator


def _rule(concept, op, value, ctype="inclusion", text=None):
    return {
        "criterion_text": text or f"{concept} {op} {value}",
        "concept": concept, "operator": op, "value": value,
        "required": ctype == "inclusion", "criterion_type": ctype,
    }


def test_perfect_match_gives_unity():
    ev = ExtractionEvaluator()
    gold = [_rule("HbA1c", ">=", "7.5"), _rule("age", "between", "18-75")]
    pred = [_rule("HbA1c", ">=", "7.5"), _rule("age", "between", "18-75")]
    r = ev.evaluate(gold, pred)
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0
    assert r["matched_count"] == 2


def test_missed_gold_rule_lowers_recall():
    ev = ExtractionEvaluator()
    gold = [_rule("HbA1c", ">=", "7.5"), _rule("eGFR", ">=", "60")]
    pred = [_rule("HbA1c", ">=", "7.5")]   # missed eGFR
    r = ev.evaluate(gold, pred)
    assert r["recall"] == 0.5
    assert r["precision"] == 1.0
    assert "eGFR >= 60" in r["missed_criteria"]


def test_spurious_pred_lowers_precision():
    ev = ExtractionEvaluator()
    gold = [_rule("HbA1c", ">=", "7.5")]
    pred = [_rule("HbA1c", ">=", "7.5"), _rule("smoking", "presence", "")]  # extra
    r = ev.evaluate(gold, pred)
    assert r["recall"] == 1.0
    assert r["precision"] == 0.5
    assert any("smoking" in s for s in r["spurious_criteria"])


def test_concept_normalization_case_and_space():
    ev = ExtractionEvaluator()
    gold = [_rule("Type 2 Diabetes", "presence", "")]
    pred = [_rule("type 2 diabetes", "presence", "")]
    r = ev.evaluate(gold, pred)
    assert r["matched_count"] == 1


def test_empty_inputs():
    ev = ExtractionEvaluator()
    r = ev.evaluate([], [])
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0
