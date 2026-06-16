"""
Measures Claude's criteria-extraction accuracy against human gold annotations.

For a protocol, compares the LLM-extracted criterion_rules against the
human-labeled gold_annotations. Two rules "match" when their concept and
criterion_type agree AND their operator+value are equivalent (with light
normalization). Produces precision/recall/F1 on the EXTRACTION step itself —
distinct from screening accuracy.
"""
from loguru import logger


class ExtractionEvaluator:
    def _normalize(self, rule: dict) -> tuple:
        """Normalize a rule for comparison: (concept_lower, type, operator, value_norm)."""
        concept = str(rule.get("concept", "")).lower().strip().replace(" ", "_")
        ctype = str(rule.get("criterion_type", "")).lower()
        op = str(rule.get("operator", "")).lower()
        val = str(rule.get("value", "")).lower().strip().replace(" ", "")
        return (concept, ctype, op, val)

    def _concept_matches(self, gold: dict, pred: dict) -> bool:
        """A gold and predicted rule match if concept + type align and op/value are close."""
        g = self._normalize(gold)
        p = self._normalize(pred)
        # concept + type must match; operator OR value match counts as a hit
        concept_ok = (g[0] == p[0]) or (g[0] in p[0]) or (p[0] in g[0])
        type_ok = g[1] == p[1]
        opval_ok = (g[2] == p[2]) or (g[3] == p[3] and g[3] != "")
        return concept_ok and type_ok and opval_ok

    def evaluate(self, gold_rules: list[dict], predicted_rules: list[dict],
                 protocol_title: str = "") -> dict:
        logger.info("=== EXTRACTION ACCURACY: {} ===", protocol_title)
        logger.info("Gold rules: {} | Predicted rules: {}",
                    len(gold_rules), len(predicted_rules))

        matched_gold = set()
        matched_pred = set()
        match_pairs = []

        for gi, g in enumerate(gold_rules):
            for pi, p in enumerate(predicted_rules):
                if pi in matched_pred:
                    continue
                if self._concept_matches(g, p):
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    match_pairs.append((g.get("concept", ""), p.get("concept", "")))
                    break

        tp = len(matched_gold)
        precision = tp / len(predicted_rules) if predicted_rules else 0.0
        recall = tp / len(gold_rules) if gold_rules else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # Identify misses
        missed = [gold_rules[i].get("criterion_text", "") for i in range(len(gold_rules))
                  if i not in matched_gold]
        spurious = [predicted_rules[i].get("criterion_text", "") for i in range(len(predicted_rules))
                    if i not in matched_pred]

        logger.info("Precision: {:.1%} | Recall: {:.1%} | F1: {:.1%}",
                    precision, recall, f1)
        if missed:
            logger.warning("Gold criteria Claude MISSED ({}):", len(missed))
            for m in missed:
                logger.warning("  ✗ {}", str(m)[:80])
        if spurious:
            logger.warning("Claude criteria with NO gold match ({}):", len(spurious))
            for s in spurious:
                logger.warning("  ? {}", str(s)[:80])

        return {
            "gold_count": len(gold_rules),
            "extracted_count": len(predicted_rules),
            "matched_count": tp,
            "precision": precision, "recall": recall, "f1": f1,
            "missed_criteria": missed, "spurious_criteria": spurious,
            "match_pairs": match_pairs,
        }


extraction_evaluator = ExtractionEvaluator()
