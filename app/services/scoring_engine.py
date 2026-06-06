import re
from loguru import logger
from app.models.patient import PatientData
from app.models.protocol import CriterionRule, CriterionType
from app.models.screening import CriterionEvaluationCreate, EvaluationStatus, VerdictType
from app.services.concept_matcher import snomed_matcher
from dataclasses import dataclass


@dataclass
class ScoringResult:
    fit_score: int
    confidence_low: int
    confidence_high: int
    overall_verdict: VerdictType
    evaluations: list[CriterionEvaluationCreate]


# Aliases so GPT-4o shorthand ("HbA1c", "eGFR") matches full LOINC display names
# stored in patient lab results ("Hemoglobin A1c/Hemoglobin.total in Blood", etc.)
_LAB_ALIASES: dict[str, list[str]] = {
    "hba1c":          ["hemoglobin a1c", "a1c", "glycated hemoglobin", "glycohemoglobin"],
    "hemoglobin a1c": ["hemoglobin a1c", "a1c", "glycated hemoglobin"],
    "egfr":           ["glomerular filtration rate", "egfr", "gfr/1.73"],
    "gfr":            ["glomerular filtration rate", "egfr"],
    "creatinine":     ["creatinine"],
    "hemoglobin":     ["hemoglobin", "haemoglobin"],
    "platelet":       ["platelet", "thrombocyte"],
    "wbc":            ["leukocyte", "white blood cell", "wbc"],
    "white blood":    ["leukocyte", "white blood cell"],
    "alt":            ["alanine aminotransferase", "alanine transaminase"],
    "alanine":        ["alanine aminotransferase", "alanine transaminase"],
    "ast":            ["aspartate aminotransferase", "aspartate transaminase"],
    "aspartate":      ["aspartate aminotransferase", "aspartate transaminase"],
    "glucose":        ["glucose", "blood sugar"],
    "blood sugar":    ["glucose"],
    "cholesterol":    ["cholesterol"],
    "ldl":            ["low-density lipoprotein", "ldl cholesterol", "cholesterol in ldl"],
    "hdl":            ["high-density lipoprotein", "hdl cholesterol", "cholesterol in hdl"],
    "triglyceride":   ["triglyceride"],
    "bmi":            ["body mass index", "bmi"],
    "body mass":      ["body mass index"],
    "systolic":       ["systolic blood pressure"],
    "diastolic":      ["diastolic blood pressure"],
    "blood pressure": ["systolic blood pressure", "diastolic blood pressure"],
    "sodium":         ["sodium"],
    "potassium":      ["potassium"],
    "bilirubin":      ["bilirubin"],
    "heart rate":     ["heart rate"],
    "pulse":          ["heart rate", "pulse rate"],
}


def _normalize(s: str) -> str:
    return s.lower().strip()


# Standard upper limits of normal used to resolve "Nx ULN" thresholds.
# Values are in the same units the scoring engine receives from FHIR (U/L, mg/dL, etc.)
_ULN: dict[str, float] = {
    "alt":         40.0,   # U/L  (alanine aminotransferase)
    "ast":         40.0,   # U/L  (aspartate aminotransferase)
    "bilirubin":    1.2,   # mg/dL
    "alkaline phosphatase": 120.0,  # U/L
    "creatinine":   1.2,   # mg/dL
}


def _extract_numeric(s: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def _resolve_threshold(threshold_str: str, concept: str) -> float | None:
    """
    Resolve a threshold string to a concrete float.
    Handles 'Nx ULN' / 'x times ULN' patterns by looking up the standard ULN
    for the given concept and multiplying.  Falls back to raw numeric extraction.
    """
    t = threshold_str.lower().strip()

    # Detect patterns like "1.5x ULN", "1.5 times ULN", "1.5× ULN", "1.5 × the ULN"
    uln_match = re.search(r"([\d.]+)\s*[x×]\s*(?:the\s+)?uln", t)
    if not uln_match:
        uln_match = re.search(r"([\d.]+)\s+times?\s+(?:the\s+)?(?:upper\s+limit|uln)", t)

    if uln_match:
        multiplier = float(uln_match.group(1))
        # Find which lab's ULN to use
        c = concept.lower()
        for key, uln_val in _ULN.items():
            if key in c or any(alias in c for alias in [key]):
                return multiplier * uln_val
        # concept not in table — cannot resolve, treat as AMBIGUOUS
        return None

    return _extract_numeric(threshold_str)


def _compare_lab(value: float, operator: str, threshold_str: str, concept: str = "") -> bool | None:
    if operator == "between":
        nums = re.findall(r"[-+]?\d*\.?\d+", threshold_str)
        if len(nums) >= 2:
            lo, hi = float(nums[0]), float(nums[1])
            return lo <= value <= hi
        return None
    threshold = _resolve_threshold(threshold_str, concept)
    if threshold is None:
        return None
    ops = {
        ">":  value > threshold,
        ">=": value >= threshold,
        "<":  value < threshold,
        "<=": value <= threshold,
        "==": abs(value - threshold) < 0.01,
        "!=": abs(value - threshold) >= 0.01,
    }
    return ops.get(operator)


def _lab_name_matches(concept_norm: str, lab_norm: str) -> bool:
    """Return True if concept likely refers to this lab observation name."""
    if concept_norm in lab_norm or lab_norm in concept_norm:
        return True
    if any(w in lab_norm for w in concept_norm.split() if len(w) > 3):
        return True
    for alias_key, aliases in _LAB_ALIASES.items():
        if alias_key in concept_norm:
            if any(a in lab_norm for a in aliases):
                return True
    return False


def _concept_in_list(concept: str, items: list[str]) -> tuple[bool, str]:
    matches = snomed_matcher.find_best_match(concept, top_k=3)
    best_terms = [m["term"].lower() for m in matches if m["score"] > 0.4]
    concept_lower = _normalize(concept)

    for item in items:
        item_lower = _normalize(item)
        if concept_lower in item_lower or item_lower in concept_lower:
            return True, item
        for term in best_terms:
            words = [w for w in term.split() if len(w) > 3]
            if any(w in item_lower for w in words):
                return True, item
    return False, ""


class ScoringEngine:
    def evaluate_patient(
        self, patient: PatientData, rules: list[CriterionRule]
    ) -> ScoringResult:
        logger.info(
            "=== SCORING PATIENT {} AGAINST {} RULES ===", patient.patient_id, len(rules)
        )

        evaluations: list[CriterionEvaluationCreate] = []
        inclusion_fails = 0
        exclusion_fails = 0
        ambiguous_count = 0
        score_deductions = 0

        for rule in rules:
            status, explanation, data_found = self._evaluate_rule(rule, patient)
            # criterion_type is authoritative; fall back to required flag only when type is unknown
            is_inclusion = (
                rule.criterion_type == CriterionType.inclusion
                or (rule.criterion_type != CriterionType.exclusion and rule.required)
            )

            if is_inclusion:
                if status == EvaluationStatus.FAIL:
                    score_deductions += 20
                    inclusion_fails += 1
                elif status == EvaluationStatus.AMBIGUOUS:
                    # AMBIGUOUS = no data found in the patient record.
                    # This is NOT a confirmed failure — missing data only widens
                    # the confidence band. Do NOT deduct score points here.
                    ambiguous_count += 1
                # PASS: no deduction
            else:
                if status == EvaluationStatus.PASS:
                    # Exclusion criterion triggered → patient is excluded
                    status = EvaluationStatus.FAIL
                    score_deductions += 25
                    exclusion_fails += 1
                    explanation = f"Patient HAS excluded condition/finding: {explanation}"
                elif status == EvaluationStatus.FAIL:
                    # Exclusion NOT triggered → good
                    status = EvaluationStatus.PASS
                    explanation = f"Exclusion criterion not met (good): {explanation}"
                elif status == EvaluationStatus.AMBIGUOUS:
                    ambiguous_count += 1

            prefix = "INCLUSION" if is_inclusion else "EXCLUSION"
            log_fn = logger.warning if status == EvaluationStatus.AMBIGUOUS else logger.info
            log_fn(
                "  [{}] {}: {} — {}",
                prefix,
                rule.concept,
                status.value,
                explanation,
            )

            evaluations.append(
                CriterionEvaluationCreate(
                    criterion_id=rule.id if hasattr(rule, "id") else None,
                    criterion_text=rule.criterion_text,
                    concept=rule.concept,
                    criterion_type=rule.criterion_type,
                    status=status,
                    explanation=explanation,
                    data_found=data_found,
                )
            )

        fit_score = max(0, min(100, 100 - score_deductions))

        # Confidence band: asymmetric — more downside uncertainty when many criteria
        # are unverified (we might be missing disqualifying data we don't have).
        total_rules = len(rules)
        band_width = min(50, ambiguous_count * 6)
        confidence_low = max(0, fit_score - band_width)
        confidence_high = min(100, fit_score + band_width // 3)

        # Verdict:
        # - Any confirmed mandatory inclusion FAIL → INELIGIBLE
        # - Score < 40 from confirmed exclusions → INELIGIBLE
        # - High score AND most criteria verifiable → ELIGIBLE
        # - High score but >50% unverified, or borderline score → REVIEW_NEEDED
        mostly_ambiguous = total_rules > 0 and (ambiguous_count / total_rules) > 0.5

        if inclusion_fails > 0:
            verdict = VerdictType.INELIGIBLE
        elif fit_score < 40:
            verdict = VerdictType.INELIGIBLE
        elif fit_score >= 75 and not mostly_ambiguous:
            verdict = VerdictType.ELIGIBLE
        else:
            verdict = VerdictType.REVIEW_NEEDED

        logger.info(
            "=== FINAL SCORE: {}/100 | Band: {}-{} | Verdict: {} "
            "| incl_fails={} excl_fails={} ambiguous={}/{} ===",
            fit_score,
            confidence_low,
            confidence_high,
            verdict.value,
            inclusion_fails,
            exclusion_fails,
            ambiguous_count,
            total_rules,
        )

        return ScoringResult(
            fit_score=fit_score,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            overall_verdict=verdict,
            evaluations=evaluations,
        )

    def _evaluate_rule(
        self, rule: CriterionRule, patient: PatientData
    ) -> tuple[EvaluationStatus, str, str]:
        concept = rule.concept
        operator = rule.operator
        value = rule.value

        if concept.lower() in ("age",):
            if patient.age is None:
                return EvaluationStatus.AMBIGUOUS, "Patient age unknown", ""
            age = patient.age
            result = _compare_lab(float(age), operator, value, concept)
            if result is True:
                return EvaluationStatus.PASS, f"Patient age {age} satisfies {operator} {value}", str(age)
            elif result is False:
                return EvaluationStatus.FAIL, f"Patient age {age} does NOT satisfy {operator} {value}", str(age)
            else:
                return EvaluationStatus.AMBIGUOUS, f"Cannot compare age {age} to '{value}'", str(age)

        # Search lab results first (handles numeric comparisons)
        concept_norm = _normalize(concept)
        matched_labs = [
            lab for lab in patient.lab_results
            if _lab_name_matches(concept_norm, _normalize(lab.name))
        ]
        if matched_labs:
            lab = matched_labs[-1]  # most recent observation
            val_str = f"{lab.value} {lab.unit}"
            result = _compare_lab(lab.value, operator, value, concept)
            if result is True:
                return (
                    EvaluationStatus.PASS,
                    f"{lab.name} = {lab.value} {lab.unit} satisfies {operator} {value}",
                    val_str,
                )
            elif result is False:
                return (
                    EvaluationStatus.FAIL,
                    f"{lab.name} = {lab.value} {lab.unit} does NOT satisfy {operator} {value}",
                    val_str,
                )
            # result is None → non-numeric threshold (e.g. "1.5x ULN" for unknown concept);
            # fall through to presence check

        if operator in ("presence", "history_of"):
            found, item = _concept_in_list(concept, patient.conditions + patient.medications)
            if found:
                return EvaluationStatus.PASS, f"Found '{item}' in patient record", item
            return (
                EvaluationStatus.AMBIGUOUS,
                f"No evidence of '{concept}' in patient record",
                "",
            )

        if operator in ("absence", "no_history_of"):
            found, item = _concept_in_list(concept, patient.conditions + patient.medications)
            if found:
                return EvaluationStatus.FAIL, f"Found '{item}' which must be absent", item
            return (
                EvaluationStatus.PASS,
                f"No evidence of '{concept}' found (satisfies absence criterion)",
                "",
            )

        # General: concept found in conditions/medications with optional numeric comparison
        found_cond, item_cond = _concept_in_list(concept, patient.conditions)
        found_med, item_med = _concept_in_list(concept, patient.medications)

        if found_cond or found_med:
            item = item_cond or item_med
            result = _compare_lab(1.0, operator, value, concept) if _extract_numeric(value) else None
            if result is None:
                return EvaluationStatus.PASS, f"Found matching record: '{item}'", item
            return (
                EvaluationStatus.PASS if result else EvaluationStatus.FAIL,
                f"Found '{item}' with value comparison",
                item,
            )

        return (
            EvaluationStatus.AMBIGUOUS,
            f"No data found for '{concept}' in patient record",
            "",
        )


scoring_engine = ScoringEngine()
