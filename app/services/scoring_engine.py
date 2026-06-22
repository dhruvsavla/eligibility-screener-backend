import re
import json
import asyncio
from datetime import date, timedelta
from loguru import logger
from dataclasses import dataclass, field
from app.models.patient import PatientData, LabResult, ConditionRecord, MedicationRecord
from app.models.protocol import CriterionRule, CriterionType
from app.models.screening import CriterionEvaluationCreate, EvaluationStatus, VerdictType
from app.services.concept_matcher import snomed_matcher
from app.services.agentic_fallback import fallback_agent
# SNOMED similarity thresholds
SNOMED_THRESHOLD_SIGNAL = 0.72   # >= 0.72 → confirmed match, use as PASS/FAIL
SNOMED_THRESHOLD_WEAK   = 0.55   # 0.55-0.72 → uncertain → AMBIGUOUS
# < 0.55 → no usable match → AMBIGUOUS

# Inclusion criteria that are inherently unverifiable from FHIR structured data.
# These describe temporal stability, administrative, patient-compliance, or compound
# multi-dimensional requirements that no structured patient record can confirm alone.
# They are excluded from the inclusion_pass_rate denominator AND from score deductions
# so that a fitting patient is not permanently blocked by coordinator-only items.
# They still surface in the rationale under "Requires manual verification".
UNVERIFIABLE_INCLUSION_PATTERNS = [
    "stable",               # temporal stability requirements
    "regimen for at least", # stable-regimen duration
    "months before",        # enrollment-timing requirements
    "willing",              # patient intent / consent
    "able to self",         # patient capability
    "self-inject",          # patient capability
    "informed consent",     # administrative
    "investigator",         # clinical judgment
    "judgment",             # clinical judgment
    "well-motivated",       # patient compliance
    "capable",              # patient capability
    "adherence",            # patient compliance
    "run-in",               # run-in period requirements
    "consent to participate",
    "risk factor",          # compound age+CVD-risk criteria (no single FHIR concept)
    "vascular disease",     # compound age+vascular criteria (no single FHIR concept)
]

LAB_LOINC_MAP: dict[str, list[str]] = {
    "hba1c":          ["4548-4", "17856-6", "59261-8", "41995-2"],
    "egfr":           ["33914-3", "62238-1", "98979-8", "48642-3", "48643-1"],
    "creatinine":     ["2160-0", "38483-4", "14682-9", "59826-8"],
    "hemoglobin":     ["718-7", "59260-0", "20509-6"],
    "wbc":            ["6690-2", "49498-9", "26464-8"],
    "platelets":      ["777-3", "26515-7", "74775-1"],
    "alt":            ["1742-6", "1743-4", "76625-3"],
    "ast":            ["1920-8", "1921-6"],
    "bilirubin":      ["1975-2", "14629-0", "59828-4"],
    "glucose":        ["2345-7", "14749-6", "2339-0"],
    "sodium":         ["2951-2", "2947-0", "39791-9"],
    "potassium":      ["2823-3", "6298-4", "39789-3"],
    "cholesterol":    ["2093-3", "35200-5", "14647-2"],
    "ldl":            ["13457-7", "18262-6", "2089-1"],
    "hdl":            ["2085-9", "14646-4"],
    "triglycerides":  ["2571-8", "12951-0"],
    "bmi":            ["39156-5", "59574-4"],
    "systolic_bp":    ["8480-6", "76534-7"],
    "diastolic_bp":   ["8462-4", "76535-4"],
}

LAB_ALIASES: dict[str, list[str]] = {
    "hba1c":          ["hemoglobin a1c", "hba1c", "glycated hemoglobin",
                       "glycohemoglobin", "a1c", "hgba1c"],
    "hemoglobin a1c": ["hemoglobin a1c", "a1c", "glycated hemoglobin"],
    "egfr":           ["egfr", "glomerular filtration", "gfr", "estimated gfr", "ckd-epi"],
    "gfr":            ["glomerular filtration", "egfr"],
    "creatinine":     ["creatinine", "creat", "scr"],
    "hemoglobin":     ["hemoglobin", "hgb", "haemoglobin"],
    "platelet":       ["platelet", "plt", "thrombocyte"],
    "wbc":            ["leukocyte", "white blood cell", "wbc"],
    "white blood":    ["leukocyte", "white blood cell"],
    "alt":            ["alanine aminotransferase", "alanine transaminase", "sgpt"],
    "alanine":        ["alanine aminotransferase", "alanine transaminase"],
    "ast":            ["aspartate aminotransferase", "aspartate transaminase", "sgot"],
    "aspartate":      ["aspartate aminotransferase", "aspartate transaminase"],
    "glucose":        ["glucose", "blood glucose", "fasting glucose", "blood sugar"],
    "blood sugar":    ["glucose"],
    "cholesterol":    ["cholesterol"],
    "ldl":            ["low-density lipoprotein", "ldl cholesterol", "cholesterol in ldl"],
    "hdl":            ["high-density lipoprotein", "hdl cholesterol", "cholesterol in hdl"],
    "triglyceride":   ["triglyceride"],
    "bmi":            ["body mass index", "bmi"],
    "body mass":      ["body mass index"],
    "systolic":       ["systolic blood pressure", "sbp"],
    "diastolic":      ["diastolic blood pressure", "dbp"],
    "blood pressure": ["systolic blood pressure", "diastolic blood pressure"],
    "sodium":         ["sodium"],
    "potassium":      ["potassium"],
    "bilirubin":      ["bilirubin", "bili"],
    "heart rate":     ["heart rate", "pulse rate"],
    "pulse":          ["heart rate", "pulse rate"],
}

_ULN: dict[str, float] = {
    "alt":         40.0,
    "ast":         40.0,
    "bilirubin":    1.2,
    "alkaline phosphatase": 120.0,
    "creatinine":   1.2,
}

_TIMEFRAME_DAYS: dict[str, float] = {
    "days":   1.0,
    "day":    1.0,
    "weeks":  7.0,
    "week":   7.0,
    "months": 30.44,
    "month":  30.44,
    "years":  365.25,
    "year":   365.25,
}


def _timeframe_cutoff(timeframe_value: float, timeframe_unit: str) -> date | None:
    """Return the earliest date that falls within the rule's time window from today.

    A record with onset_date >= cutoff is *within* the window; one before it is outside.
    Returns None if the timeframe cannot be parsed.
    """
    unit_norm = (timeframe_unit or "").lower().strip()
    days_per_unit = _TIMEFRAME_DAYS.get(unit_norm)
    if days_per_unit is None:
        return None
    delta = timedelta(days=timeframe_value * days_per_unit)
    return date.today() - delta


def _record_within_window(records: list, concept_name_lower: str, cutoff: date) -> bool:
    """True if any record whose name matches concept_name_lower has an onset_date >= cutoff."""
    for rec in records:
        name_lower = rec.name.lower()
        if concept_name_lower in name_lower or name_lower in concept_name_lower:
            if rec.onset_date is not None and rec.onset_date >= cutoff:
                return True
    return False


# Conversion factors: (from_unit, to_unit) → multiply patient value by this factor
# to convert it into the rule's unit before numeric comparison.
UNIT_CONVERSION_MAP: dict[tuple[str, str], float] = {
    ("g/dl", "g/l"):          10.0,
    ("g/l",  "g/dl"):          0.1,
    ("mg/dl", "mmol/l"):       0.05551,   # glucose
    ("mmol/l", "mg/dl"):      18.016,     # glucose
    ("umol/l", "mg/dl"):       0.01131,   # creatinine
    ("µmol/l", "mg/dl"):       0.01131,
    ("mg/dl", "umol/l"):      88.42,      # creatinine
    ("mg/dl", "µmol/l"):      88.42,
    ("nmol/l", "pmol/l"):   1000.0,
    ("pmol/l", "nmol/l"):      0.001,
}


def _normalize_unit(u: str | None) -> str:
    """Lowercase, strip whitespace and Unicode variants for unit comparison."""
    if not u:
        return ""
    return u.lower().strip().replace("μ", "µ").replace("u/", "u/")


def _convert_patient_value(patient_value: float, patient_unit: str | None, rule_unit: str | None) -> float:
    """Return patient_value converted to rule_unit if a known conversion exists, otherwise as-is."""
    pu = _normalize_unit(patient_unit)
    ru = _normalize_unit(rule_unit)
    if not pu or not ru or pu == ru:
        return patient_value
    factor = UNIT_CONVERSION_MAP.get((pu, ru))
    if factor is not None:
        logger.debug(
            "Unit conversion: {} {} → {} (×{})", patient_value, patient_unit, rule_unit, factor
        )
        return patient_value * factor
    return patient_value


@dataclass
class FitScoreData:
    fit_score: int
    deductions: int
    inclusion_pass: int
    inclusion_fail: int
    inclusion_ambiguous: int
    inclusion_total: int
    inclusion_pass_rate: float
    exclusion_triggered: int
    exclusion_ambiguous: int
    exclusion_clear: int


@dataclass
class ScoringResult:
    fit_score: int
    confidence_low: int
    confidence_high: int
    overall_verdict: VerdictType
    evaluations: list[CriterionEvaluationCreate]
    score_breakdown: dict = field(default_factory=dict)


def _normalize(s: str) -> str:
    return s.lower().strip()


def _extract_numeric(s: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def _resolve_threshold(threshold_str: str, concept: str) -> float | None:
    t = threshold_str.lower().strip()
    uln_match = re.search(r"([\d.]+)\s*[x×]\s*(?:the\s+)?uln", t)
    if not uln_match:
        uln_match = re.search(r"([\d.]+)\s+times?\s+(?:the\s+)?(?:upper\s+limit|uln)", t)
    if uln_match:
        multiplier = float(uln_match.group(1))
        c = concept.lower()
        for key, uln_val in _ULN.items():
            if key in c:
                return multiplier * uln_val
        return None
    return _extract_numeric(threshold_str)


def _compare_lab(
    value: float,
    operator: str,
    threshold_str: str,
    concept: str = "",
    patient_unit: str | None = None,
    rule_unit: str | None = None,
) -> bool | None:
    # Normalise patient value into the rule's unit before comparing
    value = _convert_patient_value(value, patient_unit, rule_unit)
    if operator in ("between", "range"):
        nums = re.findall(r"\d+\.?\d*", threshold_str)
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


class ScoringEngine:

    def _snomed_match_status(self, similarity_score: float, base_status: str) -> str:
        if similarity_score >= SNOMED_THRESHOLD_SIGNAL:
            tier = "SIGNAL"
            result = base_status
        elif similarity_score >= SNOMED_THRESHOLD_WEAK:
            tier = "WEAK"
            result = "AMBIGUOUS"
        else:
            tier = "NO_MATCH"
            result = "AMBIGUOUS"
        logger.debug(
            "SNOMED threshold: score={:.3f} → {} → status={}",
            similarity_score, tier, result
        )
        return result

    def _name_based_match_fallback(self, lab_name: str, concept: str) -> bool:
        concept_norm = concept.lower().strip()
        lab_norm = lab_name.lower().strip()
        if concept_norm in lab_norm or lab_norm in concept_norm:
            return True
        sig_words = [w for w in concept_norm.split() if len(w) > 4]
        if sig_words and all(w in lab_norm for w in sig_words):
            return True
        return False

    def _lab_name_matches(self, lab: LabResult, concept: str) -> bool:
        concept_lower = concept.lower().strip()
        concept_key = concept_lower.replace(" ", "_").replace("-", "_")

        # Priority 1: LOINC code exact match
        if lab.loinc_code:
            for map_key, codes in LAB_LOINC_MAP.items():
                if map_key in concept_key or concept_key in map_key:
                    if lab.loinc_code in codes:
                        logger.debug(
                            "Lab match via LOINC: concept='{}' lab='{}' loinc='{}'",
                            concept, lab.name, lab.loinc_code
                        )
                        return True

        # Priority 2: alias table
        lab_lower = lab.name.lower()
        for alias_key, aliases in LAB_ALIASES.items():
            if alias_key in concept_lower or concept_lower in alias_key:
                if any(a in lab_lower for a in aliases):
                    logger.debug(
                        "Lab match via alias: concept='{}' lab='{}' group='{}'",
                        concept, lab.name, alias_key
                    )
                    return True

        # Priority 3: substring / word overlap fallback
        return self._name_based_match_fallback(lab.name, concept)

    def _concept_in_list(self, concept: str, items: list[str]) -> tuple[bool, str, float]:
        """Returns (found, matched_item, best_snomed_score).

        A patient item matches the rule concept if ANY of:
          (a) substring match (with underscore→space normalization)  → score 1.0
          (a2) word-level match for compound concept names            → score 0.95
          (b) SNOMED-CT hierarchy is-a                               → score 1.0
          (c) FAISS semantic ≥ weak thresh                           → tiered score
        """
        matches = snomed_matcher.find_best_match(concept, top_k=3)
        best_score = max((m["score"] for m in matches), default=0.0) if matches else 0.0
        # Only expand via SNOMED terms that are HIGH-confidence matches (>= SIGNAL
        # threshold).  Using the WEAK threshold here caused false positives: FAISS
        # would find a loosely-related OMOP term at 0.55-0.71, then a common word
        # from that term ("diabetes", "chronic") would match an unrelated patient
        # condition.  High-confidence expansion is safer.
        best_terms = [m["term"].lower() for m in matches if m["score"] >= SNOMED_THRESHOLD_SIGNAL]

        # Normalize underscores to spaces so that compound concept names extracted
        # by Claude ("pancreatitis_hepatic_renal_thyroid_disorder") match patient
        # conditions ("Pancreatitis") via substring and word-level checks.
        concept_lower = concept.lower().strip()
        concept_normalized = concept_lower.replace("_", " ")

        # Significant clinical words from a compound concept name (filter stopwords
        # and generic terms that would cause false positives).
        _GENERIC = {"disorder", "disease", "syndrome", "condition", "history",
                    "therapy", "treatment", "procedure", "finding", "status"}
        concept_words = [
            w for w in concept_normalized.split()
            if len(w) > 5 and w not in _GENERIC
        ]

        for item in items:
            item_lower = item.lower().strip()

            # (a) Direct substring match (normalized)
            if concept_normalized in item_lower or item_lower in concept_normalized:
                return True, item, 1.0

            # (a2) ALL-word match: every significant word from the concept must
            # appear in the item.  This handles compound concepts that are stated
            # verbosely in patient records, e.g. "Uncontrolled type 2 diabetes
            # mellitus" matches the concept "uncontrolled_diabetes" because BOTH
            # "uncontrolled" and "diabetes" are present in the item.
            # We intentionally use AND (all) not OR (any) to prevent "diabetes"
            # alone matching unrelated "Type 2 diabetes mellitus" conditions.
            if concept_words and all(w in item_lower for w in concept_words):
                logger.debug(
                    "concept '{}' matched '{}' via ALL-WORD OVERLAP (score=0.95)",
                    concept, item,
                )
                return True, item, 0.95

            # (b) SNOMED-CT hierarchy: patient item IS-A the rule concept
            if snomed_matcher.concept_subsumes(concept, item):
                logger.debug(
                    "concept match '{}' ← patient '{}' via HIERARCHY (is-a)",
                    concept, item,
                )
                return True, item, 1.0

            # (c) FAISS semantic expansion via SNOMED terms
            # (c) FAISS semantic expansion via SNOMED terms
            for term in best_terms:
                term_lower = term.lower().strip()
                
                # Check for a direct phrase match first
                if term_lower in item_lower or item_lower in term_lower:
                    logger.debug(
                        "concept match '{}' ← patient '{}' via FAISS exact phrase (score={:.2f})",
                        concept, item, best_score,
                    )
                    return True, item, best_score
                
                # Fallback to ALL-WORD match (not ANY word) to prevent single-word false positives.
                # E.g., Both "anterior" AND "myocardial" AND "infarction" must be present.
                term_words = [w for w in term_lower.split() if len(w) > 4 and w not in _GENERIC]
                if term_words and all(w in item_lower for w in term_words):
                    logger.debug(
                        "concept match '{}' ← patient '{}' via FAISS all-word overlap (score={:.2f})",
                        concept, item, best_score,
                    )
                    return True, item, best_score
        return False, "", best_score

    def _is_unverifiable_inclusion(self, rule: CriterionRule) -> bool:
        """True if this inclusion criterion cannot be confirmed from FHIR structured data.

        Conditional-operator rules are always unverifiable (coordinator must evaluate).
        Otherwise, checks concept + criterion_text for patterns that indicate temporal
        stability, administrative, patient-compliance, or compound multi-dimensional
        requirements that no structured EHR record can directly confirm.
        """
        if rule.operator == "conditional":
            return True
        combined = (
            (rule.concept or "") + " " + (rule.criterion_text or "")
        ).lower()
        return any(pat in combined for pat in UNVERIFIABLE_INCLUSION_PATTERNS)

    def _calculate_fit_score(
        self,
        raw_evals: list[tuple[EvaluationStatus, CriterionRule]],
    ) -> FitScoreData:
        """
        Compute fit score from raw (pre-inversion) rule evaluations.

        Deduction table:
          INCLUSION FAIL      → -20  (confirmed missing required criterion)
          INCLUSION AMBIGUOUS → -5   (data gap penalty — prevents 100/100 with no data)
          INCLUSION UNVERIFIABLE → 0 (excluded from score AND pass-rate denominator)
          INCLUSION PASS      →  0
          EXCLUSION PASS      → -25  (exclusion triggered — disqualifier present)
          EXCLUSION AMBIGUOUS → -3   (unknown exclusion risk)
          EXCLUSION FAIL      →  0   (exclusion not triggered — good)

        Unverifiable inclusion criteria (temporal stability, administrative, compound
        multi-dimensional) are excluded from both the score deductions and the
        inclusion_pass_rate denominator. They still appear as AMBIGUOUS in the
        evaluations list so coordinators see them in the rationale.
        """
        inclusion_pass = inclusion_fail = inclusion_ambiguous = 0
        inclusion_unverifiable = 0
        exclusion_triggered = exclusion_ambiguous = exclusion_clear = 0
        deductions = 0

        for status, rule in raw_evals:
            is_inclusion = (
                rule.criterion_type == CriterionType.inclusion
                or (rule.criterion_type != CriterionType.exclusion and rule.required)
            )
            if is_inclusion:
                if self._is_unverifiable_inclusion(rule):
                    # Removed from both score AND denominator — coordinator verifies manually.
                    inclusion_unverifiable += 1
                    logger.debug(
                        "  INCLUSION UNVERIFIABLE: '{}' → excluded from score and pass-rate",
                        rule.concept
                    )
                elif status == EvaluationStatus.FAIL:
                    deductions += 20
                    inclusion_fail += 1
                    logger.debug("  INCLUSION FAIL: '{}' → -20 pts", rule.concept)
                elif status == EvaluationStatus.AMBIGUOUS:
                    deductions += 5
                    inclusion_ambiguous += 1
                    logger.debug("  INCLUSION AMBIGUOUS: '{}' → -5 pts (data gap)", rule.concept)
                else:
                    inclusion_pass += 1
                    logger.debug("  INCLUSION PASS: '{}'", rule.concept)
            else:
                if status == EvaluationStatus.PASS:
                    # Exclusion triggered — disqualifier present
                    deductions += 25
                    exclusion_triggered += 1
                    logger.debug("  EXCLUSION TRIGGERED: '{}' → -25 pts", rule.concept)
                elif status == EvaluationStatus.AMBIGUOUS:
                    deductions += 3
                    exclusion_ambiguous += 1
                    logger.debug("  EXCLUSION AMBIGUOUS: '{}' → -3 pts", rule.concept)
                else:
                    exclusion_clear += 1
                    logger.debug("  EXCLUSION CLEAR: '{}' (disqualifier not found)", rule.concept)

        fit_score = max(0, 100 - deductions)
        # Unverifiable criteria are excluded from both numerator and denominator.
        verifiable_inclusion_total = inclusion_pass + inclusion_fail + inclusion_ambiguous
        inclusion_pass_rate = (
            inclusion_pass / verifiable_inclusion_total
            if verifiable_inclusion_total > 0 else 1.0
        )
        inclusion_total = verifiable_inclusion_total + inclusion_unverifiable

        logger.info(
            "Score: deductions={} fit_score={} | "
            "incl pass={} fail={} ambig={} unverifiable={} rate={:.0%} ({}/{} verifiable) | "
            "excl triggered={} ambig={} clear={}",
            deductions, fit_score,
            inclusion_pass, inclusion_fail, inclusion_ambiguous, inclusion_unverifiable,
            inclusion_pass_rate, inclusion_pass, verifiable_inclusion_total,
            exclusion_triggered, exclusion_ambiguous, exclusion_clear,
        )

        return FitScoreData(
            fit_score=fit_score,
            deductions=deductions,
            inclusion_pass=inclusion_pass,
            inclusion_fail=inclusion_fail,
            inclusion_ambiguous=inclusion_ambiguous,
            inclusion_total=inclusion_total,
            inclusion_pass_rate=inclusion_pass_rate,
            exclusion_triggered=exclusion_triggered,
            exclusion_ambiguous=exclusion_ambiguous,
            exclusion_clear=exclusion_clear,
        )

    def _calculate_confidence_band(self, score: int, data: FitScoreData) -> tuple[int, int]:
        """
        Asymmetric band — unknown exclusions are riskier than unknown inclusions.
          Each ambiguous inclusion: 7 pts downside, 3 pts upside
          Each ambiguous exclusion: 10 pts downside, 2 pts upside
        """
        downside = min(40, data.inclusion_ambiguous * 7 + data.exclusion_ambiguous * 10)
        upside   = min(15, data.inclusion_ambiguous * 3 + data.exclusion_ambiguous * 2)
        confidence_low  = max(0,   score - downside)
        confidence_high = min(100, score + upside)
        logger.debug(
            "Confidence band: score={} down={} up={} → [{}, {}]",
            score, downside, upside, confidence_low, confidence_high,
        )
        return confidence_low, confidence_high

    def _determine_verdict(self, score: int, data: FitScoreData) -> VerdictType:
        """
        Gate 1 — INELIGIBLE: confirmed hard disqualifiers only.
        Gate 2 — ELIGIBLE: high confidence all-clear (strict to avoid FPs).
        Gate 3 — REVIEW_NEEDED: default catch-all (counts as TP under Option B sensitivity).
        """
        # Gate 1: confirmed disqualifiers
        if data.inclusion_fail >= 1:
            logger.info(
                "Verdict: INELIGIBLE (inclusion_fail={} — confirmed missing criterion)",
                data.inclusion_fail
            )
            return VerdictType.INELIGIBLE

        if data.exclusion_triggered >= 1:
            logger.info(
                "Verdict: INELIGIBLE (exclusion_triggered={} — confirmed disqualifier)",
                data.exclusion_triggered
            )
            return VerdictType.INELIGIBLE

        if score < 30:
            logger.info(
                "Verdict: INELIGIBLE (score={} < 30 — extreme data poverty, unscreenable)",
                score
            )
            return VerdictType.INELIGIBLE

        # Gate 2: high-confidence eligible.
        # ELIGIBLE depends on CONFIRMED signals only. Unverified exclusions (AMBIGUOUS
        # due to missing data) are expected on real patient records — a real chart
        # documents what a patient HAS, not every condition they don't have. A
        # coordinator treats "no evidence of exclusion X" as "proceed", not "block".
        # The confidence band and rationale surface unverified exclusions for manual
        # confirmation. Score threshold lowered 80→75 because unverified exclusions
        # each subtract 3 pts and would otherwise push real patients below 80.
        if (
            score >= 75
            and data.inclusion_pass_rate >= 0.60
            and data.inclusion_fail == 0
            and data.exclusion_triggered == 0
            # NOTE: data.exclusion_ambiguous intentionally NOT required to be 0
        ):
            logger.info(
                "Verdict: ELIGIBLE (score={} pass_rate={:.0%} | {} exclusions unverified "
                "but none triggered — coordinator should confirm)",
                score, data.inclusion_pass_rate, data.exclusion_ambiguous
            )
            return VerdictType.ELIGIBLE

        # Gate 3: coordinator review
        reasons = []
        if score < 75:
            reasons.append(f"score={score} < 75")
        if data.inclusion_pass_rate < 0.60:
            reasons.append(
                f"inclusion_pass_rate={data.inclusion_pass_rate:.0%} "
                f"({data.inclusion_pass}/{data.inclusion_total} confirmed)"
            )
        if data.inclusion_ambiguous > 0:
            reasons.append(f"inclusion_ambiguous={data.inclusion_ambiguous}")
        logger.info("Verdict: REVIEW_NEEDED — {}", "; ".join(reasons))
        return VerdictType.REVIEW_NEEDED

    def _evaluate_rule(
        self, rule: CriterionRule, patient: PatientData
    ) -> tuple[EvaluationStatus, str, str]:
        concept  = rule.concept
        operator = rule.operator
        value    = rule.value or ""
        concept_lower = concept.lower().strip()

        # ── Gender / sex ───────────────────────────────────────────────────
        if concept_lower in ("gender", "sex", "male sex", "female sex", "male sex required"):
            if patient.gender is None:
                return EvaluationStatus.AMBIGUOUS, "Patient gender unknown", ""
            pg = patient.gender.lower()
            expected = value.lower() if value else ""
            if not expected:
                if "male" in concept_lower and "female" not in concept_lower:
                    expected = "male"
                elif "female" in concept_lower:
                    expected = "female"
            if not expected:
                return EvaluationStatus.AMBIGUOUS, "Gender criterion has no expected value", pg
            if pg == expected:
                return EvaluationStatus.PASS, f"Patient gender '{pg}' matches required '{expected}'", pg
            return EvaluationStatus.FAIL, f"Patient gender '{pg}' does not match required '{expected}'", pg

        # ── Age ────────────────────────────────────────────────────────────
        if concept_lower in ("age",):
            if patient.age is None:
                return EvaluationStatus.AMBIGUOUS, "Patient age unknown", ""
            result = _compare_lab(float(patient.age), operator, value, concept)
            if result is True:
                return EvaluationStatus.PASS, f"Patient age {patient.age} satisfies {operator} {value}", str(patient.age)
            elif result is False:
                return EvaluationStatus.FAIL, f"Patient age {patient.age} does NOT satisfy {operator} {value}", str(patient.age)
            else:
                return EvaluationStatus.AMBIGUOUS, f"Cannot compare age {patient.age} to '{value}'", str(patient.age)

        # ── Lab result numeric comparison ──────────────────────────────────
        matched_labs = [lab for lab in patient.lab_results if self._lab_name_matches(lab, concept)]
        if matched_labs:
            lab = matched_labs[-1]
            val_str = f"{lab.value} {lab.unit}"
            rule_unit = getattr(rule, "unit", None)
            result = _compare_lab(lab.value, operator, value, concept, lab.unit, rule_unit)
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
            # result is None → non-numeric threshold; fall through to presence check

        # ── OR condition ───────────────────────────────────────────────────
        if operator == "one_of":
            try:
                if value.strip().startswith("["):
                    try:
                        alternatives = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        # Protocol values are sometimes stored as Python list literals
                        # with single quotes (e.g. "['current pregnancy', ...]") rather
                        # than valid JSON.  ast.literal_eval handles both.
                        import ast
                        alternatives = ast.literal_eval(value)
                else:
                    alternatives = [v.strip() for v in value.split("|")]
            except Exception:
                logger.warning("one_of rule has unparseable value: '{}' → AMBIGUOUS", value)
                return EvaluationStatus.AMBIGUOUS, "Could not parse OR condition alternatives", ""

            for alternative in alternatives:
                found, item, score = self._concept_in_list(
                    alternative, patient.conditions + patient.medications
                )
                if found and score >= SNOMED_THRESHOLD_SIGNAL:
                    return (
                        EvaluationStatus.PASS,
                        f"Patient meets alternative: '{alternative}'",
                        item,
                    )
            return (
                EvaluationStatus.AMBIGUOUS,
                f"None of {len(alternatives)} OR alternatives confirmed in patient record",
                "",
            )

        # ── Conditional criterion ──────────────────────────────────────────
        if operator == "conditional":
            return (
                EvaluationStatus.AMBIGUOUS,
                f"Conditional criterion requires manual assessment: '{rule.criterion_text}'",
                "Conditional logic — coordinator review required",
            )

        # ── Presence / history_of ──────────────────────────────────────────
        # ── Presence / history_of ──────────────────────────────────────────
        if operator in ("presence", "history_of"):
            found, item, score = self._concept_in_list(concept, patient.conditions + patient.medications)
            if found:
                final = self._snomed_match_status(score, "PASS")
                if final == "PASS":
                    # --- NEW TEMPORAL CHECK FOR PRESENCE ---
                    tf_val = getattr(rule, "timeframe_value", None)
                    tf_unit = getattr(rule, "timeframe_unit", None)
                    
                    if tf_val is not None and tf_unit:
                        cutoff = _timeframe_cutoff(tf_val, tf_unit)
                        if cutoff is not None:
                            all_records = list(patient.condition_records) + list(patient.medication_records)
                            within = _record_within_window(all_records, concept.lower(), cutoff)
                            
                            if not within:
                                # The condition exists, but it is OLDER than the required window.
                                msg = f"Found '{item}' but onset predates the {tf_val}-{tf_unit} window (cutoff {cutoff})"
                                logger.info("  TEMPORAL MISS (PRESENCE): {}", msg)
                                # Fails the presence requirement (which is a GOOD thing if it's an exclusion)
                                return EvaluationStatus.FAIL, msg, item
                                
                            return EvaluationStatus.PASS, f"Found '{item}' within the required {tf_val}-{tf_unit} window", item
                    # ----------------------------------------
                    
                    return EvaluationStatus.PASS, f"Found '{item}' in patient record", item
                return EvaluationStatus.AMBIGUOUS, f"Weak match for '{concept}' in patient record — uncertain", item
            return EvaluationStatus.AMBIGUOUS, f"No evidence of '{concept}' in patient record", ""
        # ── Absence operators ──────────────────────────────────────────────
        # _evaluate_rule returns RAW status in PRESENCE semantics (PASS = concept
        # detected in the patient). The downstream inversion (for exclusion rules)
        # and _calculate_fit_score both assume this convention, so absence/­
        # no_history_of operators must report the same presence signal — the
        # "must be absent" intent is encoded by the rule's criterion_type, not here.
        if operator in ("absence", "no_history_of", "absence_or_remote", "absence_or_historical"):
            is_inclusion = (
                rule.criterion_type == CriterionType.inclusion
                or (rule.criterion_type != CriterionType.exclusion and rule.required)
            )
            found, item, score = self._concept_in_list(concept, patient.conditions + patient.medications)
            if found:
                # Weak (uncertain) semantic match → AMBIGUOUS either way.
                if self._snomed_match_status(score, "PASS") != "PASS":
                    return EvaluationStatus.AMBIGUOUS, f"Possible match for '{concept}' — uncertain (score={score:.2f})", item

                # ── Temporal window check ───────────────────────────────────
                # If the rule specifies a time window (e.g. "no stroke in past 6 months"),
                # check whether the matching record actually falls inside the window.
                # A record that pre-dates the window makes the rule a PASS (the event is
                # outside the restricted period), not a disqualifying FAIL/TRIGGER.
                tf_val = getattr(rule, "timeframe_value", None)
                tf_unit = getattr(rule, "timeframe_unit", None)
                if tf_val is not None and tf_unit:
                    cutoff = _timeframe_cutoff(tf_val, tf_unit)
                    if cutoff is not None:
                        all_records = list(patient.condition_records) + list(patient.medication_records)
                        within = _record_within_window(all_records, concept.lower(), cutoff)
                        if not within:
                            # Concept exists in patient history but the recorded date
                            # is outside (older than) the rule's time window → no violation.
                            msg = (
                                f"Found '{item}' but onset predates the {tf_val}-{tf_unit} "
                                f"window (cutoff {cutoff}) — temporal constraint satisfied"
                            )
                            logger.info("  TEMPORAL PASS: {}", msg)
                            if is_inclusion:
                                return EvaluationStatus.PASS, msg, item
                            return EvaluationStatus.FAIL, msg, item
                        logger.info(
                            "  TEMPORAL FAIL: '{}' onset is within the {}-{} window",
                            item, tf_val, tf_unit,
                        )

                if is_inclusion:
                    # Inclusion "must not have X": presence of X violates it → FAIL.
                    return EvaluationStatus.FAIL, f"Found '{item}' which must be absent", item
                # Exclusion: presence of X means the exclusion is triggered. Report
                # PRESENCE (PASS); evaluate_patient inverts to a FAIL for display.
                return EvaluationStatus.PASS, f"Found '{item}' (excluded — must be absent)", item
            # Concept not detected → confirmed absent.
            if is_inclusion:
                return EvaluationStatus.PASS, f"No evidence of '{concept}' found (satisfies absence criterion)", ""
            return EvaluationStatus.FAIL, f"No evidence of '{concept}' found (exclusion not triggered)", ""

        # ── General fallback ───────────────────────────────────────────────
        found_cond, item_cond, score_cond = self._concept_in_list(concept, patient.conditions)
        found_med,  item_med,  score_med  = self._concept_in_list(concept, patient.medications)

        if found_cond or found_med:
            item       = item_cond or item_med
            best_score = max(score_cond, score_med)
            final      = self._snomed_match_status(best_score, "PASS")
            
            if final == "PASS":
                # Ensure we handle list values extracted by the LLM (e.g., ['male', 'female'])
                # We attempt to safely parse string representations of lists first
                parsed_value = value
                if value.strip().startswith("[") and value.strip().endswith("]"):
                    try:
                        import ast
                        parsed_value = ast.literal_eval(value)
                    except Exception:
                        pass # Fallback to original string if it fails to parse

                # If the expected value is a list of allowed options
                if isinstance(parsed_value, list):
                     # Treat it as a PASS if the found item is broadly matched by any option in the list
                     # (Since concept_in_list already confirmed presence of the concept itself)
                     return EvaluationStatus.PASS, f"Found matching record: '{item}' (matches one of allowed options)", item

                # If it's a numeric comparison
                result = _compare_lab(1.0, operator, value, concept) if _extract_numeric(value) else None
                if result is None:
                    return EvaluationStatus.PASS, f"Found matching record: '{item}'", item
                
                return (
                    EvaluationStatus.PASS if result else EvaluationStatus.FAIL,
                    f"Found '{item}' with value comparison",
                    item,
                )
            return EvaluationStatus.AMBIGUOUS, f"Weak match for '{concept}' — uncertain (score={best_score:.2f})", item

        return EvaluationStatus.AMBIGUOUS, f"No data found for '{concept}' in patient record", ""
    async def evaluate_patient(
        self, patient: PatientData, rules: list[CriterionRule]
    ) -> ScoringResult:
        logger.info(
            "=== SCORING PATIENT {} AGAINST {} RULES ===", patient.patient_id, len(rules)
        )

        # Phase 1: run all deterministic rule evaluations synchronously
        phase1: list[tuple[EvaluationStatus, str, str, CriterionRule]] = []
        for rule in rules:
            raw_status, explanation, data_found = self._evaluate_rule(rule, patient)
            phase1.append((raw_status, explanation, data_found, rule))

        # --- ADD THIS SEMAPHORE ---
        # Limit to 3 concurrent API calls to prevent Anthropic Rate Limit (429) crashes
        semaphore = asyncio.Semaphore(3) 

        # Phase 2: fire all agentic fallback calls concurrently for ambiguous/failed rules
        # Phase 2: fire all agentic fallback calls concurrently
        async def _maybe_fallback(
            idx: int,
            raw_status: EvaluationStatus,
            explanation: str,
            rule: CriterionRule,
        ) -> tuple[int, EvaluationStatus, str]:
            if raw_status not in (EvaluationStatus.AMBIGUOUS, EvaluationStatus.FAIL):
                return idx, raw_status, explanation
            if rule.concept.lower() in ("age", "gender", "sex"):
                return idx, raw_status, explanation
            
            # Use the semaphore to limit concurrent connections
            async with semaphore:
                try:
                    # --- FIX: ARTIFICIAL DELAY FOR API RATE LIMITS ---
                    # Forces the engine to wait 2 seconds between requests to prevent 429 errors
                    await asyncio.sleep(2) 
                    # -------------------------------------------------

                    new_status, ai_explanation = await fallback_agent.verify_evaluation(
                        rule=rule,
                        patient=patient,
                        current_status=raw_status,
                        python_rationale=explanation,
                    )
                    if new_status != raw_status:
                        return (
                            idx,
                            new_status,
                            f"✨ [AI Agent Override]: {ai_explanation} (Python originally said: {explanation})",
                        )
                except Exception as e:
                    logger.error("Agentic fallback failed for rule {}: {}", rule.concept, e)
            return idx, raw_status, explanation

        fallback_coros = [
            _maybe_fallback(i, s, e, r) for i, (s, e, _, r) in enumerate(phase1)
        ]
        fallback_results = await asyncio.gather(*fallback_coros)
        # Apply overrides back into phase1
        for idx, new_status, new_explanation in fallback_results:
            old_status, _, data_found, rule = phase1[idx]
            phase1[idx] = (new_status, new_explanation, data_found, rule)

        # Phase 3: build final evaluations list
        raw_evals: list[tuple[EvaluationStatus, CriterionRule]] = []
        db_evaluations: list[CriterionEvaluationCreate] = []

        for raw_status, explanation, data_found, rule in phase1:
            # Determine if rule is inclusion or exclusion
            is_inclusion = (
                rule.criterion_type == CriterionType.inclusion
                or (rule.criterion_type != CriterionType.exclusion and rule.required)
            )

            # Build DB evaluation with inversion applied for exclusion rules
            db_status      = raw_status
            db_explanation = explanation
            if not is_inclusion:
                if raw_status == EvaluationStatus.PASS:
                    db_status = EvaluationStatus.FAIL
                    db_explanation = f"Patient HAS excluded condition/finding: {explanation}"
                elif raw_status == EvaluationStatus.FAIL:
                    db_status = EvaluationStatus.PASS
                    db_explanation = f"Exclusion criterion not met (good): {explanation}"

            prefix = "INCLUSION" if is_inclusion else "EXCLUSION"
            log_fn = logger.warning if db_status == EvaluationStatus.AMBIGUOUS else logger.info
            log_fn("  [{}] {}: {} — {}", prefix, rule.concept, db_status.value, db_explanation)

            raw_evals.append((raw_status, rule))
            db_evaluations.append(
                CriterionEvaluationCreate(
                    criterion_id=rule.id if hasattr(rule, "id") else None,
                    criterion_text=rule.criterion_text,
                    concept=rule.concept,
                    criterion_type=rule.criterion_type,
                    status=db_status,
                    explanation=db_explanation,
                    data_found=data_found,
                )
            )

        fit_data = self._calculate_fit_score(raw_evals)
        confidence_low, confidence_high = self._calculate_confidence_band(fit_data.fit_score, fit_data)
        verdict = self._determine_verdict(fit_data.fit_score, fit_data)

        logger.info(
            "=== FINAL SCORE: {}/100 | Band: {}-{} | Verdict: {} "
            "| incl_fail={} excl_triggered={} incl_ambig={} excl_ambig={} ===",
            fit_data.fit_score,
            confidence_low,
            confidence_high,
            verdict.value,
            fit_data.inclusion_fail,
            fit_data.exclusion_triggered,
            fit_data.inclusion_ambiguous,
            fit_data.exclusion_ambiguous,
        )

        score_breakdown = {
            "inclusion_pass":      fit_data.inclusion_pass,
            "inclusion_fail":      fit_data.inclusion_fail,
            "inclusion_ambiguous": fit_data.inclusion_ambiguous,
            "inclusion_total":     fit_data.inclusion_total,
            "inclusion_pass_rate": round(fit_data.inclusion_pass_rate, 4),
            "exclusion_triggered": fit_data.exclusion_triggered,
            "exclusion_ambiguous": fit_data.exclusion_ambiguous,
            "exclusion_clear":     fit_data.exclusion_clear,
            "deductions":          fit_data.deductions,
        }

        return ScoringResult(
            fit_score=fit_data.fit_score,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            overall_verdict=verdict,
            evaluations=db_evaluations,
            score_breakdown=score_breakdown,
        )


scoring_engine = ScoringEngine()
