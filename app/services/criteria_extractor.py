import time
from loguru import logger
from app.services.llm_client import get_claude_client
from app.models.protocol import CriterionRuleCreate, CriterionType

SYSTEM_PROMPT = """You are a clinical trial eligibility criteria parser with expertise in regulatory and protocol documents.

Extract every inclusion and exclusion criterion from the given text.
For each criterion, return a JSON object with these EXACT fields:

{
  "criterion_text": "the original sentence or phrase verbatim",
  "concept": "the core medical concept (e.g. 'HbA1c', 'age', 'eGFR', 'prior_chemotherapy', 'pregnancy')",
  "operator": "one of the operators listed below",
  "value": "the threshold, target value, or alternatives as a string",
  "required": true for inclusion criteria (patient MUST have), false for exclusion criteria (patient MUST NOT have),
  "criterion_type": "inclusion" or "exclusion",
  "confidence": float 0.0-1.0 representing your confidence in the extraction
}

OPERATOR DEFINITIONS — choose the most specific one that applies:

Numeric operators (for lab values, age, BMI, etc.):
  ">"          patient value must be greater than threshold
  ">="         patient value must be greater than or equal to threshold
  "<"          patient value must be less than threshold
  "<="         patient value must be less than or equal to threshold
  "=="         patient value must equal threshold exactly
  "!="         patient value must not equal threshold
  "between"    patient value must be within a range (value: "low-high", e.g. "18-75")

Presence/absence operators (for conditions, medications, procedures, history):
  "presence"       patient must have this condition/medication/history
  "absence"        patient must NOT have this condition/medication/history (hard exclusion)
  "history_of"     patient must have a documented history of this
  "no_history_of"  patient must have NO documented history of this

Special operators — READ THESE CAREFULLY:
  "one_of"      OR condition — patient must satisfy AT LEAST ONE of several alternatives
                value MUST be a JSON array: ["alternative A", "alternative B", "alternative C"]
                USE THIS when the criterion says "A or B", "either X or Y", "at least one of"
                NEVER split an OR condition into separate rules — that would wrongly require ALL

  "conditional"  Conditional criterion — "if X then Y must also be true"
                Set required=false and confidence=0.5 for these
                value: describe the condition in plain English

CRITICAL RULES — FOLLOW THESE EXACTLY:

1. OR CONDITIONS: If a criterion says "patients must have A OR B" — extract as ONE rule with
   operator="one_of" and value=["A", "B"].
   WRONG: two separate rules for A and B (that would require BOTH)
   RIGHT: one rule with operator="one_of" value=["A", "B"]

2. RANGES: "age 18 to 75 years" → operator="between" value="18-75"
   "HbA1c between 7.5% and 11%" → operator="between" value="7.5-11"

3. NEGATIONS: "no prior chemotherapy" → operator="no_history_of" concept="chemotherapy"
   "must not be pregnant" → operator="absence" concept="pregnancy" criterion_type="exclusion"

4. NUMERIC THRESHOLDS: Always strip units from value. "eGFR >= 60 mL/min" → value="60" not "60 mL/min"

5. CONFIDENCE: Set confidence < 0.7 if the criterion is ambiguous, uses medical shorthand
   you are uncertain about, or has complex conditional logic.

IMPORTANT — BOTH LISTS ARE REQUIRED:
Protocols contain a separate "Inclusion Criteria" list AND a separate "Exclusion Criteria"
list. The exclusion list appears AFTER the inclusion list, often continuing the same
numbering (e.g. inclusion [1]–[9], exclusion [10]–[31]). You MUST extract criteria from
BOTH lists. If you produce only inclusion criteria with zero exclusion criteria, you have
missed the exclusion list — re-read the text and find it. A well-formed protocol typically
has as many or more exclusion criteria as inclusion criteria.

Return ONLY a valid JSON array. No markdown, no preamble, no explanation outside the JSON."""


class CriteriaExtractor:
    def __init__(self):
        # Claude client is created lazily via get_claude_client() so importing this
        # module never requires the ANTHROPIC_API_KEY to be present.
        pass

    @property
    def claude(self):
        return get_claude_client()

    def _coerce_rules(self, data, nct_id: str) -> list[CriterionRuleCreate]:
        if isinstance(data, dict):
            # tolerate {"criteria": [...]} or a single object
            if "criteria" in data and isinstance(data["criteria"], list):
                data = data["criteria"]
            else:
                data = [data]
        if not isinstance(data, list):
            logger.error("Claude returned non-list JSON for trial {}: {}", nct_id, type(data))
            return []

        rules = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                c_type = str(item.get("criterion_type", "inclusion")).lower()
                rule = CriterionRuleCreate(
                    criterion_text=item.get("criterion_text", ""),
                    concept=item.get("concept", ""),
                    operator=item.get("operator", "presence"),
                    value=str(item.get("value", "")),
                    required=bool(item.get("required", True)),
                    criterion_type=CriterionType(c_type if c_type in ("inclusion", "exclusion") else "inclusion"),
                    confidence=float(item.get("confidence", 0.5)),
                )
                rules.append(rule)
            except Exception as parse_err:
                logger.warning("Skipping malformed criterion item: {} — {}", item, parse_err)

        return rules

    def _build_user_prompt(
        self, raw_criteria_text: str, nct_id: str, ner_entities: list[dict] | None = None
    ) -> str:
        hint_block = ""
        if ner_entities:
            entity_names = sorted({e["text"] for e in ner_entities if e.get("text")})
            if entity_names:
                logger.info(
                    "scispaCy pre-annotation: {} entities passed to Claude as extraction hints",
                    len(entity_names),
                )
                hint_block = (
                    "\nscispaCy detected the following clinical entities in this text (use as hints,\n"
                    "but extract ALL criteria including any entities it may have missed):\n"
                    + ", ".join(entity_names)
                    + "\n"
                )

        return f"""Extract eligibility criteria from trial {nct_id}.

EXAMPLE of correct extraction (do not include this in your output):
Input: "Patients must be aged 18-65 years and have HbA1c >= 7.5%, or have been
        diagnosed with Type 1 or Type 2 diabetes. No prior insulin therapy."
Correct output:
[
  {{"criterion_text": "aged 18-65 years", "concept": "age", "operator": "between",
    "value": "18-65", "required": true, "criterion_type": "inclusion", "confidence": 0.99}},
  {{"criterion_text": "HbA1c >= 7.5%", "concept": "HbA1c", "operator": ">=",
    "value": "7.5", "required": true, "criterion_type": "inclusion", "confidence": 0.98}},
  {{"criterion_text": "Type 1 or Type 2 diabetes", "concept": "diabetes_type",
    "operator": "one_of", "value": ["Type 1 diabetes mellitus", "Type 2 diabetes mellitus"],
    "required": true, "criterion_type": "inclusion", "confidence": 0.97}},
  {{"criterion_text": "No prior insulin therapy", "concept": "insulin",
    "operator": "no_history_of", "value": "insulin therapy",
    "required": false, "criterion_type": "exclusion", "confidence": 0.96}}
]
{hint_block}
Now extract from this trial's criteria text:

{raw_criteria_text}"""

    def extract(
        self,
        raw_criteria_text: str,
        nct_id: str,
        ner_entities: list[dict] | None = None,
    ) -> list[CriterionRuleCreate]:
        if not raw_criteria_text or not raw_criteria_text.strip():
            logger.warning("Empty criteria text for trial {}", nct_id)
            return []

        logger.info(
            "Sending criteria text to Claude Sonnet ({} chars) for trial {}",
            len(raw_criteria_text), nct_id,
        )

        user_prompt = self._build_user_prompt(raw_criteria_text, nct_id, ner_entities)

        try:
            start = time.time()
            # Scale max_tokens with input size: large protocols (PDF, >5k chars) can
            # produce 30+ criteria; 8192 tokens prevents JSON truncation mid-array.
            output_tokens = 8192 if len(raw_criteria_text) > 5000 else 4096
            data = self.claude.complete_json(SYSTEM_PROMPT, user_prompt, max_tokens=output_tokens)
            elapsed = int((time.time() - start) * 1000)
            logger.info("Claude Sonnet responded in {}ms for trial {}", elapsed, nct_id)
        except Exception as e:
            logger.error("Unexpected error calling Claude Sonnet for trial {}: {}", nct_id, e)
            return []

        rules = self._coerce_rules(data, nct_id)

        logger.info("✓ Claude Sonnet returned {} criterion rules for trial {}", len(rules), nct_id)
        for rule in rules:
            logger.info(
                "  [{}] {} {} {} (confidence: {:.2f})",
                rule.criterion_type.upper(),
                rule.concept,
                rule.operator,
                rule.value,
                rule.confidence,
            )

        return rules


criteria_extractor = CriteriaExtractor()
