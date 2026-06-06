import json
import time
from loguru import logger
from openai import OpenAI, APIError
from app.config import settings
from app.models.protocol import CriterionRuleCreate, CriterionType

SYSTEM_PROMPT = """You are a clinical trial eligibility criteria parser.
Extract every inclusion and exclusion criterion from the given text.
For each criterion return a JSON object with these exact fields:
  - criterion_text: the original sentence
  - concept: the core medical concept (e.g. "HbA1c", "age", "eGFR", "prior_chemotherapy")
  - operator: one of [">", ">=", "<", "<=", "==", "!=", "between", "presence", "absence", "history_of", "no_history_of"]
  - value: the threshold or target value as a string (e.g. "7.5%", "18-65 years", "insulin")
  - required: true for inclusion criteria, false for exclusion criteria
  - criterion_type: "inclusion" or "exclusion"
  - confidence: float 0.0-1.0 how confident you are in the extraction
Return ONLY a JSON array. No preamble, no markdown."""


class CriteriaExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _parse_response(self, content: str, nct_id: str) -> list[CriterionRuleCreate]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("JSON parse error for trial {}: {} — raw: {}...", nct_id, e, content[:200])
            return []

        rules = []
        for item in data:
            try:
                c_type = item.get("criterion_type", "inclusion").lower()
                rule = CriterionRuleCreate(
                    criterion_text=item.get("criterion_text", ""),
                    concept=item.get("concept", ""),
                    operator=item.get("operator", "presence"),
                    value=str(item.get("value", "")),
                    required=bool(item.get("required", True)),
                    criterion_type=CriterionType(c_type),
                    confidence=float(item.get("confidence", 0.5)),
                )
                rules.append(rule)
            except Exception as parse_err:
                logger.warning("Skipping malformed criterion item: {} — {}", item, parse_err)

        return rules

    def extract(self, raw_criteria_text: str, nct_id: str) -> list[CriterionRuleCreate]:
        if not raw_criteria_text or not raw_criteria_text.strip():
            logger.warning("Empty criteria text for trial {}", nct_id)
            return []

        logger.info(
            "Sending criteria text to GPT-4o ({} chars) for trial {}", len(raw_criteria_text), nct_id
        )

        user_prompt = f"Extract eligibility criteria from this trial {nct_id}:\n\n{raw_criteria_text}"

        try:
            start = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            elapsed = int((time.time() - start) * 1000)
            content = response.choices[0].message.content or ""
            logger.info("GPT-4o responded in {}ms for trial {}", elapsed, nct_id)
        except APIError as e:
            logger.error("OpenAI APIError for trial {}: {}", nct_id, e)
            return []
        except Exception as e:
            logger.error("Unexpected error calling OpenAI for trial {}: {}", nct_id, e)
            return []

        rules = self._parse_response(content, nct_id)

        if not rules and content:
            logger.warning("Retrying GPT-4o with explicit JSON reminder for trial {}", nct_id)
            try:
                retry_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": "Respond only with a valid JSON array. No markdown, no explanation.",
                        },
                    ],
                    temperature=0.0,
                    max_tokens=4000,
                )
                rules = self._parse_response(
                    retry_response.choices[0].message.content or "", nct_id
                )
            except Exception as retry_err:
                logger.error("Retry also failed for trial {}: {}", nct_id, retry_err)
                return []

        logger.info("✓ GPT-4o returned {} criterion rules for trial {}", len(rules), nct_id)
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
