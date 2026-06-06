import json
import time
from loguru import logger
from openai import OpenAI, APIError
from app.config import settings
from app.models.patient import PatientData
from app.models.protocol import CriterionRule
from app.models.screening import CriterionEvaluationCreate

SYSTEM_PROMPT = """You are a clinical trial coordinator assistant. Given a patient summary and
their eligibility screening results, write a concise plain-English rationale card.
Your output is read by site coordinators who are NOT data scientists.
Format:
- Start with a one-sentence verdict summary
- List PASS criteria briefly (2-3 words each)
- List FAIL criteria with a specific explanation of WHY they failed
- List AMBIGUOUS criteria with what data is missing
- End with a recommended action (Proceed / Do Not Enroll / Manual Review Required)
Keep total length under 300 words. Use plain language, no jargon."""


class RationaleGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate(
        self,
        patient: PatientData,
        rules: list[CriterionRule],
        evaluations: list[CriterionEvaluationCreate],
        fit_score: int,
        verdict: str,
    ) -> str:
        logger.info(
            "Generating rationale for patient {} (score: {}, verdict: {})",
            patient.patient_id,
            fit_score,
            verdict,
        )

        patient_summary = {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "conditions": patient.conditions,
            "medications": patient.medications,
            "lab_results": [
                {"name": l.name, "value": l.value, "unit": l.unit}
                for l in patient.lab_results
            ],
        }
        eval_summary = [
            {
                "criterion": e.criterion_text,
                "concept": e.concept,
                "type": e.criterion_type,
                "status": e.status.value if hasattr(e.status, "value") else e.status,
                "explanation": e.explanation,
                "data_found": e.data_found,
            }
            for e in evaluations
        ]

        user_prompt = json.dumps(
            {
                "fit_score": fit_score,
                "verdict": verdict,
                "patient": patient_summary,
                "evaluations": eval_summary,
            },
            indent=2,
        )

        try:
            start = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            elapsed = int((time.time() - start) * 1000)
            rationale = response.choices[0].message.content or ""
            logger.info("✓ Rationale generated ({} chars) in {}ms", len(rationale), elapsed)
            for line in rationale.split("\n")[:5]:
                if line.strip():
                    logger.info("  RATIONALE: {}", line[:120])
            return rationale
        except APIError as e:
            logger.error("OpenAI APIError generating rationale: {}", e)
            return self._fallback_rationale(patient, evaluations, fit_score, verdict)
        except Exception as e:
            logger.error("Unexpected error generating rationale: {}", e)
            return self._fallback_rationale(patient, evaluations, fit_score, verdict)

    def _fallback_rationale(
        self,
        patient: PatientData,
        evaluations: list[CriterionEvaluationCreate],
        fit_score: int,
        verdict: str,
    ) -> str:
        passes = [e for e in evaluations if str(getattr(e.status, "value", e.status)) == "PASS"]
        fails = [e for e in evaluations if str(getattr(e.status, "value", e.status)) == "FAIL"]
        ambiguous = [e for e in evaluations if str(getattr(e.status, "value", e.status)) == "AMBIGUOUS"]

        lines = [
            f"Patient {patient.name} received a fit score of {fit_score}/100 with verdict: {verdict}.",
            "",
            f"PASS ({len(passes)}): " + ", ".join(e.concept for e in passes[:5]),
            f"FAIL ({len(fails)}): " + "; ".join(f"{e.concept}: {e.explanation}" for e in fails[:3]),
            f"AMBIGUOUS ({len(ambiguous)}): " + ", ".join(e.concept for e in ambiguous[:3]),
            "",
            "Recommended action: "
            + ("Proceed" if verdict == "ELIGIBLE" else "Do Not Enroll" if verdict == "INELIGIBLE" else "Manual Review Required"),
        ]
        return "\n".join(lines)


rationale_generator = RationaleGenerator()
