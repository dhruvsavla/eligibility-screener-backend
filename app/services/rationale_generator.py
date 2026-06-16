import json
import time
from loguru import logger
from app.services.llm_client import get_claude_client
from app.services.scoring_engine import UNVERIFIABLE_INCLUSION_PATTERNS
from app.models.patient import PatientData
from app.models.protocol import CriterionRule
from app.models.screening import CriterionEvaluationCreate


def _is_unverifiable_concept(concept: str, criterion_text: str) -> bool:
    combined = (concept + " " + criterion_text).lower()
    return any(pat in combined for pat in UNVERIFIABLE_INCLUSION_PATTERNS)

SYSTEM_PROMPT = """You are a clinical trial coordinator assistant. Given a patient summary and
their eligibility screening results, write a concise plain-English rationale card.
Your output is read by site coordinators who are NOT data scientists.
Format:
- Start with a one-sentence verdict summary
- List PASS criteria briefly (2-3 words each)
- List FAIL criteria with a specific explanation of WHY they failed
- List AMBIGUOUS criteria with what data is missing
- For inclusion criteria that require coordinator verification (temporal stability,
  patient willingness, administrative consent, or compound age+disease criteria),
  list them under "Requires manual verification:" — these could not be confirmed
  from the structured patient record and must be assessed during the screening visit.
- If the verdict is ELIGIBLE but some exclusion criteria could not be verified from
  the patient record (status AMBIGUOUS), you MUST list those unverified exclusions
  explicitly under a "Could not verify — please confirm:" heading. The coordinator
  needs to know which disqualifying conditions were assumed absent due to missing
  data rather than confirmed absent.
- End with a recommended action (Proceed / Do Not Enroll / Manual Review Required)
Keep total length under 350 words. Use plain language, no jargon."""


class RationaleGenerator:
    def __init__(self):
        # Claude client is created lazily via get_claude_client().
        pass

    @property
    def claude(self):
        return get_claude_client()

    def generate(
        self,
        patient: PatientData,
        rules: list[CriterionRule],
        evaluations: list[CriterionEvaluationCreate],
        fit_score: int,
        verdict: str,
    ) -> str:
        logger.info(
            "Generating rationale for patient {} (score: {}, verdict: {}) via Claude Sonnet",
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
            rationale = self.claude.complete(SYSTEM_PROMPT, user_prompt, max_tokens=600)
            elapsed = int((time.time() - start) * 1000)
            logger.info("✓ Rationale generated ({} chars) in {}ms", len(rationale), elapsed)
            for line in rationale.split("\n")[:5]:
                if line.strip():
                    logger.info("  RATIONALE: {}", line[:120])
            return rationale
        except Exception as e:
            logger.error("Error generating rationale via Claude Sonnet: {}", e)
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
        ]

        # Unverifiable inclusion criteria — surface for all verdicts
        unverifiable_incl = [
            e for e in evaluations
            if str(getattr(e.criterion_type, "value", e.criterion_type)) == "inclusion"
            and str(getattr(e.status, "value", e.status)) == "AMBIGUOUS"
            and _is_unverifiable_concept(e.concept, e.criterion_text or "")
        ]
        if unverifiable_incl:
            lines.append("")
            lines.append("Requires manual verification (not confirmable from structured record):")
            for e in unverifiable_incl:
                lines.append(f"  - {e.concept}")

        if verdict == "ELIGIBLE":
            ambiguous_excl = [
                e for e in evaluations
                if str(getattr(e.criterion_type, "value", e.criterion_type)) == "exclusion"
                and str(getattr(e.status, "value", e.status)) == "AMBIGUOUS"
            ]
            if ambiguous_excl:
                lines.append("")
                lines.append("Could not verify — please confirm before enrolling:")
                for e in ambiguous_excl:
                    lines.append(f"  - {e.concept}: {e.explanation}")

        lines += [
            "",
            "Recommended action: "
            + ("Proceed" if verdict == "ELIGIBLE" else "Do Not Enroll" if verdict == "INELIGIBLE" else "Manual Review Required"),
        ]
        return "\n".join(lines)


rationale_generator = RationaleGenerator()
