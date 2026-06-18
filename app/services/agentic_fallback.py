import json
from loguru import logger
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from app.models.patient import PatientData
from app.models.protocol import CriterionRule
from app.models.screening import EvaluationStatus

class AgenticFallbackService:
    def __init__(self):
        # Using Claude Sonnet as requested in your original specs
        self.llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0.0)
        
        self.prompt = PromptTemplate.from_template("""
        You are a Senior Clinical Trial Investigator. 
        Your deterministic Python engine just evaluated a patient against a trial criterion and flagged it as {current_status}.
        
        We need you to act as a fallback safety net to ensure the Python engine didn't miss a clinical nuance, exception, or compound logic in the original text.
        
        --- PROTOCOL RULE ---
        Original Text: "{criterion_text}"
        Extracted Logic: {concept} {operator} {value}
        
        --- PATIENT DATA SUMMARY ---
        Conditions: {patient_conditions}
        Medications: {patient_medications}
        Labs: {patient_labs}
        
        --- PYTHON ENGINE'S RATIONALE ---
        Reason for Failure/Ambiguity: "{python_rationale}"
        
        --- YOUR TASK ---
        Review the patient's data against the ORIGINAL text. Did the Python engine miss an exception? Does the patient actually pass based on clinical reasoning?
        
        Return ONLY a JSON object with two keys:
        - "status": Must be exactly "PASS", "FAIL", or "AMBIGUOUS".
        - "rationale": A 1-sentence clinical explanation for your decision.
        """)

    def verify_evaluation(
        self, 
        rule: CriterionRule, 
        patient: PatientData, 
        current_status: EvaluationStatus, 
        python_rationale: str
    ) -> tuple[EvaluationStatus, str]:
        
        # 1. Summarize the patient chart so we don't blow up the context window
        conditions = [c for c in patient.conditions]
        medications = [m for m in patient.medications]
        labs = [f"{l.name}: {l.value} {l.unit}" for l in patient.lab_results]

        # 2. Ask Claude for a second opinion
        chain = self.prompt | self.llm
        try:
            response = chain.invoke({
                "current_status": current_status.value,
                "criterion_text": rule.criterion_text,
                "concept": rule.concept,
                "operator": rule.operator,
                "value": rule.value,
                "patient_conditions": ", ".join(conditions) if conditions else "None",
                "patient_medications": ", ".join(medications) if medications else "None",
                "patient_labs": ", ".join(labs) if labs else "None",
                "python_rationale": python_rationale
            })
            
            # 3. Parse the JSON response
            result = json.loads(response.content)
            
            # Map string back to the Enum
            status_map = {
                "PASS": EvaluationStatus.PASS,
                "FAIL": EvaluationStatus.FAIL,
                "AMBIGUOUS": EvaluationStatus.AMBIGUOUS
            }
            new_status = status_map.get(result.get("status", "AMBIGUOUS"), EvaluationStatus.AMBIGUOUS)
            
            logger.info("Agentic Fallback triggered for {}: Changed {} -> {}", rule.concept, current_status.value, new_status.value)
            return new_status, result.get("rationale", "AI Override applied.")
            
        except Exception as e:
            logger.error("Agentic fallback failed: {}", e)
            # If the LLM fails, safely default back to the Python engine's original decision
            return current_status, python_rationale

fallback_agent = AgenticFallbackService()