import json
import re
from loguru import logger
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from app.models.patient import PatientData
from app.models.protocol import CriterionRule
from app.models.screening import EvaluationStatus

class AgenticFallbackService:
    def __init__(self):
        # Switched to Haiku to prevent API Rate Limiting (429) crashes
        self.llm = ChatAnthropic(
            model_name="claude-3-haiku-20240307", 
            temperature=0.0,
            max_tokens=200
        )
        
        self.prompt = PromptTemplate.from_template("""
        You are a Senior Clinical Data Reviewer. 
        Your deterministic Python engine evaluated a patient's chart to find a specific clinical concept, and flagged the result as {current_status}.
        
        We need you to double-check the raw data to see if the Python engine missed it due to acronyms, synonyms, or complex lab reasoning.
        
        --- CONCEPT TO FIND ---
        Original Text: "{criterion_text}"
        Extracted Logic: {concept} {operator} {value}
        
        --- PATIENT DATA SUMMARY ---
        Conditions: {patient_conditions}
        Medications: {patient_medications}
        Labs: {patient_labs}
        
        --- PYTHON ENGINE'S RATIONALE ---
        "{python_rationale}"
        
        --- YOUR TASK ---
        Determine if the specific clinical concept/logic is TRUE or PRESENT in the patient data.
        WARNING: Do NOT evaluate overall trial eligibility. ONLY evaluate if this specific medical concept is met by the data.
        
        Return ONLY a raw JSON object. Do not include markdown formatting, backticks, or conversational text.
        {{
            "status": "PASS", 
            "rationale": "1-sentence clinical explanation."
        }}
        """)

    async def verify_evaluation(
        self,
        rule: CriterionRule,
        patient: PatientData,
        current_status: EvaluationStatus,
        python_rationale: str
    ) -> tuple[EvaluationStatus, str]:

        # 1. Summarize the patient chart
        conditions = [c for c in patient.conditions]
        medications = [m for m in patient.medications]
        labs = [f"{l.name}: {l.value} {l.unit}" for l in patient.lab_results]

        # 2. Ask Claude for a second opinion
        chain = self.prompt | self.llm
        
        raw_content = ""
        try:
            response = await chain.ainvoke({
                "current_status": current_status.value,
                "criterion_text": rule.criterion_text,
                "concept": rule.concept,
                "operator": rule.operator,
                "value": rule.value if rule.value else "N/A",
                "patient_conditions": ", ".join(conditions) if conditions else "None",
                "patient_medications": ", ".join(medications) if medications else "None",
                "patient_labs": ", ".join(labs) if labs else "None",
                "python_rationale": python_rationale
            })
            
            raw_content = response.content.strip()
            
            # --- ROBUST JSON PARSING ---
            # Ignores markdown backticks if Claude accidentally includes them
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                raw_content = json_match.group(0)
                
            result = json.loads(raw_content)
            # ---------------------------
            
            # Map string back to the Enum
            status_map = {
                "PASS": EvaluationStatus.PASS,
                "FAIL": EvaluationStatus.FAIL,
                "AMBIGUOUS": EvaluationStatus.AMBIGUOUS
            }
            new_status = status_map.get(result.get("status", "AMBIGUOUS"), EvaluationStatus.AMBIGUOUS)
            
            logger.info("✨ Agentic Fallback Success for {}: {} -> {}", rule.concept, current_status.value, new_status.value)
            return new_status, result.get("rationale", "AI Override applied.")
            
        except json.JSONDecodeError as e:
            logger.error("🚨 JSON PARSE ERROR in Fallback for '{}'. Claude said: \n{}", rule.concept, raw_content)
            return current_status, python_rationale
        except Exception as e:
            logger.error("🚨 API/NETWORK ERROR in Fallback for '{}': {}", rule.concept, str(e))
            return current_status, python_rationale

fallback_agent = AgenticFallbackService()