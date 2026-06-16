"""
LangChain document agent for protocol criteria extraction.

This agent orchestrates the full protocol-to-rules pipeline using tools:
  1. chunk_document      — splits long protocol PDFs into sections
  2. locate_eligibility  — finds the I/E criteria section in the document
  3. extract_entities    — runs scispaCy NER on the located section
  4. lookup_snomed       — maps concepts to SNOMED/OMOP codes during extraction
  5. extract_criteria    — Claude structured extraction into rule JSON

Uses langchain-anthropic ChatAnthropic with Claude Sonnet as the agent LLM.
The agent reasons over which tools to call, enabling it to handle both short
API-text criteria and long multi-page PDF protocols.
"""
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import settings
from app.services.ner_service import get_ner_service
from app.services.concept_matcher import get_concept_matcher
from app.services.criteria_extractor import CriteriaExtractor


class ProtocolDocumentAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=settings.ANTHROPIC_MAX_TOKENS,
        )
        self.ner = get_ner_service()
        self.matcher = get_concept_matcher()
        self.extractor = CriteriaExtractor()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=400,
            separators=["\n\nInclusion", "\n\nExclusion", "\n\n", "\n", ". "]
        )
        self._tools = self._build_tools()
        self._agent_executor = self._build_agent()
        logger.info("ProtocolDocumentAgent initialized with {} tools", len(self._tools))

    def _build_tools(self):
        ner = self.ner
        matcher = self.matcher
        splitter = self.splitter

        @tool
        def chunk_document(text: str) -> str:
            """Split a long protocol document into chunks. Returns chunk count and previews."""
            chunks = splitter.split_text(text)
            logger.info("[agent:chunk_document] split into {} chunks", len(chunks))
            return f"Document split into {len(chunks)} chunks. " + \
                   "; ".join(f"chunk{i}: {c[:80]}..." for i, c in enumerate(chunks[:5]))

        @tool
        def locate_eligibility(text: str) -> str:
            """Find and return the eligibility/inclusion/exclusion section from protocol text."""
            from app.services.pdf_protocol_parser import PDFProtocolParser
            section = PDFProtocolParser().locate_eligibility_section(text, max_section_chars=40000)
            logger.info("[agent:locate_eligibility] located section: {} chars", len(section))
            return section

        @tool
        def extract_clinical_entities(text: str) -> str:
            """Run scispaCy clinical NER on text. Returns detected medical entities."""
            ents = ner.extract_entities(text)
            logger.info("[agent:extract_clinical_entities] {} entities", len(ents))
            return ", ".join(e["text"] for e in ents) or "no entities detected"

        @tool
        def lookup_snomed(concept: str) -> str:
            """Map a clinical concept to its best SNOMED/OMOP match with hierarchy."""
            matches = matcher.find_best_match(concept, top_k=2)
            logger.info("[agent:lookup_snomed] '{}' → {}", concept, matches)
            if not matches:
                return f"no SNOMED match for '{concept}'"
            top = matches[0]
            return f"{concept} → {top['term']} (code {top.get('code','?')}, score {top['score']:.2f})"

        return [chunk_document, locate_eligibility, extract_clinical_entities, lookup_snomed]

    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a clinical trial protocol document agent. Your job is to read a "
             "protocol document and prepare it for structured criteria extraction. "
             "Use your tools to: locate the eligibility section in long documents, "
             "run clinical NER to detect medical entities, and look up SNOMED codes for "
             "key concepts. Then summarize the located eligibility section and the key "
             "clinical entities you found. Be thorough — do not skip the exclusion criteria."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(self.llm, self._tools, prompt)
        return AgentExecutor(agent=agent, tools=self._tools, verbose=True,
                             max_iterations=8, handle_parsing_errors=True)

    def process_protocol(self, raw_text: str, nct_id: str) -> dict:
        """
        Full agent-orchestrated pipeline:
          1. Locate eligibility section in FULL text (handles long PDFs)
          2. Agent reasons over the located section (not a blind prefix of raw_text)
          3. CriteriaExtractor (Claude) does final structured rule extraction
        Returns: {rules: list[CriterionRuleCreate], agent_trace: str, entities: list}
        """
        logger.info("=== LANGCHAIN AGENT processing protocol {} ({:,} chars raw) ===",
                    nct_id, len(raw_text))

        # Locate eligibility section FIRST — works on full text, skips ToC entries.
        from app.services.pdf_protocol_parser import PDFProtocolParser
        located = PDFProtocolParser().locate_eligibility_section(raw_text)
        logger.info("Located eligibility section: {:,} chars (of {:,} raw)", len(located), len(raw_text))

        # Step 1: agent reasons over the LOCATED section (not raw_text[:12000]).
        agent_trace = ""
        try:
            agent_input = (
                f"Process this clinical trial protocol ({nct_id}). The eligibility criteria "
                f"section is below. Extract ALL inclusion AND exclusion criteria — do not stop "
                f"after the inclusion list.\n\n{located}"
            )
            agent_result = self._agent_executor.invoke({"input": agent_input})
            raw_output = agent_result.get("output", "")
            # LangChain-Anthropic can return a list of content blocks when Claude
            # uses parallel tool calls; flatten to a plain string for SQLite storage.
            if isinstance(raw_output, list):
                agent_trace = " ".join(
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in raw_output
                )
            else:
                agent_trace = str(raw_output)
            logger.info("[agent] reasoning complete: {} chars of output", len(agent_trace))
        except Exception as e:
            # Agent reasoning is for transparency; never let it block rule extraction.
            agent_trace = f"[agent reasoning unavailable: {e}]"
            logger.warning("[agent] reasoning step failed: {} — proceeding to extraction", e)

        # Step 2: structured extraction using the located section.
        entities = self.ner.extract_entities(located)
        rules = self.extractor.extract(located, nct_id, ner_entities=entities)

        logger.info("=== AGENT produced {} structured rules for {} ===", len(rules), nct_id)
        return {"rules": rules, "agent_trace": agent_trace, "entities": entities}


_agent: ProtocolDocumentAgent | None = None


def get_protocol_agent() -> ProtocolDocumentAgent:
    global _agent
    if _agent is None:
        _agent = ProtocolDocumentAgent()
    return _agent
