"""Tests for the LangChain protocol document agent.

The agent's LLM (ChatAnthropic) and AgentExecutor are mocked so the test does
not require network access or an API key. We verify that the agent initializes
with its 4 tools and that process_protocol returns structured rules + a trace.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.models.protocol import CriterionRuleCreate, CriterionType


def _fake_rule():
    return CriterionRuleCreate(
        criterion_text="HbA1c >= 7.5%", concept="HbA1c", operator=">=",
        value="7.5", required=True, criterion_type=CriterionType("inclusion"),
        confidence=0.95,
    )


@pytest.fixture
def agent():
    # Patch the heavy/external pieces before importing the agent module symbols.
    with patch("app.services.langchain_agent.ChatAnthropic"), \
         patch("app.services.langchain_agent.create_tool_calling_agent"), \
         patch("app.services.langchain_agent.AgentExecutor") as MockExec:
        MockExec.return_value.invoke.return_value = {"output": "agent reasoning trace"}
        from app.services.langchain_agent import ProtocolDocumentAgent
        a = ProtocolDocumentAgent()
    return a


def test_agent_has_four_tools(agent):
    assert len(agent._tools) == 4
    names = {t.name for t in agent._tools}
    assert "chunk_document" in names
    assert "locate_eligibility" in names
    assert "extract_clinical_entities" in names
    assert "lookup_snomed" in names


def test_process_protocol_returns_rules_and_trace(agent):
    # Stub NER + extractor so no real model / LLM is needed.
    agent.ner = MagicMock()
    agent.ner.extract_entities.return_value = [{"text": "diabetes"}]
    agent.extractor = MagicMock()
    agent.extractor.extract.return_value = [_fake_rule()]

    result = agent.process_protocol("Inclusion Criteria: HbA1c >= 7.5%", "NCT99999999")

    assert "rules" in result and len(result["rules"]) == 1
    assert result["rules"][0].concept == "HbA1c"
    assert result["agent_trace"] == "agent reasoning trace"
    assert result["entities"] == [{"text": "diabetes"}]
    # extractor was called WITH the NER entities as hints
    _, kwargs = agent.extractor.extract.call_args
    assert kwargs.get("ner_entities") == [{"text": "diabetes"}]


def test_process_protocol_survives_agent_failure(agent):
    agent._agent_executor.invoke.side_effect = RuntimeError("agent boom")
    agent.ner = MagicMock()
    agent.ner.extract_entities.return_value = []
    agent.extractor = MagicMock()
    agent.extractor.extract.return_value = [_fake_rule()]

    result = agent.process_protocol("some text", "NCT00000001")
    # Rules still produced even though the reasoning step failed
    assert len(result["rules"]) == 1
    assert "agent reasoning unavailable" in result["agent_trace"]
