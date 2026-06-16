import pytest
from unittest.mock import patch, MagicMock
from app.services.criteria_extractor import CriteriaExtractor
from app.models.protocol import CriterionType


@pytest.fixture
def extractor():
    return CriteriaExtractor()


VALID_PARSED = [
    {
        "criterion_text": "Participant must be between 18 and 65 years of age",
        "concept": "age", "operator": "between", "value": "18-65 years",
        "required": True, "criterion_type": "inclusion", "confidence": 0.99,
    },
    {
        "criterion_text": "HbA1c >= 7.5% at screening",
        "concept": "HbA1c", "operator": ">=", "value": "7.5%",
        "required": True, "criterion_type": "inclusion", "confidence": 0.97,
    },
    {
        "criterion_text": "No prior insulin therapy",
        "concept": "insulin", "operator": "absence", "value": "insulin",
        "required": False, "criterion_type": "exclusion", "confidence": 0.95,
    },
]


def _mock_claude(parsed):
    """Patch get_claude_client so complete_json returns `parsed`."""
    client = MagicMock()
    client.complete_json.return_value = parsed
    return patch("app.services.criteria_extractor.get_claude_client", return_value=client)


class TestCriteriaExtractor:
    def test_valid_response_parsed_correctly(self, extractor):
        with _mock_claude(VALID_PARSED):
            rules = extractor.extract("Some criteria text", "NCT12345678")

        assert len(rules) == 3
        assert rules[0].concept == "age"
        assert rules[0].operator == "between"
        assert rules[0].criterion_type == CriterionType.inclusion
        assert rules[0].required is True
        assert rules[1].concept == "HbA1c"
        assert abs(rules[1].confidence - 0.97) < 0.01
        assert rules[2].criterion_type == CriterionType.exclusion
        assert rules[2].required is False

    def test_malformed_json_returns_empty_list(self, extractor):
        # complete_json returns [] on unparseable output (handled inside ClaudeClient)
        with _mock_claude([]):
            rules = extractor.extract("Some criteria", "NCT00000000")
        assert rules == []

    def test_empty_criteria_text_returns_empty_list(self, extractor):
        rules = extractor.extract("", "NCT00000000")
        assert rules == []

    def test_whitespace_only_text_returns_empty_list(self, extractor):
        rules = extractor.extract("   \n\t  ", "NCT00000000")
        assert rules == []

    def test_dict_wrapped_in_criteria_key(self, extractor):
        with _mock_claude({"criteria": VALID_PARSED}):
            rules = extractor.extract("Some criteria", "NCT12345678")
        assert len(rules) == 3

    def test_llm_error_returns_empty_list(self, extractor):
        client = MagicMock()
        client.complete_json.side_effect = Exception("Connection error")
        with patch("app.services.criteria_extractor.get_claude_client", return_value=client):
            rules = extractor.extract("Some criteria", "NCT00000000")
        assert rules == []

    def test_ner_entities_passed_as_hints(self, extractor):
        client = MagicMock()
        client.complete_json.return_value = VALID_PARSED
        entities = [{"text": "diabetes"}, {"text": "insulin"}]
        with patch("app.services.criteria_extractor.get_claude_client", return_value=client):
            extractor.extract("Some criteria", "NCT12345678", ner_entities=entities)
        # The user prompt (2nd positional arg) must contain the entity hint block
        _, user_prompt = client.complete_json.call_args[0][:2]
        assert "scispaCy detected" in user_prompt
        assert "diabetes" in user_prompt

    def test_confidence_values_in_range(self, extractor):
        with _mock_claude(VALID_PARSED):
            rules = extractor.extract("Some criteria", "NCT12345678")
        for rule in rules:
            assert 0.0 <= rule.confidence <= 1.0
