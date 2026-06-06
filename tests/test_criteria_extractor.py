import json
import pytest
from unittest.mock import patch, MagicMock
from app.services.criteria_extractor import CriteriaExtractor
from app.models.protocol import CriterionType


@pytest.fixture
def extractor():
    return CriteriaExtractor()


VALID_GPT_RESPONSE = json.dumps([
    {
        "criterion_text": "Participant must be between 18 and 65 years of age",
        "concept": "age",
        "operator": "between",
        "value": "18-65 years",
        "required": True,
        "criterion_type": "inclusion",
        "confidence": 0.99,
    },
    {
        "criterion_text": "HbA1c >= 7.5% at screening",
        "concept": "HbA1c",
        "operator": ">=",
        "value": "7.5%",
        "required": True,
        "criterion_type": "inclusion",
        "confidence": 0.97,
    },
    {
        "criterion_text": "No prior insulin therapy",
        "concept": "insulin",
        "operator": "absence",
        "value": "insulin",
        "required": False,
        "criterion_type": "exclusion",
        "confidence": 0.95,
    },
])


def _mock_openai_response(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestCriteriaExtractor:
    def test_valid_response_parsed_correctly(self, extractor):
        with patch.object(extractor.client.chat.completions, "create", return_value=_mock_openai_response(VALID_GPT_RESPONSE)):
            rules = extractor.extract("Some criteria text", "NCT12345678")

        assert len(rules) == 3
        assert rules[0].concept == "age"
        assert rules[0].operator == "between"
        assert rules[0].criterion_type == CriterionType.inclusion
        assert rules[0].required is True

        assert rules[1].concept == "HbA1c"
        assert rules[1].operator == ">="
        assert abs(rules[1].confidence - 0.97) < 0.01

        assert rules[2].criterion_type == CriterionType.exclusion
        assert rules[2].required is False

    def test_malformed_json_returns_empty_list(self, extractor):
        malformed = "This is not JSON at all {broken"
        with patch.object(extractor.client.chat.completions, "create", return_value=_mock_openai_response(malformed)):
            with patch.object(extractor.client.chat.completions, "create", side_effect=[
                _mock_openai_response(malformed),
                _mock_openai_response(malformed),
            ]):
                rules = extractor.extract("Some criteria", "NCT00000000")
        assert rules == []

    def test_empty_criteria_text_returns_empty_list(self, extractor):
        rules = extractor.extract("", "NCT00000000")
        assert rules == []

    def test_whitespace_only_text_returns_empty_list(self, extractor):
        rules = extractor.extract("   \n\t  ", "NCT00000000")
        assert rules == []

    def test_markdown_wrapped_json_parsed(self, extractor):
        wrapped = f"```json\n{VALID_GPT_RESPONSE}\n```"
        with patch.object(extractor.client.chat.completions, "create", return_value=_mock_openai_response(wrapped)):
            rules = extractor.extract("Some criteria", "NCT12345678")
        assert len(rules) == 3

    def test_api_error_returns_empty_list(self, extractor):
        from openai import APIError
        with patch.object(extractor.client.chat.completions, "create", side_effect=Exception("Connection error")):
            rules = extractor.extract("Some criteria", "NCT00000000")
        assert rules == []

    def test_confidence_values_in_range(self, extractor):
        with patch.object(extractor.client.chat.completions, "create", return_value=_mock_openai_response(VALID_GPT_RESPONSE)):
            rules = extractor.extract("Some criteria", "NCT12345678")
        for rule in rules:
            assert 0.0 <= rule.confidence <= 1.0
