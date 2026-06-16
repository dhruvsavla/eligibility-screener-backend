"""Tests for the shared Claude (Anthropic) client wrapper."""
from unittest.mock import MagicMock, patch
from app.services.llm_client import ClaudeClient


def _client_with_responses(*texts):
    """Build a ClaudeClient whose underlying anthropic client yields `texts` in order."""
    with patch("app.services.llm_client.anthropic.Anthropic") as MockAnthropic:
        instance = MockAnthropic.return_value
        responses = []
        for t in texts:
            block = MagicMock()
            block.type = "text"
            block.text = t
            resp = MagicMock()
            resp.content = [block]
            responses.append(resp)
        instance.messages.create.side_effect = responses
        client = ClaudeClient()
    return client


def test_try_parse_plain_json():
    assert ClaudeClient._try_parse_json('[{"a": 1}]') == [{"a": 1}]


def test_try_parse_strips_markdown_fence():
    fenced = "```json\n[{\"a\": 1}]\n```"
    assert ClaudeClient._try_parse_json(fenced) == [{"a": 1}]


def test_try_parse_bad_json_returns_none():
    assert ClaudeClient._try_parse_json("not json {") is None


def test_complete_returns_text():
    client = _client_with_responses("hello world")
    assert client.complete("sys", "user") == "hello world"


def test_complete_json_first_try():
    client = _client_with_responses('[{"x": 1}]')
    assert client.complete_json("sys", "user") == [{"x": 1}]


def test_complete_json_retries_then_succeeds():
    # First response is unparseable, second is valid → retry path
    client = _client_with_responses("garbage {", '[{"y": 2}]')
    result = client.complete_json("sys", "user")
    assert result == [{"y": 2}]


def test_complete_json_returns_empty_after_two_failures():
    client = _client_with_responses("garbage {", "still bad {")
    assert client.complete_json("sys", "user") == []
