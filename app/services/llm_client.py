"""
Shared Anthropic Claude Sonnet client.
All LLM calls in the application route through this wrapper.
"""
import json
import anthropic
from loguru import logger
from app.config import settings


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.ANTHROPIC_MODEL
        self.max_tokens = settings.ANTHROPIC_MAX_TOKENS
        logger.info("ClaudeClient initialized with model={}", self.model)

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Single completion. Returns the text content of the first text block."""
        logger.debug("Claude request: system={} chars, user={} chars",
                     len(system), len(user))
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        logger.debug("Claude response: {} chars", len(text))
        return text

    def complete_json(self, system: str, user: str, max_tokens: int | None = None) -> list | dict:
        """
        Completion expecting JSON output. Strips markdown fences, parses.
        Retries ONCE with an explicit JSON reminder on parse failure.
        """
        raw = self.complete(system, user, max_tokens)
        parsed = self._try_parse_json(raw)
        if parsed is not None:
            return parsed

        logger.warning("First JSON parse failed — retrying with explicit reminder")
        retry_user = user + "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown, no preamble."
        raw2 = self.complete(system, retry_user, max_tokens)
        parsed2 = self._try_parse_json(raw2)
        if parsed2 is not None:
            return parsed2

        logger.error("JSON parse failed twice. Raw response: {}", raw2[:500])
        return []

    @staticmethod
    def _try_parse_json(text: str):
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # strip ```json ... ``` fences
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0] if "```" in cleaned else cleaned
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return None


_claude_client: ClaudeClient | None = None


def get_claude_client() -> ClaudeClient:
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client
