"""Anthropic provider for LLM-as-Judge.

Uses the ``anthropic`` SDK (>=0.30) messages API.  Install with::

    pip install llm-judge[anthropic]
"""

from __future__ import annotations

import os

from llm_judge.providers.base import BaseProvider

try:
    import anthropic
except ImportError as exc:
    raise ImportError(
        "The 'anthropic' package is required for the Anthropic provider. "
        "Install it with:  pip install llm-judge[anthropic]"
    ) from exc


class AnthropicProvider(BaseProvider):
    """Anthropic messages API provider.

    The Anthropic API treats the ``system`` message differently from user/
    assistant turns — it must be passed as a top-level ``system`` parameter
    rather than inside the ``messages`` list.

    Args:
        model: Model identifier (e.g. ``'claude-sonnet-4-20250514'``).
        api_key: API key. Falls back to the ``ANTHROPIC_API_KEY`` env var.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response (default 4096).
        **kwargs: Extra keyword arguments forwarded to the client.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, api_key=api_key, temperature=temperature)
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=resolved_key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_system(
        messages: list[dict[str, str]],
    ) -> tuple[str | anthropic.NotGiven, list[dict[str, str]]]:
        """Separate a system message from the rest of the conversation.

        Anthropic's API requires the system prompt as a dedicated parameter
        rather than an entry in the messages list.

        Returns:
            A 2-tuple of (system_text_or_NOT_GIVEN, remaining_messages).
        """
        system_text: str | anthropic.NotGiven = anthropic.NOT_GIVEN
        remaining: list[dict[str, str]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg["content"]
            else:
                remaining.append(msg)

        return system_text, remaining

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send an async message-creation request.

        Args:
            messages: Chat messages with ``role`` and ``content`` keys.

        Returns:
            The assistant's response text.
        """
        system, msgs = self._split_system(messages)
        response = await self._async_client.messages.create(
            model=self.model,
            system=system,
            messages=msgs,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self._max_tokens,
        )
        return response.content[0].text

    def complete_sync(self, messages: list[dict[str, str]]) -> str:
        """Send a synchronous message-creation request.

        Args:
            messages: Chat messages with ``role`` and ``content`` keys.

        Returns:
            The assistant's response text.
        """
        system, msgs = self._split_system(messages)
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=msgs,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self._max_tokens,
        )
        return response.content[0].text
