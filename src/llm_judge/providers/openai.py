"""OpenAI provider for LLM-as-Judge.

Uses the ``openai`` SDK (>=1.0) chat completions API.  Install with::

    pip install llm-judge[openai]
"""

from __future__ import annotations

import os

from llm_judge.providers.base import BaseProvider

try:
    import openai
except ImportError as exc:
    raise ImportError(
        "The 'openai' package is required for the OpenAI provider. "
        "Install it with:  pip install llm-judge[openai]"
    ) from exc


class OpenAIProvider(BaseProvider):
    """OpenAI chat-completion provider.

    Args:
        model: Model identifier (e.g. ``'gpt-4o-mini'``).
        api_key: API key. Falls back to the ``OPENAI_API_KEY`` env var.
        temperature: Sampling temperature.
        **kwargs: Extra keyword arguments forwarded to the OpenAI client.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, api_key=api_key, temperature=temperature)
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=resolved_key)
        self._async_client = openai.AsyncOpenAI(api_key=resolved_key)

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send an async chat-completion request.

        Args:
            messages: Chat messages with ``role`` and ``content`` keys.

        Returns:
            The assistant's response text.
        """
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def complete_sync(self, messages: list[dict[str, str]]) -> str:
        """Send a synchronous chat-completion request.

        Args:
            messages: Chat messages with ``role`` and ``content`` keys.

        Returns:
            The assistant's response text.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
