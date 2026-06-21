"""Google Gemini provider for LLM-as-Judge.

Uses the ``google-genai`` SDK (>=1.0).  Install with::

    pip install llm-judge[google]
"""

from __future__ import annotations

import os

from llm_judge.providers.base import BaseProvider

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "The 'google-genai' package is required for the Google provider. "
        "Install it with:  pip install llm-judge[google]"
    ) from exc


class GoogleProvider(BaseProvider):
    """Google Gemini provider via the ``google-genai`` SDK.

    Args:
        model: Model identifier (e.g. ``'gemini-2.0-flash'``).
        api_key: API key. Falls back to the ``GOOGLE_API_KEY`` env var.
        temperature: Sampling temperature.
        **kwargs: Extra keyword arguments forwarded to the client.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(model=model, api_key=api_key, temperature=temperature)
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=resolved_key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[types.Content]]:
        """Convert OpenAI-style messages to Gemini Content objects.

        The Gemini API uses a ``system_instruction`` parameter for system
        messages and a list of ``Content`` objects for the conversation.
        Roles are mapped as: ``user`` → ``user``, ``assistant`` → ``model``.

        Returns:
            A 2-tuple of (system_instruction_or_None, contents).
        """
        system_instruction: str | None = None
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")

            if role == "system":
                system_instruction = text
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=text)],
                    )
                )

        return system_instruction, contents

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send an async generation request.

        Args:
            messages: Chat messages with ``role`` and ``content`` keys.

        Returns:
            The model's response text.
        """
        system_instruction, contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
        )

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""

    def complete_sync(self, messages: list[dict[str, str]]) -> str:
        """Send a synchronous generation request.

        Args:
            messages: Chat messages with ``role`` and ``content`` keys.

        Returns:
            The model's response text.
        """
        system_instruction, contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""
