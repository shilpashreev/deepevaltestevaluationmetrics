"""Abstract base class for LLM providers.

All concrete providers (OpenAI, Anthropic, Google) inherit from
``BaseProvider`` and implement both the async and synchronous
completion methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract LLM provider interface.

    Args:
        model: Model identifier (e.g. ``'gpt-4o-mini'``).
        api_key: API key for authentication. When *None*, the provider
            should fall back to its default environment variable.
        temperature: Sampling temperature for the judge model.
        **kwargs: Provider-specific options forwarded to the underlying SDK.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Send a chat-completion request and return the response text.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.

        Returns:
            The assistant's response as a plain string.
        """
        ...

    @abstractmethod
    def complete_sync(self, messages: list[dict[str, str]]) -> str:
        """Synchronous version of :meth:`complete`.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.

        Returns:
            The assistant's response as a plain string.
        """
        ...
