"""Provider registry and factory for LLM backends.

Use :func:`get_provider` to instantiate a provider by name::

    provider = get_provider("openai", model="gpt-4o-mini")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_judge.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Lazy registry: maps provider name → (module_path, class_name)
# Providers are imported on demand so missing optional deps don't break
# the entire package at import time.
# ---------------------------------------------------------------------------
_PROVIDER_SPECS: dict[str, tuple[str, str]] = {
    "openai": ("llm_judge.providers.openai", "OpenAIProvider"),
    "anthropic": ("llm_judge.providers.anthropic", "AnthropicProvider"),
    "google": ("llm_judge.providers.google", "GoogleProvider"),
}

# Populated on first access or when a provider is explicitly loaded.
PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}


def _load_provider_class(name: str) -> type[BaseProvider]:
    """Lazily import and cache a provider class by name.

    Raises:
        ImportError: With a helpful message if the provider's optional
            dependency is not installed.
        KeyError: If *name* is not a known provider.
    """
    if name in PROVIDER_REGISTRY:
        return PROVIDER_REGISTRY[name]

    if name not in _PROVIDER_SPECS:
        available = ", ".join(sorted(_PROVIDER_SPECS))
        raise KeyError(
            f"Unknown provider '{name}'. Available providers: {available}"
        )

    module_path, class_name = _PROVIDER_SPECS[name]

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not load provider '{name}'. "
            f"Install the optional dependency with:  pip install llm-judge[{name}]"
        ) from exc

    cls: type[BaseProvider] = getattr(module, class_name)
    PROVIDER_REGISTRY[name] = cls
    return cls


def get_provider(
    name: str,
    model: str,
    api_key: str | None = None,
    **kwargs: object,
) -> BaseProvider:
    """Instantiate an LLM provider by name.

    Args:
        name: Provider identifier (``'openai'``, ``'anthropic'``, ``'google'``).
        model: Model identifier to use for completions.
        api_key: Optional API key; when *None*, the provider's own env-var
            fallback is used.
        **kwargs: Extra keyword arguments forwarded to the provider constructor.

    Returns:
        A ready-to-use ``BaseProvider`` instance.

    Raises:
        KeyError: If the provider name is not recognised.
        ImportError: If the provider's SDK package is not installed.
    """
    cls = _load_provider_class(name)
    return cls(model=model, api_key=api_key, **kwargs)
