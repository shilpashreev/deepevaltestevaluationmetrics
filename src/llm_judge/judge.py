"""LLM Judge Engine — core evaluation logic.

Evaluates test cases against criteria by prompting an LLM provider,
parsing structured JSON responses, and producing Judgment objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from difflib import SequenceMatcher
from typing import Any

from llm_judge.models import EvalCriterion, Judgment, TestCase
from llm_judge.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)```",
    re.DOTALL,
)
"""Regex to extract JSON from fenced markdown code blocks."""

_BARE_JSON_RE = re.compile(
    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    re.DOTALL,
)
"""Fallback regex to extract a top-level JSON object from raw text."""

_BASE_BACKOFF_SECONDS = 1.0
"""Initial backoff delay; doubles on each retry (1s → 2s → 4s …)."""


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Evaluates a *TestCase* against an *EvalCriterion* using an LLM.

    The judge builds a chat-completion prompt from the criterion templates,
    sends it to the configured provider, and parses the structured JSON
    response into a ``Judgment``.

    Args:
        provider: An LLM provider implementing ``BaseProvider``.
        max_retries: Maximum number of retry attempts on transient failures.
    """

    def __init__(self, provider: BaseProvider, max_retries: int = 3) -> None:
        self.provider = provider
        self.max_retries = max_retries

    # -- public API ---------------------------------------------------------

    async def evaluate(
        self,
        test_case: TestCase,
        criterion: EvalCriterion,
    ) -> Judgment:
        """Evaluate a single test case against a single criterion (async).

        Retries with exponential backoff on transient provider errors.

        Args:
            test_case: The test case containing prompt/response.
            criterion: The evaluation criterion to judge against.

        Returns:
            A ``Judgment`` with verdict, score, and reasoning.
        """
        messages = self._build_messages(test_case, criterion)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = await self.provider.complete(messages)
                return self._parse_judgment(raw_response, criterion)
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                last_error = exc
                if attempt < self.max_retries:
                    delay = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "Attempt %d/%d for criterion '%s' on test '%s' failed "
                        "(%s). Retrying in %.1fs …",
                        attempt,
                        self.max_retries,
                        criterion.name,
                        test_case.id,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted — return an error judgment.
        return Judgment(
            criterion=criterion.name,
            verdict=criterion.verdicts[-1] if criterion.verdicts else "error",
            score=0.0,
            reasoning=f"Evaluation failed after {self.max_retries} attempts: {last_error}",
            raw_response="",
        )

    def evaluate_sync(
        self,
        test_case: TestCase,
        criterion: EvalCriterion,
    ) -> Judgment:
        """Synchronous wrapper around :meth:`evaluate`.

        Uses ``asyncio.run`` when no event loop is running, otherwise
        falls back to the provider's synchronous completion method.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(self.evaluate(test_case, criterion))

        # Running inside an existing loop — use synchronous provider path.
        messages = self._build_messages(test_case, criterion)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self.provider.complete_sync(messages)
                return self._parse_judgment(raw_response, criterion)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    delay = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "Sync attempt %d/%d failed (%s). Retrying in %.1fs …",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return Judgment(
            criterion=criterion.name,
            verdict=criterion.verdicts[-1] if criterion.verdicts else "error",
            score=0.0,
            reasoning=f"Evaluation failed after {self.max_retries} attempts: {last_error}",
            raw_response="",
        )

    # -- prompt construction ------------------------------------------------

    def _build_messages(
        self,
        test_case: TestCase,
        criterion: EvalCriterion,
    ) -> list[dict[str, str]]:
        """Build the chat-completion message list for the judge prompt.

        The system message is the criterion's ``system_prompt``.  The user
        message is rendered from ``evaluation_prompt`` with placeholders
        ``{prompt}``, ``{response}``, and ``{reference}`` filled in.

        Few-shot examples (if any) are inserted as alternating user/assistant
        pairs between the system message and the final user message.

        Args:
            test_case: Source of prompt, response, and reference values.
            criterion: Contains prompt templates and few-shot examples.

        Returns:
            A list of ``{"role": …, "content": …}`` dicts.
        """
        reference_text = test_case.reference or "N/A"

        # System message
        system_content = criterion.system_prompt.strip()
        if not system_content:
            system_content = (
                "You are an expert evaluator. Assess the given response "
                "according to the criterion and return your judgment as JSON."
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Few-shot examples
        for example in criterion.few_shot_examples:
            if "user" in example:
                messages.append({"role": "user", "content": example["user"]})
            if "assistant" in example:
                messages.append({"role": "assistant", "content": example["assistant"]})

        # Evaluation prompt (user turn)
        user_content = criterion.evaluation_prompt.format(
            prompt=test_case.prompt,
            response=test_case.response,
            reference=reference_text,
        )

        # Append expected response format instruction
        verdict_list = ", ".join(f'"{v}"' for v in criterion.verdicts)
        user_content += (
            "\n\nRespond with a JSON object containing exactly these keys:\n"
            '- "reasoning": a brief explanation of your assessment\n'
            f'- "verdict": one of [{verdict_list}]\n'
            '- "score": a float between 0.0 and 1.0\n'
        )

        messages.append({"role": "user", "content": user_content})
        return messages

    # -- response parsing ---------------------------------------------------

    def _parse_judgment(
        self,
        raw_response: str,
        criterion: EvalCriterion,
    ) -> Judgment:
        """Parse an LLM response string into a ``Judgment``.

        Handles several common response formats:
        1. Pure JSON.
        2. JSON inside a fenced markdown code block.
        3. JSON embedded in surrounding prose.

        If parsing fails entirely, a ``Judgment`` is returned with the error
        recorded in the ``reasoning`` field and the worst verdict applied.

        Args:
            raw_response: The raw text returned by the LLM provider.
            criterion: Used for verdict validation and fallback scoring.

        Returns:
            A ``Judgment`` instance.
        """
        data = self._extract_json(raw_response)

        if data is None:
            logger.warning(
                "Failed to parse JSON from LLM response for criterion '%s'. "
                "Raw response: %.200s",
                criterion.name,
                raw_response,
            )
            return Judgment(
                criterion=criterion.name,
                verdict=criterion.verdicts[-1] if criterion.verdicts else "error",
                score=0.0,
                reasoning=f"[PARSE ERROR] Could not extract JSON from response: {raw_response[:500]}",
                raw_response=raw_response,
            )

        reasoning = str(data.get("reasoning", ""))
        raw_verdict = str(data.get("verdict", ""))
        raw_score = data.get("score")

        # Validate / coerce verdict
        verdict = self._resolve_verdict(raw_verdict, criterion)

        # Resolve score
        score = self._resolve_score(raw_score, verdict, criterion)

        return Judgment(
            criterion=criterion.name,
            verdict=verdict,
            score=score,
            reasoning=reasoning,
            raw_response=raw_response,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Try multiple strategies to extract a JSON object from *text*.

        Returns:
            Parsed dict on success, ``None`` on failure.
        """
        # Strategy 1: direct parse
        try:
            return json.loads(text.strip())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: extract from markdown code block
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())  # type: ignore[no-any-return]
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: find bare JSON object in surrounding text
        match = _BARE_JSON_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))  # type: ignore[no-any-return]
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    @staticmethod
    def _resolve_verdict(raw_verdict: str, criterion: EvalCriterion) -> str:
        """Map *raw_verdict* to one of the criterion's valid verdicts.

        If the raw verdict doesn't match exactly (case-insensitive), the
        closest match by string similarity is chosen.  If nothing is close
        enough, the **last** verdict (assumed to be the worst) is returned.
        """
        if not criterion.verdicts:
            return raw_verdict or "unknown"

        # Exact match (case-insensitive)
        lower_map = {v.lower(): v for v in criterion.verdicts}
        if raw_verdict.lower() in lower_map:
            return lower_map[raw_verdict.lower()]

        # Fuzzy match — pick best by SequenceMatcher ratio
        best_ratio = 0.0
        best_verdict = criterion.verdicts[-1]
        for valid in criterion.verdicts:
            ratio = SequenceMatcher(None, raw_verdict.lower(), valid.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_verdict = valid

        if best_ratio >= 0.6:
            logger.info(
                "Fuzzy-matched verdict '%s' → '%s' (%.0f%% similar)",
                raw_verdict,
                best_verdict,
                best_ratio * 100,
            )
            return best_verdict

        logger.warning(
            "Verdict '%s' not recognised for criterion '%s'; defaulting to '%s'",
            raw_verdict,
            criterion.name,
            criterion.verdicts[-1],
        )
        return criterion.verdicts[-1]

    @staticmethod
    def _resolve_score(
        raw_score: Any,
        verdict: str,
        criterion: EvalCriterion,
    ) -> float:
        """Coerce *raw_score* to a float in [0, 1].

        Falls back to the criterion's ``verdict_scores`` mapping when the
        raw value is missing or invalid.
        """
        if raw_score is not None:
            try:
                score = float(raw_score)
                return max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                pass

        # Fall back to verdict_scores
        return criterion.verdict_scores.get(verdict, 0.0)
