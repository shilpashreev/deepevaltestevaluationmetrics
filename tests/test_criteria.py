"""Unit tests for llm_judge.criteria."""
from __future__ import annotations

import pytest

from llm_judge.criteria import (
    CRITERIA_REGISTRY,
    get_criteria,
    get_criterion,
    list_criteria,
)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

class TestCriteriaRegistry:
    """Verify the global CRITERIA_REGISTRY contains all built-in criteria."""

    EXPECTED_CRITERIA = [
        "accuracy",
        "relevance",
        "coherence",
        "safety",
        "completeness",
        "conciseness",
        "instruction_following",
    ]

    def test_all_criteria_registered(self) -> None:
        for name in self.EXPECTED_CRITERIA:
            assert name in CRITERIA_REGISTRY, f"Missing criterion: {name}"

    def test_no_unexpected_criteria(self) -> None:
        """Registry should contain exactly the expected set."""
        assert set(CRITERIA_REGISTRY.keys()) == set(self.EXPECTED_CRITERIA)

    def test_criterion_structure(self) -> None:
        for name, criterion in CRITERIA_REGISTRY.items():
            assert criterion.name == name
            assert len(criterion.description) > 0, f"{name}: empty description"
            assert len(criterion.system_prompt) > 0, f"{name}: empty system_prompt"
            assert len(criterion.evaluation_prompt) > 0, f"{name}: empty evaluation_prompt"
            assert len(criterion.verdicts) == 3, f"{name}: expected 3 verdicts"
            assert len(criterion.verdict_scores) == 3, f"{name}: expected 3 verdict_scores"

            # Scores must be exactly {0.0, 0.5, 1.0}
            scores = sorted(criterion.verdict_scores.values())
            assert scores == [0.0, 0.5, 1.0], (
                f"{name}: verdict_scores should be [0.0, 0.5, 1.0], got {scores}"
            )

    def test_evaluation_prompt_has_placeholders(self) -> None:
        for name, criterion in CRITERIA_REGISTRY.items():
            assert "{prompt}" in criterion.evaluation_prompt, (
                f"{name}: missing {{prompt}} placeholder"
            )
            assert "{response}" in criterion.evaluation_prompt, (
                f"{name}: missing {{response}} placeholder"
            )

    def test_verdicts_match_verdict_scores_keys(self) -> None:
        """Every verdict label should have a corresponding score mapping."""
        for name, criterion in CRITERIA_REGISTRY.items():
            for verdict in criterion.verdicts:
                assert verdict in criterion.verdict_scores, (
                    f"{name}: verdict '{verdict}' not in verdict_scores"
                )


# ---------------------------------------------------------------------------
# get_criterion
# ---------------------------------------------------------------------------

class TestGetCriterion:
    """Tests for the get_criterion helper."""

    def test_valid(self) -> None:
        c = get_criterion("accuracy")
        assert c.name == "accuracy"

    def test_invalid_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            get_criterion("nonexistent")

    def test_returns_same_object(self) -> None:
        """Registry should return the same object each time."""
        c1 = get_criterion("safety")
        c2 = get_criterion("safety")
        assert c1 is c2


# ---------------------------------------------------------------------------
# get_criteria
# ---------------------------------------------------------------------------

class TestGetCriteria:
    """Tests for the get_criteria helper."""

    def test_multiple(self) -> None:
        criteria = get_criteria(["accuracy", "safety"])
        assert len(criteria) == 2
        assert criteria[0].name == "accuracy"
        assert criteria[1].name == "safety"

    def test_single(self) -> None:
        criteria = get_criteria(["coherence"])
        assert len(criteria) == 1

    def test_empty_list(self) -> None:
        criteria = get_criteria([])
        assert criteria == []

    def test_preserves_order(self) -> None:
        names = ["safety", "accuracy", "relevance"]
        criteria = get_criteria(names)
        assert [c.name for c in criteria] == names

    def test_invalid_raises(self) -> None:
        with pytest.raises(KeyError):
            get_criteria(["accuracy", "does_not_exist"])


# ---------------------------------------------------------------------------
# list_criteria
# ---------------------------------------------------------------------------

class TestListCriteria:
    """Tests for the list_criteria helper."""

    def test_returns_all(self) -> None:
        names = list_criteria()
        assert len(names) == 7

    def test_contains_accuracy(self) -> None:
        assert "accuracy" in list_criteria()

    def test_returns_list_of_strings(self) -> None:
        names = list_criteria()
        assert all(isinstance(n, str) for n in names)
