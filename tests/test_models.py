"""Unit tests for llm_judge.models."""
from __future__ import annotations

import pytest

from llm_judge.models import (
    EvalCriterion,
    Judgment,
    SuiteResult,
    TestCase,
    TestResult,
)


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

class TestTestCase:
    """Tests for the TestCase dataclass."""

    def test_creation_minimal(self) -> None:
        tc = TestCase(id="test-1", prompt="Hello", response="Hi")
        assert tc.id == "test-1"
        assert tc.prompt == "Hello"
        assert tc.response == "Hi"
        assert tc.reference is None
        assert tc.tags == []
        assert tc.metadata == {}

    def test_creation_full(self) -> None:
        tc = TestCase(
            id="test-2",
            prompt="Q",
            response="A",
            reference="ref",
            tags=["t1"],
            metadata={"k": "v"},
        )
        assert tc.reference == "ref"
        assert tc.tags == ["t1"]
        assert tc.metadata == {"k": "v"}

    def test_default_tags_are_independent(self) -> None:
        """Verify each instance gets its own default list."""
        tc1 = TestCase(id="a", prompt="p", response="r")
        tc2 = TestCase(id="b", prompt="p", response="r")
        tc1.tags.append("added")
        assert tc2.tags == []

    def test_default_metadata_are_independent(self) -> None:
        tc1 = TestCase(id="a", prompt="p", response="r")
        tc2 = TestCase(id="b", prompt="p", response="r")
        tc1.metadata["key"] = "val"
        assert tc2.metadata == {}


# ---------------------------------------------------------------------------
# EvalCriterion
# ---------------------------------------------------------------------------

class TestEvalCriterion:
    """Tests for the EvalCriterion dataclass."""

    def test_creation(self) -> None:
        c = EvalCriterion(
            name="accuracy",
            description="Check accuracy",
            system_prompt="You are a judge",
            evaluation_prompt="Evaluate {prompt} -> {response}",
            verdicts=["correct", "partial", "incorrect"],
            verdict_scores={"correct": 1.0, "partial": 0.5, "incorrect": 0.0},
        )
        assert c.name == "accuracy"
        assert len(c.verdicts) == 3
        assert c.few_shot_examples == []

    def test_few_shot_examples_default_independent(self) -> None:
        c1 = EvalCriterion(
            name="a", description="d", system_prompt="s",
            evaluation_prompt="e", verdicts=["v"], verdict_scores={"v": 1.0},
        )
        c2 = EvalCriterion(
            name="b", description="d", system_prompt="s",
            evaluation_prompt="e", verdicts=["v"], verdict_scores={"v": 1.0},
        )
        c1.few_shot_examples.append({"example": "data"})
        assert c2.few_shot_examples == []


# ---------------------------------------------------------------------------
# Judgment
# ---------------------------------------------------------------------------

class TestJudgment:
    """Tests for the Judgment dataclass."""

    def test_creation(self) -> None:
        j = Judgment(
            criterion="accuracy",
            verdict="correct",
            score=1.0,
            reasoning="Good answer",
        )
        assert j.criterion == "accuracy"
        assert j.verdict == "correct"
        assert j.score == 1.0
        assert j.reasoning == "Good answer"
        assert j.raw_response == ""

    def test_with_raw_response(self) -> None:
        j = Judgment(
            criterion="safety",
            verdict="safe",
            score=1.0,
            reasoning="No issues",
            raw_response='{"verdict": "safe"}',
        )
        assert j.raw_response == '{"verdict": "safe"}'


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------

class TestTestResult:
    """Tests for TestResult and its compute_aggregate method."""

    @staticmethod
    def _make_case() -> TestCase:
        return TestCase(id="t", prompt="p", response="r")

    def test_compute_aggregate_pass(self) -> None:
        result = TestResult(test_case=self._make_case())
        result.judgments = [
            Judgment(criterion="accuracy", verdict="correct", score=1.0, reasoning="ok"),
            Judgment(criterion="relevance", verdict="relevant", score=1.0, reasoning="ok"),
        ]
        result.compute_aggregate(threshold=0.7)
        assert result.aggregate_score == 1.0
        assert result.passed is True

    def test_compute_aggregate_fail(self) -> None:
        result = TestResult(test_case=self._make_case())
        result.judgments = [
            Judgment(criterion="accuracy", verdict="incorrect", score=0.0, reasoning="wrong"),
            Judgment(criterion="relevance", verdict="relevant", score=1.0, reasoning="ok"),
        ]
        result.compute_aggregate(threshold=0.7)
        assert result.aggregate_score == pytest.approx(0.5)
        assert result.passed is False

    def test_compute_aggregate_boundary(self) -> None:
        """Score exactly at threshold should pass."""
        result = TestResult(test_case=self._make_case())
        result.judgments = [
            Judgment(criterion="accuracy", verdict="partial", score=0.7, reasoning="ok"),
        ]
        result.compute_aggregate(threshold=0.7)
        assert result.aggregate_score == pytest.approx(0.7)
        assert result.passed is True

    def test_compute_aggregate_no_judgments(self) -> None:
        result = TestResult(test_case=self._make_case())
        result.compute_aggregate(threshold=0.7)
        assert result.aggregate_score == 0.0
        assert result.passed is False

    def test_default_values(self) -> None:
        result = TestResult(test_case=self._make_case())
        assert result.judgments == []
        assert result.passed is False
        assert result.aggregate_score == 0.0
        assert result.error is None
        assert result.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# SuiteResult
# ---------------------------------------------------------------------------

class TestSuiteResult:
    """Tests for SuiteResult and its compute_summary method."""

    def test_compute_summary(self) -> None:
        tc1 = TestCase(id="t1", prompt="p", response="r")
        tc2 = TestCase(id="t2", prompt="p", response="r")
        r1 = TestResult(test_case=tc1, passed=True, aggregate_score=0.9)
        r2 = TestResult(test_case=tc2, passed=False, aggregate_score=0.3)
        suite = SuiteResult(suite_name="test", description="desc", results=[r1, r2])
        suite.compute_summary()

        assert suite.total_tests == 2
        assert suite.passed_tests == 1
        assert suite.failed_tests == 1
        assert suite.error_tests == 0
        assert suite.average_score == pytest.approx(0.6)
        assert suite.timestamp != ""

    def test_compute_summary_with_errors(self) -> None:
        tc = TestCase(id="t1", prompt="p", response="r")
        r1 = TestResult(test_case=tc, error="API failed")
        suite = SuiteResult(suite_name="test", description="desc", results=[r1])
        suite.compute_summary()

        assert suite.total_tests == 1
        assert suite.error_tests == 1
        assert suite.passed_tests == 0
        assert suite.failed_tests == 0

    def test_compute_summary_all_passed(self) -> None:
        results = [
            TestResult(
                test_case=TestCase(id=f"t{i}", prompt="p", response="r"),
                passed=True,
                aggregate_score=1.0,
            )
            for i in range(5)
        ]
        suite = SuiteResult(suite_name="perfect", description="all pass", results=results)
        suite.compute_summary()

        assert suite.total_tests == 5
        assert suite.passed_tests == 5
        assert suite.failed_tests == 0
        assert suite.average_score == pytest.approx(1.0)

    def test_compute_summary_empty(self) -> None:
        suite = SuiteResult(suite_name="empty", description="no tests")
        suite.compute_summary()

        assert suite.total_tests == 0
        assert suite.average_score == 0.0

    def test_timestamp_not_overwritten(self) -> None:
        """If timestamp is already set, compute_summary should preserve it."""
        suite = SuiteResult(
            suite_name="ts",
            description="d",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        suite.compute_summary()
        assert suite.timestamp == "2025-01-01T00:00:00+00:00"
