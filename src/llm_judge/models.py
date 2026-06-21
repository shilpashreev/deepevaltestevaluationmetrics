"""Core data models for the LLM-as-Judge framework.

Defines the primary dataclasses used throughout the evaluation pipeline:
test cases, evaluation criteria, judgments, and result aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TestCase:
    """A single test case to evaluate.

    Attributes:
        id: Unique identifier for the test case.
        prompt: The original prompt sent to the LLM.
        response: The LLM's response to evaluate.
        reference: Optional reference/gold-standard answer for comparison.
        tags: Categorical tags for filtering and grouping.
        metadata: Arbitrary key-value metadata attached to this case.
    """

    id: str
    prompt: str
    response: str
    reference: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCriterion:
    """An evaluation criterion that defines how the LLM judge scores responses.

    Attributes:
        name: Short identifier for the criterion (e.g. 'accuracy').
        description: Human-readable description of what this criterion measures.
        system_prompt: System message that sets the judge's role and behavior.
        evaluation_prompt: Jinja2/format-string template with {prompt}, {response},
            and {reference} placeholders.
        verdicts: Ordered list of possible verdict labels (best to worst).
        verdict_scores: Mapping from verdict label to numeric score (0.0–1.0).
        few_shot_examples: Optional list of example evaluations for in-context
            learning. Each dict should have 'prompt', 'response', 'reasoning',
            and 'verdict' keys.
    """

    name: str
    description: str
    system_prompt: str
    evaluation_prompt: str
    verdicts: list[str]
    verdict_scores: dict[str, float]
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Judgment:
    """A single judgment produced by the LLM judge for one criterion.

    Attributes:
        criterion: Name of the criterion that was evaluated.
        verdict: The categorical verdict label.
        score: Numeric score between 0.0 and 1.0.
        reasoning: The judge's chain-of-thought reasoning.
        raw_response: The unprocessed response string from the LLM judge.
    """

    criterion: str
    verdict: str
    score: float
    reasoning: str
    raw_response: str = ""


@dataclass
class TestResult:
    """Result of evaluating a single test case across one or more criteria.

    Attributes:
        test_case: The test case that was evaluated.
        judgments: List of judgments, one per criterion.
        passed: Whether the test case passed (aggregate_score >= threshold).
        aggregate_score: Mean score across all judgments.
        error: Error message if evaluation failed.
        duration_seconds: Wall-clock time for the evaluation.
    """

    test_case: TestCase
    judgments: list[Judgment] = field(default_factory=list)
    passed: bool = False
    aggregate_score: float = 0.0
    error: str | None = None
    duration_seconds: float = 0.0

    def compute_aggregate(self, threshold: float = 0.7) -> None:
        """Compute aggregate score from judgments and determine pass/fail.

        Sets ``aggregate_score`` to the mean of all judgment scores and sets
        ``passed`` to ``True`` if the aggregate meets or exceeds *threshold*.

        Args:
            threshold: Minimum aggregate score required to pass (0.0–1.0).
        """
        if not self.judgments:
            self.aggregate_score = 0.0
            self.passed = False
            return

        self.aggregate_score = sum(j.score for j in self.judgments) / len(
            self.judgments
        )
        self.passed = self.aggregate_score >= threshold


@dataclass
class SuiteResult:
    """Aggregated results for an entire test suite.

    Attributes:
        suite_name: Name of the test suite.
        description: Human-readable description of the suite.
        results: Individual test results.
        total_tests: Total number of test cases evaluated.
        passed_tests: Number of test cases that passed.
        failed_tests: Number of test cases that failed (did not pass).
        error_tests: Number of test cases that encountered errors.
        average_score: Mean aggregate score across all test results.
        duration_seconds: Total wall-clock time for the suite run.
        timestamp: ISO-8601 timestamp of when the suite completed.
        config: Snapshot of the configuration used for this run.
    """

    suite_name: str
    description: str
    results: list[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    average_score: float = 0.0
    duration_seconds: float = 0.0
    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> None:
        """Recompute all summary statistics from the current results list.

        Iterates over ``results`` to populate *total_tests*, *passed_tests*,
        *failed_tests*, *error_tests*, *average_score*, and *timestamp*.
        """
        self.total_tests = len(self.results)
        self.passed_tests = sum(1 for r in self.results if r.passed)
        self.error_tests = sum(1 for r in self.results if r.error is not None)
        self.failed_tests = self.total_tests - self.passed_tests - self.error_tests
        self.duration_seconds = sum(r.duration_seconds for r in self.results)

        scored = [r.aggregate_score for r in self.results if r.error is None]
        self.average_score = sum(scored) / len(scored) if scored else 0.0

        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
