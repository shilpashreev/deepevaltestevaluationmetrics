"""LLM-as-Judge test framework for evaluating language model outputs.

Quick start::

    from llm_judge import TestCase, EvalCriterion, Judgment, TestResult, SuiteResult
"""

from __future__ import annotations

from llm_judge.models import (
    EvalCriterion,
    Judgment,
    SuiteResult,
    TestCase,
    TestResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "EvalCriterion",
    "Judgment",
    "SuiteResult",
    "TestCase",
    "TestResult",
]
