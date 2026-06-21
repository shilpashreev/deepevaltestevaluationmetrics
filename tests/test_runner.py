"""Unit tests for llm_judge.runner."""
from __future__ import annotations

import pytest
import yaml

from llm_judge.runner import TestRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_suite(tmp_path, data: dict) -> str:
    """Write *data* as YAML to a temp file and return the path."""
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(yaml.dump(data, default_flow_style=False))
    return str(suite_file)


VALID_SUITE_DATA = {
    "suite": {
        "name": "Test Suite",
        "description": "A test",
        "criteria": ["accuracy"],
        "pass_threshold": 0.7,
    },
    "tests": [
        {
            "id": "test-1",
            "prompt": "Hello",
            "response": "Hi there",
        }
    ],
}


# ---------------------------------------------------------------------------
# load_suite
# ---------------------------------------------------------------------------

class TestLoadSuite:
    """Tests for TestRunner.load_suite."""

    def test_load_valid_suite(self, tmp_path) -> None:
        path = _write_suite(tmp_path, VALID_SUITE_DATA)
        meta, cases = TestRunner.load_suite(path)

        assert meta["name"] == "Test Suite"
        assert meta["description"] == "A test"
        assert meta["criteria"] == ["accuracy"]
        assert meta["pass_threshold"] == 0.7
        assert len(cases) == 1
        assert cases[0].id == "test-1"
        assert cases[0].prompt == "Hello"
        assert cases[0].response == "Hi there"

    def test_load_suite_with_reference_and_tags(self, tmp_path) -> None:
        data = {
            "suite": {
                "name": "Test",
                "description": "Test",
                "criteria": ["accuracy"],
                "pass_threshold": 0.7,
            },
            "tests": [
                {
                    "id": "test-1",
                    "prompt": "Q",
                    "response": "A",
                    "reference": "Ref",
                    "tags": ["tag1", "tag2"],
                }
            ],
        }
        path = _write_suite(tmp_path, data)
        meta, cases = TestRunner.load_suite(path)

        assert cases[0].reference == "Ref"
        assert cases[0].tags == ["tag1", "tag2"]

    def test_load_suite_multiple_tests(self, tmp_path) -> None:
        data = {
            "suite": {
                "name": "Multi",
                "description": "Multiple tests",
                "criteria": ["accuracy"],
                "pass_threshold": 0.5,
            },
            "tests": [
                {"id": f"t{i}", "prompt": f"q{i}", "response": f"a{i}"}
                for i in range(5)
            ],
        }
        path = _write_suite(tmp_path, data)
        _, cases = TestRunner.load_suite(path)
        assert len(cases) == 5


# ---------------------------------------------------------------------------
# validate_suite
# ---------------------------------------------------------------------------

class TestValidateSuite:
    """Tests for TestRunner.validate_suite."""

    def test_valid_suite(self, tmp_path) -> None:
        path = _write_suite(tmp_path, VALID_SUITE_DATA)
        errors = TestRunner.validate_suite(path)
        assert errors == []

    def test_missing_suite_key(self, tmp_path) -> None:
        path = _write_suite(tmp_path, {"tests": []})
        errors = TestRunner.validate_suite(path)
        assert len(errors) > 0

    def test_missing_tests_key(self, tmp_path) -> None:
        path = _write_suite(tmp_path, {
            "suite": {
                "name": "T", "description": "D",
                "criteria": ["accuracy"], "pass_threshold": 0.7,
            }
        })
        errors = TestRunner.validate_suite(path)
        assert len(errors) > 0

    def test_missing_test_fields(self, tmp_path) -> None:
        data = {
            "suite": {
                "name": "Test",
                "description": "Test",
                "criteria": ["accuracy"],
                "pass_threshold": 0.7,
            },
            "tests": [
                {"id": "t1"},  # missing prompt and response
            ],
        }
        path = _write_suite(tmp_path, data)
        errors = TestRunner.validate_suite(path)
        assert len(errors) > 0

    def test_missing_suite_name(self, tmp_path) -> None:
        data = {
            "suite": {
                "description": "D",
                "criteria": ["accuracy"],
                "pass_threshold": 0.7,
            },
            "tests": [
                {"id": "t1", "prompt": "Q", "response": "A"},
            ],
        }
        path = _write_suite(tmp_path, data)
        errors = TestRunner.validate_suite(path)
        assert len(errors) > 0

    def test_missing_criteria(self, tmp_path) -> None:
        data = {
            "suite": {
                "name": "T",
                "description": "D",
                "pass_threshold": 0.7,
            },
            "tests": [
                {"id": "t1", "prompt": "Q", "response": "A"},
            ],
        }
        path = _write_suite(tmp_path, data)
        errors = TestRunner.validate_suite(path)
        assert len(errors) > 0

    def test_duplicate_test_ids(self, tmp_path) -> None:
        data = {
            "suite": {
                "name": "T",
                "description": "D",
                "criteria": ["accuracy"],
                "pass_threshold": 0.7,
            },
            "tests": [
                {"id": "dup", "prompt": "Q1", "response": "A1"},
                {"id": "dup", "prompt": "Q2", "response": "A2"},
            ],
        }
        path = _write_suite(tmp_path, data)
        errors = TestRunner.validate_suite(path)
        assert any("duplicate" in e.lower() or "dup" in e.lower() for e in errors)
