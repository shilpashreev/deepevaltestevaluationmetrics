"""Test Runner — orchestrates the full evaluation pipeline.

Loads YAML test suites, evaluates every test case against configured
criteria using concurrent LLM calls, and aggregates results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm_judge.criteria import get_criteria
from llm_judge.config import Config
from llm_judge.judge import LLMJudge
from llm_judge.models import EvalCriterion, SuiteResult, TestCase, TestResult
from llm_judge.providers import get_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suite file schema constants
# ---------------------------------------------------------------------------

_REQUIRED_SUITE_KEYS = {"suite", "tests"}
_REQUIRED_TEST_KEYS = {"id", "prompt", "response"}

# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Runs evaluation suites by coordinating judge, provider, and reporter.

    Args:
        config: Framework configuration (provider, model, criteria, etc.).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.provider = get_provider(
            config.provider,
            config.model,
            config.api_key,
            temperature=config.temperature,
        )
        self.judge = LLMJudge(self.provider, max_retries=config.max_retries)
        self.criteria: list[EvalCriterion] = get_criteria(config.criteria)

    # -- public API ---------------------------------------------------------

    async def run_suite(self, suite_path: str) -> SuiteResult:
        """Run all tests defined in a YAML suite file (or directory).

        Each test is evaluated against every configured criterion. Calls are
        executed concurrently up to ``config.concurrency`` in-flight tasks.
        Progress is rendered via *rich*.

        Args:
            suite_path: Path to a ``.yaml`` file or a directory of them.

        Returns:
            A ``SuiteResult`` containing per-test results and summary stats.
        """
        path = Path(suite_path)
        if path.is_dir():
            yaml_files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
            if not yaml_files:
                raise FileNotFoundError(
                    f"No YAML files found in directory: {suite_path}"
                )
            # Merge all suites in the directory into a single run.
            all_test_cases: list[TestCase] = []
            suite_meta: dict[str, Any] = {
                "name": path.name,
                "description": f"Combined suites from {suite_path}",
            }
            for yf in yaml_files:
                meta, cases = self.load_suite(str(yf))
                all_test_cases.extend(cases)
                # Use first file's name/desc if available
                if suite_meta["name"] == path.name and "name" in meta:
                    suite_meta["name"] = meta["name"]
                    suite_meta["description"] = meta.get("description", "")
        else:
            suite_meta, all_test_cases = self.load_suite(suite_path)

        suite_name = suite_meta.get("name", "unnamed")
        description = suite_meta.get("description", "")

        semaphore = asyncio.Semaphore(self.config.concurrency)
        results: list[TestResult] = []
        suite_start = time.perf_counter()

        # Attempt to show rich progress; fall back silently if unavailable.
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
        except ImportError:  # pragma: no cover
            progress = None  # type: ignore[assignment]

        async def _run_one(tc: TestCase, task_id: Any | None) -> TestResult:
            async with semaphore:
                result = await self.run_test(tc)
                if progress is not None and task_id is not None:
                    progress.advance(task_id)
                return result

        if progress is not None:
            with progress:
                task_id = progress.add_task(
                    f"Evaluating {suite_name}",
                    total=len(all_test_cases),
                )
                coros = [_run_one(tc, task_id) for tc in all_test_cases]
                results = await asyncio.gather(*coros)
        else:
            coros = [_run_one(tc, None) for tc in all_test_cases]
            results = await asyncio.gather(*coros)

        suite_duration = time.perf_counter() - suite_start

        suite_result = SuiteResult(
            suite_name=suite_name,
            description=description,
            results=list(results),
            duration_seconds=round(suite_duration, 3),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config={
                "provider": self.config.provider,
                "model": self.config.model,
                "criteria": self.config.criteria,
                "pass_threshold": self.config.pass_threshold,
                "concurrency": self.config.concurrency,
                "temperature": self.config.temperature,
            },
        )
        suite_result.compute_summary()
        return suite_result

    async def run_test(self, test_case: TestCase) -> TestResult:
        """Evaluate a single test case against all configured criteria.

        Errors on individual criteria do **not** propagate — they are
        recorded inside the ``TestResult`` so the suite can continue.

        Args:
            test_case: The test case to evaluate.

        Returns:
            A ``TestResult`` with judgments and aggregate score.
        """
        test_start = time.perf_counter()
        result = TestResult(test_case=test_case)

        try:
            tasks = [
                self.judge.evaluate(test_case, criterion)
                for criterion in self.criteria
            ]
            judgments = await asyncio.gather(*tasks, return_exceptions=True)

            for j in judgments:
                if isinstance(j, Exception):
                    logger.error(
                        "Criterion evaluation failed for test '%s': %s",
                        test_case.id,
                        j,
                    )
                    result.error = str(j)
                else:
                    result.judgments.append(j)

            result.compute_aggregate(self.config.pass_threshold)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected error evaluating test '%s': %s",
                test_case.id,
                exc,
            )
            result.error = str(exc)

        result.duration_seconds = round(time.perf_counter() - test_start, 3)
        return result

    def run_suite_sync(self, suite_path: str) -> SuiteResult:
        """Synchronous convenience wrapper around :meth:`run_suite`.

        Creates a new event loop via ``asyncio.run()``.
        """
        return asyncio.run(self.run_suite(suite_path))

    # -- suite loading / validation -----------------------------------------

    @staticmethod
    def load_suite(suite_path: str) -> tuple[dict[str, Any], list[TestCase]]:
        """Parse a YAML suite file into metadata and ``TestCase`` objects.

        Expected YAML structure::

            name: My Suite
            description: Optional description
            tests:
              - id: test_1
                prompt: "…"
                response: "…"
                reference: "…"       # optional
                tags: [tag1, tag2]   # optional
                metadata: {}         # optional

        Args:
            suite_path: Filesystem path to the YAML file.

        Returns:
            A 2-tuple of (metadata dict, list of TestCase).

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the YAML structure is invalid.
        """
        path = Path(suite_path)
        if not path.exists():
            raise FileNotFoundError(f"Suite file not found: {suite_path}")

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(
                f"Suite file must contain a YAML mapping, got {type(data).__name__}"
            )

        missing = _REQUIRED_SUITE_KEYS - data.keys()
        if missing:
            raise ValueError(
                f"Suite file missing required keys: {', '.join(sorted(missing))}"
            )

        tests_raw = data.get("tests", [])
        if not isinstance(tests_raw, list):
            raise ValueError("'tests' must be a list")

        test_cases: list[TestCase] = []
        for idx, entry in enumerate(tests_raw):
            if not isinstance(entry, dict):
                raise ValueError(f"Test entry at index {idx} is not a mapping")

            entry_missing = _REQUIRED_TEST_KEYS - entry.keys()
            if entry_missing:
                raise ValueError(
                    f"Test at index {idx} missing keys: "
                    f"{', '.join(sorted(entry_missing))}"
                )

            test_cases.append(
                TestCase(
                    id=str(entry["id"]),
                    prompt=str(entry["prompt"]),
                    response=str(entry["response"]),
                    reference=entry.get("reference"),
                    tags=entry.get("tags", []),
                    metadata=entry.get("metadata", {}),
                )
            )

        meta = data.get("suite", {})
        if "name" not in meta:
            raise ValueError("Suite file missing required keys: name")
        return meta, test_cases

    @staticmethod
    def validate_suite(suite_path: str) -> list[str]:
        """Validate a suite file without running it.

        Args:
            suite_path: Path to the YAML suite file.

        Returns:
            A list of error strings (empty means the file is valid).
        """
        errors: list[str] = []
        path = Path(suite_path)

        if not path.exists():
            return [f"File not found: {suite_path}"]

        if path.suffix.lower() not in {".yaml", ".yml"}:
            errors.append(f"Unexpected file extension: {path.suffix}")

        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            return [f"YAML parse error: {exc}"]

        if not isinstance(data, dict):
            return [f"Root element must be a mapping, got {type(data).__name__}"]

        missing = _REQUIRED_SUITE_KEYS - data.keys()
        if missing:
            errors.append(f"Missing required keys: {', '.join(sorted(missing))}")
            return errors
        else:
            suite_meta = data.get("suite", {})
            if not isinstance(suite_meta, dict):
                errors.append("'suite' must be a mapping")
            else:
                if "name" not in suite_meta:
                    errors.append("'suite' missing required key: name")
                if "criteria" not in suite_meta:
                    errors.append("'suite' missing required key: criteria")

        tests_raw = data.get("tests")
        if tests_raw is None:
            return errors  # already reported above
        if not isinstance(tests_raw, list):
            errors.append("'tests' must be a list")
            return errors

        seen_ids: set[str] = set()
        for idx, entry in enumerate(tests_raw):
            prefix = f"tests[{idx}]"
            if not isinstance(entry, dict):
                errors.append(f"{prefix}: must be a mapping")
                continue

            entry_missing = _REQUIRED_TEST_KEYS - entry.keys()
            if entry_missing:
                errors.append(
                    f"{prefix}: missing keys {', '.join(sorted(entry_missing))}"
                )

            test_id = entry.get("id")
            if test_id is not None:
                test_id_str = str(test_id)
                if test_id_str in seen_ids:
                    errors.append(f"{prefix}: duplicate id '{test_id_str}'")
                seen_ids.add(test_id_str)

            tags = entry.get("tags")
            if tags is not None and not isinstance(tags, list):
                errors.append(f"{prefix}: 'tags' must be a list")

            metadata = entry.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                errors.append(f"{prefix}: 'metadata' must be a mapping")

        return errors
