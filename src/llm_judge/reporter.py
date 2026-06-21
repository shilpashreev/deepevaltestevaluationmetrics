"""Report Generator — console, JSON, and HTML output strategies.

Transforms a ``SuiteResult`` into human-readable reports for different
output channels.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_judge.models import SuiteResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_FRAMEWORK_VERSION = "0.1.0"


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts.

    Handles nested dataclasses, lists, and dicts.  Non-serialisable
    values are converted to their ``str`` representation.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            field.name: _dataclass_to_dict(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp like ``20260620T061300``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class Reporter:
    """Generates evaluation reports in various formats.

    All methods are static — the class is purely a namespace.
    """

    # -- dispatch -----------------------------------------------------------

    @staticmethod
    def report(
        result: SuiteResult,
        format: str = "console",  # noqa: A002
        output_dir: str = "./results",
    ) -> str | None:
        """Generate a report in the requested format.

        Args:
            result: The suite evaluation result to report on.
            format: One of ``"console"``, ``"json"``, ``"html"``.
            output_dir: Directory for file-based reports.

        Returns:
            Output file path for file-based formats, or ``None`` for console.

        Raises:
            ValueError: If *format* is not recognised.
        """
        match format.lower():
            case "console":
                Reporter.console_report(result)
                return None
            case "json":
                return Reporter.json_report(result, output_dir)
            case "html":
                return Reporter.html_report(result, output_dir)
            case _:
                raise ValueError(
                    f"Unknown report format '{format}'. "
                    "Choose from: console, json, html"
                )

    # -- console ------------------------------------------------------------

    @staticmethod
    def console_report(result: SuiteResult) -> None:
        """Print a rich-formatted report to the terminal.

        Includes a summary panel and a per-test results table with colour
        coding for pass/fail/error status.
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError:
            # Graceful plain-text fallback
            Reporter._console_report_plain(result)
            return

        console = Console()

        # --- summary panel ---
        pass_rate = (
            (result.passed_tests / result.total_tests * 100)
            if result.total_tests
            else 0.0
        )
        summary_lines = [
            f"[bold]Suite:[/bold]        {result.suite_name}",
            f"[bold]Description:[/bold]  {result.description}" if result.description else "",
            f"[bold]Timestamp:[/bold]    {result.timestamp}",
            "",
            f"[bold]Total tests:[/bold]  {result.total_tests}",
            f"[bold green]Passed:[/bold green]       {result.passed_tests}",
            f"[bold red]Failed:[/bold red]       {result.failed_tests}",
            f"[bold yellow]Errors:[/bold yellow]       {result.error_tests}",
            "",
            f"[bold]Pass rate:[/bold]    {pass_rate:.1f}%",
            f"[bold]Avg score:[/bold]    {result.average_score:.3f}",
            f"[bold]Duration:[/bold]     {result.duration_seconds:.2f}s",
        ]
        panel = Panel(
            "\n".join(line for line in summary_lines if line is not None),
            title="📊 Evaluation Summary",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)

        # --- results table ---
        table = Table(
            title="Test Results",
            show_lines=True,
            title_style="bold",
        )
        table.add_column("Test ID", style="cyan", no_wrap=True)
        table.add_column("Tags", style="dim")
        table.add_column("Score", justify="right")

        # Collect criterion names from first result that has judgments
        criterion_names: list[str] = []
        for tr in result.results:
            if tr.judgments:
                criterion_names = [j.criterion for j in tr.judgments]
                break

        for cname in criterion_names:
            table.add_column(cname, justify="center")

        table.add_column("Status", justify="center")

        for tr in result.results:
            status_emoji = "✅" if tr.passed else ("⚠️" if tr.error else "❌")
            score_style = "green" if tr.aggregate_score >= 0.7 else (
                "yellow" if tr.aggregate_score >= 0.4 else "red"
            )

            row: list[str | Text] = [
                tr.test_case.id,
                ", ".join(tr.test_case.tags) if tr.test_case.tags else "—",
                Text(f"{tr.aggregate_score:.2f}", style=score_style),
            ]

            judgment_map = {j.criterion: j for j in tr.judgments}
            for cname in criterion_names:
                j = judgment_map.get(cname)
                if j:
                    v_style = "green" if j.score >= 0.7 else (
                        "yellow" if j.score >= 0.4 else "red"
                    )
                    row.append(Text(j.verdict, style=v_style))
                else:
                    row.append(Text("—", style="dim"))

            row.append(status_emoji)
            table.add_row(*row)

        console.print(table)

    @staticmethod
    def _console_report_plain(result: SuiteResult) -> None:
        """Fallback plain-text report when rich is not installed."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Suite: {result.suite_name}")
        print(f"Tests: {result.total_tests}  |  "
              f"Passed: {result.passed_tests}  |  "
              f"Failed: {result.failed_tests}  |  "
              f"Errors: {result.error_tests}")
        print(f"Avg Score: {result.average_score:.3f}  |  "
              f"Duration: {result.duration_seconds:.2f}s")
        print(sep)
        for tr in result.results:
            status = "PASS" if tr.passed else ("ERROR" if tr.error else "FAIL")
            print(f"  [{status}] {tr.test_case.id}  score={tr.aggregate_score:.2f}")
        print(sep + "\n")

    # -- JSON ---------------------------------------------------------------

    @staticmethod
    def json_report(result: SuiteResult, output_dir: str) -> str:
        """Write a JSON report to *output_dir* and return the file path.

        Uses a custom serializer to handle nested dataclass fields.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        payload = _dataclass_to_dict(result)
        payload["framework_version"] = _FRAMEWORK_VERSION

        filename = f"report_{_timestamp_slug()}.json"
        filepath = out / filename

        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("JSON report written to %s", filepath)
        return str(filepath)

    # -- HTML ---------------------------------------------------------------

    @staticmethod
    def html_report(result: SuiteResult, output_dir: str) -> str:
        """Render an HTML report via Jinja2 and write it to *output_dir*.

        The Jinja2 template is loaded from the ``templates/`` directory
        adjacent to the package root (or via ``importlib.resources``).

        Returns:
            Path to the generated HTML file.
        """
        try:
            from jinja2 import Environment, FileSystemLoader
        except ImportError as exc:
            raise ImportError(
                "Jinja2 is required for HTML reports. "
                "Install it with:  pip install jinja2"
            ) from exc

        # Locate template directory
        template_dir = Reporter._find_template_dir()
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        template = env.get_template("report.html")

        # Build template context
        results_data = _dataclass_to_dict(result.results)
        html_content = template.render(
            suite_name=result.suite_name,
            description=result.description,
            timestamp=result.timestamp,
            total_tests=result.total_tests,
            passed_tests=result.passed_tests,
            failed_tests=result.failed_tests,
            error_tests=result.error_tests,
            average_score=result.average_score,
            duration_seconds=result.duration_seconds,
            results=results_data,
            framework_version=_FRAMEWORK_VERSION,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filename = f"report_{_timestamp_slug()}.html"
        filepath = out / filename

        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(html_content)

        logger.info("HTML report written to %s", filepath)
        return str(filepath)

    @staticmethod
    def _find_template_dir() -> Path:
        """Resolve the ``templates/`` directory.

        Search order:
        1. ``<package_root>/../templates/``  (editable / development install)
        2. ``importlib.resources``          (installed package)
        """
        # Relative to this file: src/llm_judge/reporter.py → ../../templates
        pkg_dir = Path(__file__).resolve().parent  # src/llm_judge/
        candidates = [
            pkg_dir.parent.parent / "templates",  # repo root
            pkg_dir / "templates",                 # inside package
        ]
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "report.html").exists():
                return candidate

        # Try importlib.resources (Python 3.9+)
        try:
            import importlib.resources as pkg_resources

            ref = pkg_resources.files("llm_judge") / "templates"
            if ref.is_dir():  # type: ignore[union-attr]
                return Path(str(ref))
        except Exception:  # noqa: BLE001
            pass

        raise FileNotFoundError(
            "Cannot locate templates/report.html. "
            "Ensure the templates directory exists alongside the package root."
        )
