"""CLI Interface — ``llm-judge`` command-line tool.

Subcommands
-----------
run       Run a test suite and generate a report.
validate  Validate a suite YAML file without executing it.
criteria  List all built-in evaluation criteria.
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        _print_error("\n⏹  Interrupted by user.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        _print_error(f"Error: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-judge",
        description="LLM-as-Judge evaluation framework",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
    )

    # --- run ---------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run a test suite and generate a report",
    )
    run_parser.add_argument(
        "suite_path",
        help="Path to a YAML suite file or directory of suites",
    )
    run_parser.add_argument(
        "--config",
        "-c",
        dest="config_path",
        default=None,
        help="Path to a YAML/JSON config file",
    )
    run_parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider (e.g. openai, anthropic)",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (e.g. gpt-4o-mini)",
    )
    run_parser.add_argument(
        "--criteria",
        default=None,
        help="Comma-separated list of criteria names",
    )
    run_parser.add_argument(
        "--report",
        default=None,
        choices=["console", "json", "html"],
        help="Report format (default: from config or 'console')",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for file-based reports",
    )
    run_parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Disable concurrent evaluation (concurrency=1)",
    )
    run_parser.set_defaults(func=cmd_run)

    # --- validate ----------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a suite YAML file",
    )
    validate_parser.add_argument(
        "suite_path",
        help="Path to a YAML suite file",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # --- criteria ----------------------------------------------------------
    criteria_parser = subparsers.add_parser(
        "criteria",
        help="List available evaluation criteria",
    )
    criteria_parser.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list"],
        help="Action to perform (default: list)",
    )
    criteria_parser.set_defaults(func=cmd_criteria_list)

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand."""
    # Lazy imports to keep CLI startup fast
    from llm_judge.config import load_config
    from llm_judge.reporter import Reporter
    from llm_judge.runner import TestRunner

    console = _get_console()

    # Load config, then override with CLI arguments
    config = load_config(args.config_path)

    if args.provider is not None:
        config.provider = args.provider
    if args.model is not None:
        config.model = args.model
    if args.criteria is not None:
        config.criteria = [c.strip() for c in args.criteria.split(",") if c.strip()]
    if args.report is not None:
        config.report_format = args.report
    if args.output is not None:
        config.output_dir = args.output
    if args.sequential:
        config.concurrency = 1

    # Validate that we have an API key
    if not config.api_key:
        _print_error(
            "No API key configured.\n"
            "Set it via:\n"
            "  • Environment variable (e.g. OPENAI_API_KEY)\n"
            "  • Config file: api_key field\n"
            "  • Or pass --config with a file containing the key"
        )
        sys.exit(1)

    console.print(
        f"[bold blue]🔬 Running suite:[/bold blue] {args.suite_path}\n"
        f"   Provider: {config.provider} / {config.model}\n"
        f"   Criteria: {', '.join(config.criteria)}\n"
        f"   Concurrency: {config.concurrency}\n"
    )

    try:
        runner = TestRunner(config)
    except Exception as exc:
        _print_error(f"Failed to initialise runner: {exc}")
        sys.exit(1)

    try:
        result = runner.run_suite_sync(args.suite_path)
    except FileNotFoundError as exc:
        _print_error(str(exc))
        sys.exit(1)
    except ValueError as exc:
        _print_error(f"Suite validation error: {exc}")
        sys.exit(1)

    # Generate report
    output_path = Reporter.report(
        result,
        format=config.report_format,
        output_dir=config.output_dir,
    )

    if output_path:
        console.print(f"\n[bold green]📄 Report written to:[/bold green] {output_path}")

    # Exit code reflects pass/fail
    if result.failed_tests > 0 or result.error_tests > 0:
        sys.exit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    """Execute the ``validate`` subcommand."""
    from llm_judge.runner import TestRunner

    console = _get_console()
    errors = TestRunner.validate_suite(args.suite_path)

    if not errors:
        console.print(f"[bold green]✅ {args.suite_path} is valid[/bold green]")
    else:
        console.print(f"[bold red]❌ {args.suite_path} has {len(errors)} error(s):[/bold red]")
        for err in errors:
            console.print(f"   • {err}")
        sys.exit(1)


def cmd_criteria_list(args: argparse.Namespace) -> None:
    """Execute the ``criteria list`` subcommand."""
    from llm_judge.criteria import CRITERIA_REGISTRY

    console = _get_console()

    if not CRITERIA_REGISTRY:
        console.print("[yellow]No criteria registered.[/yellow]")
        return

    try:
        from rich.table import Table

        table = Table(title="Available Evaluation Criteria", show_lines=True)
        table.add_column("Name", style="cyan bold", no_wrap=True)
        table.add_column("Description")
        table.add_column("Verdicts", style="dim")

        for name, criterion in sorted(CRITERIA_REGISTRY.items()):
            table.add_row(
                name,
                criterion.description,
                ", ".join(criterion.verdicts),
            )
        console.print(table)

    except ImportError:
        # Plain-text fallback
        print("\nAvailable Criteria:")
        print("-" * 60)
        for name, criterion in sorted(CRITERIA_REGISTRY.items()):
            print(f"  {name:20s}  {criterion.description}")
            print(f"  {'':20s}  Verdicts: {', '.join(criterion.verdicts)}")
        print()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _get_console():
    """Return a rich Console, or a lightweight stub if rich is unavailable."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return _PlainConsole()


class _PlainConsole:
    """Minimal stand-in for ``rich.Console`` when rich is not installed."""

    @staticmethod
    def print(*args: object, **kwargs: object) -> None:  # noqa: A003
        import re

        text = " ".join(str(a) for a in args)
        # Strip rich markup tags for plain output
        text = re.sub(r"\[/?[^\]]*\]", "", text)
        print(text)


def _print_error(message: str) -> None:
    """Print an error message to stderr with optional rich formatting."""
    console = _get_console()
    try:
        console.print(f"[bold red]{message}[/bold red]", highlight=False)
    except Exception:  # noqa: BLE001
        print(message, file=sys.stderr)


# ---------------------------------------------------------------------------
# Allow ``python -m llm_judge.cli``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
