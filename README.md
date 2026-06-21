# LLM Judge 🧑‍⚖️

**A structured test framework for evaluating LLM outputs using LLM-as-a-Judge.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)

---

## Overview

**LLM Judge** is a Python framework that automates the evaluation of language model outputs using the *LLM-as-a-Judge* paradigm. You define test suites in YAML, specify evaluation criteria, and let a judge LLM score each response for accuracy, safety, coherence, and more.

The framework supports multiple LLM providers (OpenAI, Anthropic, Google), ships with seven built-in evaluation criteria, and produces rich reports in console, JSON, and HTML formats. It is designed for CI/CD integration, regression testing, and systematic quality assurance of LLM-powered applications.

## Features

- **YAML-based test suites** — declarative, version-controllable test definitions
- **7 built-in evaluation criteria** — accuracy, relevance, coherence, safety, completeness, conciseness, instruction-following
- **Multi-provider support** — OpenAI, Anthropic, Google Generative AI
- **Pluggable architecture** — define custom criteria with system prompts, verdict labels, and scoring
- **Rich reporting** — console tables, structured JSON, and HTML reports
- **Async execution** — parallel evaluation for faster suite runs
- **Tag-based filtering** — run subsets of tests by tag
- **CI/CD ready** — exit codes, machine-readable output, threshold-based pass/fail

---

## Quick Start

```bash
# 1. Install
pip install -e ".[openai]"

# 2. Set your API key
export OPENAI_API_KEY="sk-..."

# 3. Run a test suite
llm-judge run test_suites/basic_qa.yaml
```

---

## Installation

### From source (recommended during development)

```bash
git clone <repo-url>
cd llm-judge-framework
pip install -e ".[dev]"
```

### With specific providers

```bash
# OpenAI only
pip install -e ".[openai]"

# Anthropic only
pip install -e ".[anthropic]"

# Google Generative AI only
pip install -e ".[google]"

# All providers
pip install -e ".[all]"

# All providers + dev tools
pip install -e ".[all,dev]"
```

---

## Usage

### Writing Test Suites

Test suites are YAML files with two top-level keys: `suite` (metadata) and `tests` (test cases).

```yaml
suite:
  name: "My Evaluation Suite"
  description: "Evaluate model responses for Q&A accuracy"
  criteria:
    - accuracy
    - relevance
    - coherence
  pass_threshold: 0.7

tests:
  - id: "geography-01"
    prompt: "What is the capital of France?"
    response: "The capital of France is Paris."
    reference: "Paris"
    tags: [geography, factual]

  - id: "coding-01"
    prompt: "How do you create a list in Python?"
    response: "Use square brackets: my_list = [1, 2, 3]"
    reference: "Use square brackets [] or list()"
    tags: [programming]
```

#### Suite Fields

| Field | Required | Description |
|-------|----------|-------------|
| `suite.name` | ✅ | Human-readable suite name |
| `suite.description` | ✅ | What this suite evaluates |
| `suite.criteria` | ✅ | List of criteria to evaluate against |
| `suite.pass_threshold` | ✅ | Minimum aggregate score to pass (0.0–1.0) |

#### Test Case Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique test identifier |
| `prompt` | ✅ | The original prompt |
| `response` | ✅ | The LLM response to evaluate |
| `reference` | ❌ | Gold-standard reference answer |
| `tags` | ❌ | List of tags for filtering |

### Running Tests

```bash
# Run a single suite
llm-judge run test_suites/basic_qa.yaml

# Run with a specific provider and model
llm-judge run test_suites/basic_qa.yaml --provider openai --model gpt-4o

# Filter by tags
llm-judge run test_suites/basic_qa.yaml --tags geography,science

# Output JSON report
llm-judge run test_suites/basic_qa.yaml --format json --output results.json

# Validate suite without running
llm-judge validate test_suites/basic_qa.yaml

# List available criteria
llm-judge criteria
```

### Configuration

Create a `llm_judge.yaml` config file in your project root or pass `--config`:

```yaml
provider: openai
model: gpt-4o
temperature: 0.0
max_retries: 3
timeout: 30
concurrency: 4
```

Environment variables take precedence over config file values:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI |
| `ANTHROPIC_API_KEY` | API key for Anthropic |
| `GOOGLE_API_KEY` | API key for Google Generative AI |
| `LLM_JUDGE_PROVIDER` | Default provider name |
| `LLM_JUDGE_MODEL` | Default model name |

### Custom Criteria

Define custom evaluation criteria in Python:

```python
from llm_judge.models import EvalCriterion
from llm_judge.criteria import CRITERIA_REGISTRY

custom = EvalCriterion(
    name="tone",
    description="Evaluate whether the response has a professional tone",
    system_prompt=(
        "You are an expert evaluator. Assess whether the AI assistant's "
        "response maintains a professional and appropriate tone."
    ),
    evaluation_prompt=(
        "Prompt: {prompt}\n\n"
        "Response: {response}\n\n"
        "Evaluate the tone of the response. Is it professional, "
        "neutral, or unprofessional?\n\n"
        "Verdict (professional / neutral / unprofessional):"
    ),
    verdicts=["professional", "neutral", "unprofessional"],
    verdict_scores={
        "professional": 1.0,
        "neutral": 0.5,
        "unprofessional": 0.0,
    },
)

# Register it
CRITERIA_REGISTRY["tone"] = custom
```

Then reference `tone` in your YAML suite's `criteria` list.

---

## Built-in Criteria

| Criterion | Description | Verdicts |
|-----------|-------------|----------|
| **accuracy** | Factual correctness against the reference answer | correct · partial · incorrect |
| **relevance** | How well the response addresses the prompt | relevant · partial · irrelevant |
| **coherence** | Logical flow, clarity, and readability | coherent · partial · incoherent |
| **safety** | Absence of harmful, biased, or dangerous content | safe · partial · unsafe |
| **completeness** | Whether the response fully answers the question | complete · partial · incomplete |
| **conciseness** | Appropriate brevity without unnecessary information | concise · partial · verbose |
| **instruction_following** | Adherence to explicit instructions in the prompt | follows · partial · ignores |

Each criterion uses a 3-tier verdict system mapping to scores: **1.0** (best), **0.5** (partial), **0.0** (worst). The aggregate score for a test case is the mean across all evaluated criteria.

---

## Providers

| Provider | Package | Example Models |
|----------|---------|----------------|
| **OpenAI** | `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| **Anthropic** | `anthropic` | `claude-sonnet-4-20250514`, `claude-haiku-4-20250414` |
| **Google** | `google-genai` | `gemini-2.5-pro`, `gemini-2.5-flash` |

Install provider-specific dependencies:

```bash
pip install -e ".[openai]"      # OpenAI
pip install -e ".[anthropic]"   # Anthropic
pip install -e ".[google]"      # Google
pip install -e ".[all]"         # All providers
```

---

## Report Formats

### Console (default)

Rich-formatted table output with color-coded pass/fail status:

```
┌─────────────────────┬──────────┬───────┬────────┐
│ Test ID             │ Verdict  │ Score │ Status │
├─────────────────────┼──────────┼───────┼────────┤
│ capital-france      │ correct  │  1.0  │   ✅   │
│ python-list         │ correct  │  1.0  │   ✅   │
│ incorrect-capital   │ incorrect│  0.0  │   ❌   │
└─────────────────────┴──────────┴───────┴────────┘

Suite: Basic QA Accuracy
Passed: 6/8 (75.0%)  |  Average Score: 0.81
```

### JSON

Machine-readable output for CI/CD integration:

```bash
llm-judge run suite.yaml --format json --output results.json
```

### HTML

Self-contained HTML report with an interactive **score matrix** and click-to-expand drill-down.

```bash
llm-judge run suite.yaml --format html --output report.html
```

#### 🔗 View the live sample report

A generated sample report is included at [`results/report.html`](results/report.html). GitHub
displays `.html` as source, so use the rendered preview link below — **no download required**:

> **▶ [View the rendered HTML report](https://htmlpreview.github.io/?https://github.com/shilpashreev/deepevaltestevaluationmetrics/blob/main/results/report.html)**

#### Report structure

**1. Dashboard** — totals, pass-rate ring, average score, and duration at the top.

**2. Score matrix** — one **row per test case**, one **column per criterion** (sourced from
`CRITERIA_REGISTRY`), with a color-coded score in every cell (🟩 ≥ 0.70 · 🟧 0.40–0.69 ·
🟥 < 0.40), plus an aggregate and pass/fail status:

| Test Case | accuracy | relevance | coherence | safety | completeness | conciseness | instruction_following | Aggregate | Status |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| capital-france | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | **0.93** | ✅ |
| incorrect-capital | 0.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | **0.79** | ✅ |
| partial-answer | 0.50 | 0.50 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 | **0.64** | ❌ |
| safe-topic | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | ✅ |

**3. Click-to-expand drill-down** — clicking any row reveals, for that test:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Input — Prompt        Expected Output — Reference     Actual Output     │
│  "Name three primary   "Red, blue, and yellow…"        "Red and blue     │
│   colors."                                              are primary…"     │
├────────────────────────────────────────────────────────────────────────┤
│  Why it failed — per criterion   (aggregate 0.64, threshold 0.70)        │
│   🟧 accuracy      · partially_correct · 0.50  — omits yellow; 1 of 3…   │
│   🟧 relevance     · partially_relevant · 0.50 — only two of three…      │
│   🟥 completeness  · incomplete · 0.00         — asks for THREE, got two │
│   🟩 coherence     · coherent · 1.00           — clear and well-formed   │
└────────────────────────────────────────────────────────────────────────┘
```

Each criterion card shows its **verdict**, **score**, and the judge's **reasoning** — so a
failing test makes plain *which* criteria dragged the score down and *why*. The report also
supports searching by test ID and filtering by Pass / Fail / Error and by tag.

> **Note:** the scores in the bundled sample report are illustrative (no live model was called).
> Run the CLI with a configured provider to produce reports from real judge output.

---

## Project Structure

```
llm-judge-framework/
├── pyproject.toml              # Project metadata & dependencies
├── README.md                   # This file
├── src/
│   └── llm_judge/
│       ├── __init__.py         # Package root
│       ├── models.py           # Core dataclasses (TestCase, Judgment, etc.)
│       ├── criteria.py         # Built-in evaluation criteria registry
│       ├── runner.py           # Test suite loader & runner
│       ├── providers/          # LLM provider adapters
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract provider interface
│       │   ├── openai.py       # OpenAI adapter
│       │   ├── anthropic.py    # Anthropic adapter
│       │   └── google.py       # Google GenAI adapter
│       ├── reporting/          # Output formatters
│       │   ├── __init__.py
│       │   ├── console.py      # Rich console reporter
│       │   ├── json.py         # JSON reporter
│       │   └── html.py         # HTML reporter
│       └── cli.py              # CLI entry point
├── test_suites/                # Example YAML test suites
│   ├── basic_qa.yaml           # Q&A accuracy tests
│   └── safety.yaml             # Safety & guardrails tests
└── tests/                      # Unit tests
    ├── test_models.py
    ├── test_criteria.py
    └── test_runner.py
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork & branch** — Create a feature branch from `main`
2. **Write tests** — All new features should have corresponding tests
3. **Follow style** — Use type hints, Google-style docstrings, `from __future__ import annotations`
4. **Run checks** — Ensure `pytest` passes before submitting

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_models.py -v
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
