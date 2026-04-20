# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `subs_diff/`:
- `cli.py` and `__main__.py` provide the CLI entry points.
- `parser.py`, `align.py`, `heuristics.py`, `segments.py`, and `llm.py` implement parsing, matching, scoring, long-segment checks, and LLM verification.
- `report.py` and `reporter.py` generate JSON/HTML reports.
- Shared dataclasses and config types are in `types.py` and `config.py`.

Tests live in `tests/` and follow the same feature split (for example, `tests/test_align.py`, `tests/test_cli_filters.py`).

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode.
- `pip install -e ".[dev]"` installs development tools (`pytest`, `ruff`, `black`, `mypy`).
- `python -m subs_diff compare --stt A.srt --ref B.srt --out report.json` runs the main compare flow.
- `pytest -q` runs the test suite.
- `pytest --cov=subs_diff --cov-report=html` runs tests with coverage output in `htmlcov/`.
- `ruff check .` runs lint checks.
- `black .` formats code.
- `mypy subs_diff` runs strict type checking.

## Coding Style & Naming Conventions
- Python 3.10+ codebase; keep compatibility with versions listed in `pyproject.toml`.
- Use 4-space indentation and max line length `100` (Black/Ruff config).
- Use snake_case for functions/variables/modules; PascalCase for dataclasses/types.
- Prefer explicit type annotations; `mypy` is configured with `strict = true`.
- Keep modules focused; add new logic to existing domain modules before creating new top-level files.

## Testing Guidelines
- Framework: `pytest` (`tests/`, files named `test_*.py`).
- Add tests for every behavior change, especially CLI flags and alignment heuristics.
- Name tests by behavior, e.g. `test_compare_resumes_from_checkpoint`.
- For bug fixes, add a regression test that fails before the fix.

## Commit & Pull Request Guidelines
Current history uses short, direct subjects (for example, `long segments detection`, `Update .gitignore`). Follow that style, but make subjects specific and actionable.

- Commit message format: short imperative subject, optionally with scope (e.g. `align: tighten time window filter`).
- PRs should include: purpose, key changes, test evidence (`pytest`/lint/type-check output), and sample CLI command(s) for manual verification.
- Link related issues and attach report artifacts/screenshots when output format changes.

## Security & Configuration Tips
- Do not commit API keys or local config files.
- Prefer CLI/config storage for secrets (`subs_diff config set ...`) and keep generated reports/debug logs out of commits unless needed for fixtures.
