"""Tests for report JSON serialization."""

from subs_diff.report import (
    REPORT_SCHEMA_VERSION,
    generate_html_from_json,
    generate_report,
    load_report_json,
    report_to_dict,
    save_report_json,
)
from subs_diff.types import Category, Issue, LLMVerdict, Severity


def _issue() -> Issue:
    verdict = LLMVerdict(
        severity=Severity.HIGH,
        category=Category.WRONG_NAMED_ENTITY,
        short_reason="Wrong name",
        evidence="A vs B",
        confidence=0.9,
        suggested_fix="Sherlock Holmes",
    )
    return Issue(
        issue_id="ISS-001",
        time_range=(1000, 2000),
        a_segments=[1],
        b_segments=[1],
        severity=Severity.HIGH,
        category=Category.WRONG_NAMED_ENTITY,
        short_reason="Wrong name",
        evidence="A vs B",
        forced_detected=False,
        llm_verdict=verdict,
        a_text="Sherlock Stout",
        b_text="Sherlock Holmes",
    )


def test_report_to_dict_includes_schema_version():
    report = generate_report([_issue()], "a.srt", "b.srt", {"llm_mode": "off"})

    data = report_to_dict(report)

    assert data["schema_version"] == REPORT_SCHEMA_VERSION


def test_load_report_json_roundtrips_issue_and_verdict(tmp_path):
    report_path = tmp_path / "report.json"
    report = generate_report([_issue()], "a.srt", "b.srt", {"llm_mode": "off"})

    save_report_json(report, report_path)
    loaded = load_report_json(report_path)

    assert loaded.metadata.stt_file == "a.srt"
    assert len(loaded.issues) == 1
    assert loaded.issues[0].llm_verdict is not None
    assert loaded.issues[0].llm_verdict.suggested_fix == "Sherlock Holmes"


def test_generate_html_from_json_uses_report_loader(tmp_path):
    report_path = tmp_path / "report.json"
    report = generate_report([_issue()], "a.srt", "b.srt", {"llm_mode": "off"})
    save_report_json(report, report_path)

    html = generate_html_from_json(report_path)

    assert "SubDiff Report" in html
    assert "Sherlock Stout" in html
