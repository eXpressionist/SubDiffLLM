"""Тесты resume-чекпоинта в CLI."""

import json

from subs_diff.cli import _load_resume_checkpoint


def test_load_resume_checkpoint(tmp_path):
    report_path = tmp_path / "report.json"
    payload = {
        "metadata": {
            "generated_at": "2026-01-01T00:00:00Z",
            "stt_file": "a.srt",
            "ref_file": "b.srt",
            "config": {"partial": True, "processed": 7, "total": 10},
        },
        "issues": [
            {
                "issue_id": "ISS-001",
                "time_range": {"start_ms": 1000, "end_ms": 2000},
                "a_segments": [1],
                "b_segments": [1],
                "severity": "med",
                "category": "missing_content",
                "short_reason": "test",
                "evidence": "A vs B",
                "forced_detected": False,
                "llm_verdict": None,
                "a_text": "A",
                "b_text": "B",
            }
        ],
        "summary": {"total_issues": 1, "by_severity": {"med": 1}, "by_category": {"missing_content": 1}},
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    issues, processed = _load_resume_checkpoint(report_path)
    assert processed == 7
    assert len(issues) == 1
    assert issues[0].issue_id == "ISS-001"

