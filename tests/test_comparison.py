"""Integration tests for the comparison pipeline."""

import asyncio
from pathlib import Path

from subs_diff.comparison import run_comparison
from subs_diff.report import load_report_json
from subs_diff.types import Config, LLMMode

SRT_CONTENT = """1
00:00:01,000 --> 00:00:02,000
Hello there.

2
00:00:03,000 --> 00:00:04,000
General Kenobi.
"""


def test_run_comparison_writes_json_report_with_llm_off():
    stt_path = Path("comparison_test_stt.srt")
    ref_path = Path("comparison_test_ref.srt")
    out_path = Path("comparison_test_report.json")
    html_path = out_path.with_suffix(".html")

    for path in (stt_path, ref_path, out_path, html_path):
        if path.exists():
            path.unlink()

    try:
        stt_path.write_text(SRT_CONTENT, encoding="utf-8")
        ref_path.write_text(SRT_CONTENT, encoding="utf-8")

        config = Config(
            stt_file=str(stt_path),
            ref_file=str(ref_path),
            out_file=str(out_path),
            llm_mode=LLMMode.OFF,
            no_html=True,
        )

        exit_code = asyncio.run(run_comparison(config))

        assert exit_code == 0
        assert out_path.exists()
        assert not html_path.exists()

        report = load_report_json(out_path)
        assert report.metadata.stt_file == str(stt_path)
        assert report.metadata.ref_file == str(ref_path)
        assert report.metadata.config["llm_mode"] == "off"
        assert report.summary.total_issues == 0
        assert report.issues == []
    finally:
        for path in (stt_path, ref_path, out_path, html_path):
            if path.exists():
                path.unlink()
