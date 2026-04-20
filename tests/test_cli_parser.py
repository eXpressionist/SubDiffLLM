"""Tests for CLI parser wiring."""

from subs_diff.cli import create_parser


def test_compare_parser_accepts_required_arguments():
    parser = create_parser()

    args = parser.parse_args(
        ["compare", "--stt", "a.srt", "--ref", "b.srt", "--out", "report.json", "--llm", "off"]
    )

    assert args.command == "compare"
    assert args.stt == "a.srt"
    assert args.ref == "b.srt"
    assert args.out == "report.json"
    assert args.llm == "off"


def test_legacy_compare_arguments_are_still_accepted():
    parser = create_parser()

    args = parser.parse_args(["--stt", "a.srt", "--ref", "b.srt", "--out", "report.json"])

    assert args.command is None
    assert args.stt == "a.srt"
    assert args.ref == "b.srt"
    assert args.out == "report.json"


def test_config_parser_accepts_set_command():
    parser = create_parser()

    args = parser.parse_args(["config", "set", "--api-key", "sk-test"])

    assert args.command == "config"
    assert args.config_command == "set"
    assert args.api_key == "sk-test"
