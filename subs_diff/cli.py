"""CLI entry point for SubDiff."""

import argparse
import asyncio
import logging
import sys
from typing import Any

from subs_diff.comparison import run_comparison
from subs_diff.config import get_api_key, get_config_path, load_config, save_llm_config
from subs_diff.types import Config, LLMMode


def create_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(
        prog="subs-diff",
        description="Compare two .srt subtitle files and report likely STT mistakes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s compare --stt audio.srt --ref reference.srt --out report.json
  %(prog)s compare --stt A.srt --ref B.srt --out report.json --llm off
  %(prog)s config set --api-key sk-... --api-model meta-llama/llama-3.2-3b-instruct
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")
    _add_compare_parser(subparsers)
    _add_config_parser(subparsers)
    _add_legacy_compare_arguments(parser)
    return parser


def _add_compare_parser(subparsers: Any) -> None:
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two SRT files",
        description="Compare two subtitle files",
    )
    _add_compare_arguments(compare_parser, suppress_help=False)


def _add_config_parser(subparsers: Any) -> None:
    config_parser = subparsers.add_parser(
        "config",
        help="Manage saved LLM settings",
        description="Manage saved LLM settings",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config commands")

    set_parser = config_subparsers.add_parser("set", help="Save settings")
    set_parser.add_argument("--api-key", type=str, default=None, help="API key")
    set_parser.add_argument("--api-url", type=str, default=None, help="API base URL")
    set_parser.add_argument("--api-model", type=str, default=None, help="API model")
    set_parser.add_argument("--local-url", type=str, default=None, help="Local LLM URL")
    set_parser.add_argument("--local-model", type=str, default=None, help="Local model")
    set_parser.add_argument("--site-url", type=str, default=None, help="OpenRouter referer URL")
    set_parser.add_argument("--site-name", type=str, default=None, help="OpenRouter title")
    set_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    show_parser = config_subparsers.add_parser("show", help="Show settings")
    show_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    clear_parser = config_subparsers.add_parser("clear", help="Clear settings")
    clear_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")


def _add_legacy_compare_arguments(parser: argparse.ArgumentParser) -> None:
    """Support the older form: subs-diff --stt A.srt --ref B.srt --out report.json."""
    _add_compare_arguments(parser, suppress_help=True)


def _add_compare_arguments(parser: argparse.ArgumentParser, *, suppress_help: bool) -> None:
    help_text = argparse.SUPPRESS if suppress_help else None

    parser.add_argument(
        "--stt",
        required=not suppress_help,
        type=str,
        help=help_text or "STT SRT path",
    )
    parser.add_argument(
        "--ref",
        required=not suppress_help,
        type=str,
        help=help_text or "Reference SRT path",
    )
    parser.add_argument(
        "--out",
        required=not suppress_help,
        type=str,
        help=help_text or "Output JSON report path",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=1.0,
        help=help_text or "Timing tolerance in seconds",
    )
    parser.add_argument(
        "--max-merge",
        type=int,
        default=5,
        help=help_text or "Max adjacent segments to merge",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help=help_text or "Minimum similarity threshold for candidates",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["off", "auto", "local", "api"],
        default="auto",
        help=help_text or "LLM mode",
    )
    parser.add_argument("--llm-model", type=str, default=None, help=help_text or "API model")
    parser.add_argument(
        "--llm-local-model",
        type=str,
        default=None,
        help=help_text or "Local model",
    )
    parser.add_argument("--llm-url", type=str, default=None, help=help_text or "Local LLM URL")
    parser.add_argument("--llm-key", type=str, default=None, help=help_text or "API key")
    parser.add_argument("--llm-api-url", type=str, default=None, help=help_text or "API base URL")
    parser.add_argument(
        "--llm-site-url",
        type=str,
        default=None,
        help=help_text or "OpenRouter referer URL",
    )
    parser.add_argument(
        "--llm-site-name",
        type=str,
        default=None,
        help=help_text or "OpenRouter title",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help=help_text or "Do not write HTML report",
    )
    parser.add_argument("--checkpoint", action="store_true", help=help_text or "Write checkpoints")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help=help_text or "Checkpoint interval in candidates",
    )
    parser.add_argument("--resume", action="store_true", help=help_text or "Resume checkpoint")
    parser.add_argument(
        "--llm-context-size",
        type=int,
        default=1,
        help=help_text or "Neighboring A segments to send as context",
    )
    parser.add_argument(
        "--ref-only-missing",
        action="store_true",
        help=help_text or "Only keep content present in REF but missing in STT",
    )
    parser.add_argument("--llm-debug", action="store_true", help=help_text or "Log LLM traffic")
    parser.add_argument(
        "--llm-debug-file",
        type=str,
        default="llm_debug.log",
        help=help_text or "LLM debug log path",
    )
    parser.add_argument(
        "--check-long-segments",
        action="store_true",
        help=help_text or "Report overlong STT segments",
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=6.0,
        help=help_text or "Maximum segment duration in seconds",
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help=help_text or "Use batch/full-context LLM verification",
    )
    parser.add_argument(
        "--full-context-max-pairs",
        type=int,
        default=50,
        help=help_text or "Pairs per full-context batch",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help=help_text or "Verbose output")


def main() -> int:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "config":
        if hasattr(args, "verbose") and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        return handle_config_command(args)

    if args.command == "compare" or args.command is None:
        if not args.stt or not args.ref or not args.out:
            parser.print_help()
            return 1
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        return run_compare(args)

    parser.print_help()
    return 1


def handle_config_command(args: argparse.Namespace) -> int:
    """Handle the config subcommand."""
    if args.config_command == "set":
        config_path = save_llm_config(
            api_key=args.api_key,
            api_url=args.api_url,
            api_model=args.api_model,
            local_url=args.local_url,
            local_model=args.local_model,
            site_url=args.site_url,
            site_name=args.site_name,
        )
        print(f"Конфигурация сохранена: {config_path}")

        config = load_config()
        llm = config.get("llm", {})
        print("\nСохранённые настройки:")
        if llm.get("api_key"):
            key_preview = llm["api_key"][:8] + "..." if len(llm["api_key"]) > 8 else llm["api_key"]
            print(f"  API Key: {key_preview}")
        print(f"  API URL: {llm.get('api_url', 'N/A')}")
        print(f"  API Model: {llm.get('api_model', 'N/A')}")
        print(f"  Local URL: {llm.get('local_url', 'N/A')}")
        print(f"  Local Model: {llm.get('local_model', 'N/A')}")
        return 0

    if args.config_command == "show":
        config = load_config()
        config_path = get_config_path()

        if not config:
            print(f"Конфигурация пуста. Файл: {config_path}")
            return 0

        print(f"Конфигурационный файл: {config_path}")
        print("\nСодержимое:")
        llm = config.get("llm", {})
        if llm:
            print("\nLLM настройки:")
            if llm.get("api_key"):
                key_preview = (
                    llm["api_key"][:8] + "..." if len(llm["api_key"]) > 8 else llm["api_key"]
                )
                print(f"  API Key: {key_preview}")
            print(f"  API URL: {llm.get('api_url', 'N/A')}")
            print(f"  API Model: {llm.get('api_model', 'N/A')}")
            print(f"  Local URL: {llm.get('local_url', 'N/A')}")
            print(f"  Local Model: {llm.get('local_model', 'N/A')}")
            print(f"  Site URL: {llm.get('site_url', 'N/A')}")
            print(f"  Site Name: {llm.get('site_name', 'N/A')}")
        else:
            print("  (пусто)")
        return 0

    if args.config_command == "clear":
        config_path = get_config_path()
        if config_path.exists():
            config_path.unlink()
            print(f"Конфигурация удалена: {config_path}")
        else:
            print("Конфигурация уже пуста")
        return 0

    print("Используйте: subs-diff config {set|show|clear}")
    return 1


def run_compare(args: argparse.Namespace) -> int:
    """Run comparison from parsed CLI arguments."""
    saved_config = load_config()
    llm_config = saved_config.get("llm", {})

    llm_api_key = args.llm_key or get_api_key()
    llm_model = args.llm_model or llm_config.get("api_model", "meta-llama/llama-3.2-3b-instruct")
    llm_local_model = args.llm_local_model or llm_config.get("local_model", "llama3.2")
    llm_url = args.llm_url or llm_config.get("local_url", "http://localhost:11434")
    llm_api_url = args.llm_api_url or llm_config.get("api_url", "https://openrouter.ai/api/v1")
    llm_site_url = args.llm_site_url or llm_config.get(
        "site_url",
        "https://github.com/eXpressionist/SubDiffLLM",
    )
    llm_site_name = args.llm_site_name or llm_config.get("site_name", "SubDiff")

    config = Config(
        stt_file=args.stt,
        ref_file=args.ref,
        out_file=args.out,
        time_tol=args.time_tol,
        max_merge=args.max_merge,
        min_score=args.min_score,
        llm_mode=LLMMode(args.llm),
        llm_model=llm_model,
        llm_local_model=llm_local_model,
        llm_url=llm_url,
        llm_api_url=llm_api_url,
        llm_api_key=llm_api_key,
        llm_site_url=llm_site_url,
        llm_site_name=llm_site_name,
        no_html=args.no_html,
        checkpoint=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        ref_only_missing=args.ref_only_missing,
        verbose=args.verbose,
        llm_context_size=args.llm_context_size,
        llm_debug=args.llm_debug,
        llm_debug_file=args.llm_debug_file,
        check_long_segments=getattr(args, "check_long_segments", False),
        max_segment_duration=getattr(args, "max_segment_duration", 6.0),
        full_context=getattr(args, "full_context", False),
        full_context_max_pairs=getattr(args, "full_context_max_pairs", 50),
    )

    return asyncio.run(run_comparison(config))


if __name__ == "__main__":
    sys.exit(main())
