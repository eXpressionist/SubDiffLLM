"""CLI интерфейс для SubDiff."""

import argparse
import asyncio
import json
import logging
import re
import signal
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from subs_diff.types import (
    Config,
    LLMMode,
    Issue,
    LLMVerdict,
    Severity,
    Category,
    Candidate,
    MergedSegment,
    SimilarityMetrics,
    LongSegmentInfo,
    SplitSuggestion,
)
from subs_diff.parser import parse_srt_file, detect_forced_segments
from subs_diff.align import align_segments, merge_segments
from subs_diff.llm import create_llm_client
from subs_diff.heuristics import compute_similarity
from subs_diff.segments import analyze_long_segments, DEFAULT_MAX_DURATION_MS
from subs_diff.report import (
    generate_report,
    save_report_json,
    save_report_html,
    create_issue_from_candidate,
    generate_html_from_json,
)
from subs_diff.config import (
    load_config,
    save_llm_config,
    get_api_key,
    get_config_path,
)


# Глобальная переменная для сохранения состояния при прерывании
_interrupt_state = {
    "issues": [],
    "total": 0,
    "processed": 0,
    "out_file": None,
}


def _save_checkpoint(issues, out_file, processed=0, total=0):
    """Сохранить промежуточные результаты."""
    if out_file:
        try:
            report = generate_report(
                issues=issues,
                stt_file="",
                ref_file="",
                config={"partial": True, "processed": processed, "total": total},
            )
            save_report_json(report, out_file)
            logging.info(f"Чекпоинт сохранён: {processed}/{total} проблем обработано")
        except Exception as e:
            logging.error(f"Ошибка сохранения чекпоинта: {e}")


def _handle_interrupt(signum, frame):
    """Обработчик прерывания (Ctrl+C)."""
    logging.warning("\nПрерывание пользователем...")
    if _interrupt_state["issues"] and _interrupt_state["out_file"]:
        _save_checkpoint(
            _interrupt_state["issues"],
            _interrupt_state["out_file"],
            _interrupt_state["processed"],
            _interrupt_state["total"],
        )
        logging.info(f"Частичные результаты сохранены в {_interrupt_state['out_file']}")
    sys.exit(1)


logger = logging.getLogger(__name__)

TRIVIAL_MISSING_TOKENS = {
    "а", "и", "но", "да", "же", "вот", "ну", "ли", "то", "ведь", "типа",
    "в", "во", "на", "к", "ко", "у", "с", "со", "из", "от", "до", "по", "за",
    "про", "для", "при", "о", "об", "обо", "под", "над", "через", "между",
    "бы", "б", "просто", "как", "так", "это", "вот",
}


def _load_resume_checkpoint(out_file: Path) -> tuple[list[Issue], int]:
    """
    Загрузить partial-checkpoint из JSON отчёта.

    Returns:
        (issues, processed_count)
    """
    if not out_file.exists():
        return [], 0

    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    cfg = metadata.get("config", {})
    processed = int(cfg.get("processed", 0)) if cfg.get("partial") else 0

    issues: list[Issue] = []
    for issue_data in data.get("issues", []):
        verdict_data = issue_data.get("llm_verdict")
        llm_verdict = None
        if verdict_data:
            llm_verdict = LLMVerdict(
                severity=Severity(verdict_data["severity"]),
                category=Category(verdict_data["category"]),
                short_reason=verdict_data["short_reason"],
                evidence=verdict_data["evidence"],
                confidence=float(verdict_data.get("confidence", 0.5)),
                suggested_fix=verdict_data.get("suggested_fix"),
                is_forced=bool(verdict_data.get("is_forced", False)),
            )

        issues.append(
            Issue(
                issue_id=issue_data["issue_id"],
                time_range=(
                    issue_data["time_range"]["start_ms"],
                    issue_data["time_range"]["end_ms"],
                ),
                a_segments=issue_data["a_segments"],
                b_segments=issue_data["b_segments"],
                severity=Severity(issue_data["severity"]),
                category=Category(issue_data["category"]),
                short_reason=issue_data["short_reason"],
                evidence=issue_data["evidence"],
                forced_detected=bool(issue_data["forced_detected"]),
                llm_verdict=llm_verdict,
                a_text=issue_data.get("a_text", ""),
                b_text=issue_data.get("b_text", ""),
            )
        )

    return issues, processed


def _should_skip_trivial_missing_content(candidate, llm_verdict: Optional[LLMVerdict]) -> bool:
    """Не добавлять issue, если missing_content основан только на мелких служебных словах."""
    if llm_verdict is None or llm_verdict.category != Category.MISSING_CONTENT:
        return False

    reason = llm_verdict.short_reason.lower()
    m = re.search(r"пропущено\s+'([^']+)'", reason)
    if m and m.group(1).strip().lower() in TRIVIAL_MISSING_TOKENS:
        return True

    # LLM иногда ошибочно считает, что "пропущен вопрос о X", хотя X есть в обеих репликах.
    # Проверяем фокус вопроса по грубому корню (например "усах" ~ "усы").
    q = re.search(r"пропущен\s+вопрос\s+о\s+([^\s.,!?]+)", reason)
    if q:
        focus = q.group(1).strip().lower()
        if len(focus) >= 3:
            root_len = 2 if len(focus) <= 4 else 3
            root = focus[:root_len]
            a_text = candidate.a_segment.text.lower()
            b_text = candidate.b_segment.text.lower()
            if root in a_text and root in b_text:
                return True

    a_vocab = {t.lower() for t in candidate.a_segment.tokens}
    missing_tokens = [t.lower() for t in candidate.b_segment.tokens if t.lower() not in a_vocab]
    if not missing_tokens:
        return True

    meaningful = [
        t for t in missing_tokens
        if t not in TRIVIAL_MISSING_TOKENS
    ]

    # Если пропущены только частицы/предлоги — это не issue.
    if not meaningful:
        return True

    return False


def _should_keep_issue_in_ref_only_mode(
    llm_verdict: Optional[LLMVerdict],
    is_forced_like: bool,
) -> bool:
    """В режиме ref-only оставляем только missing_content/forced_mismatch."""
    if is_forced_like:
        return True
    if llm_verdict is None:
        return False
    return llm_verdict.category in {Category.MISSING_CONTENT, Category.FORCED_MISMATCH}


def _create_long_segment_issue(
    long_info: LongSegmentInfo,
    issue_id: str,
) -> Issue:
    """Создать Issue из информации о длинном сегменте."""
    seg = long_info.segment
    duration_sec = seg.duration_ms / 1000.0

    ref_texts = [s.text for s in long_info.ref_segments]
    ref_text_combined = " ".join(ref_texts) if ref_texts else ""

    if long_info.split_suggestions:
        suggestions_text = []
        for j, sugg in enumerate(long_info.split_suggestions, 1):
            from subs_diff.types import ms_to_srt
            split_time_str = ms_to_srt(sugg.split_time_ms)
            suggestions_text.append(
                f"  {j}. Время: {split_time_str}, уверенность: {sugg.confidence:.0%}"
            )
        evidence = f"Предложения разделения:\n" + "\n".join(suggestions_text)
    else:
        evidence = "Точек разделения не найдено (нет подходящих сегментов в reference)"

    b_segments_indices = [s.index for s in long_info.ref_segments]

    return Issue(
        issue_id=issue_id,
        time_range=(seg.start_ms, seg.end_ms),
        a_segments=[seg.index],
        b_segments=b_segments_indices,
        severity=Severity.MED,
        category=Category.LONG_SEGMENT,
        short_reason=f"Длинный сегмент: {duration_sec:.1f} сек",
        evidence=evidence,
        forced_detected=False,
        llm_verdict=None,
        a_text=seg.text,
        b_text=ref_text_combined,
    )


def _build_ref_only_candidates(
    aligned_pairs: list[tuple[MergedSegment, MergedSegment]],
    extra_candidates: list[Candidate],
) -> list[Candidate]:
    """Расширить список проверок: все aligned пары + уже найденные кандидаты."""
    candidates: list[Candidate] = list(extra_candidates)
    seen = {
        (tuple(c.a_segment.indices), tuple(c.b_segment.indices), c.is_forced_like)
        for c in candidates
    }

    for merged_a, merged_b in aligned_pairs:
        metrics = (
            compute_similarity(merged_a, merged_b)
            if merged_a.text and merged_b.text
            else SimilarityMetrics(
                jaccard=0.0,
                char_3gram=0.0,
                levenshtein=0.0,
                length_ratio=0.0,
                rare_token_overlap=0.0,
                rare_token_missing=0.0,
            )
        )
        cand = Candidate(
            a_segment=merged_a,
            b_segment=merged_b,
            metrics=metrics,
            is_forced_like=False,
        )
        key = (tuple(cand.a_segment.indices), tuple(cand.b_segment.indices), cand.is_forced_like)
        if key not in seen:
            candidates.append(cand)
            seen.add(key)

    return candidates


def create_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="subs-diff",
        description="Сравнение двух файлов субтитров .srt для выявления ошибок STT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --stt audio.srt --ref translated.srt --out report.json
  %(prog)s --stt A.srt --ref B.srt --out report.json --llm off
  %(prog)s --stt A.srt --ref B.srt --out report.json --llm api --llm-key $OPENROUTER_API_KEY

Управление конфигурацией:
  %(prog)s config set --api-key sk-... --api-model meta-llama/llama-3.2-3b-instruct
  %(prog)s config show
  %(prog)s config clear

Для пересборки HTML из готового JSON:
  python -m subs_diff.reporter report.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # Команда compare (основная)
    compare_parser = subparsers.add_parser(
        "compare",
        help="Сравнить два SRT файла",
        description="Сравнение двух файлов субтитров",
    )
    compare_parser.add_argument(
        "--stt",
        required=True,
        type=str,
        help="Путь к SRT файлу от STT (Whisper и т.д.)",
    )
    compare_parser.add_argument(
        "--ref",
        required=True,
        type=str,
        help="Путь к reference SRT файлу (переведённый/эталонный)",
    )
    compare_parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Путь для выходного JSON отчёта",
    )

    # Опции выравнивания
    compare_parser.add_argument(
        "--time-tol",
        type=float,
        default=1.0,
        help="Допуск по времени в секундах (по умолчанию: 1.0)",
    )
    compare_parser.add_argument(
        "--max-merge",
        type=int,
        default=5,
        help="Макс. соседних сегментов для склейки (по умолчанию: 5)",
    )
    compare_parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Минимальный порог сходства для кандидата (по умолчанию: 0.5)",
    )

    # LLM опции
    compare_parser.add_argument(
        "--llm",
        type=str,
        choices=["off", "auto", "local", "api"],
        default="auto",
        help="Режим LLM: off=выключена, auto=автовыбор, local=Ollama, api=OpenRouter (по умолчанию: auto)",
    )
    compare_parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Модель LLM для api режима (по умолчанию: из конфига)",
    )
    compare_parser.add_argument(
        "--llm-local-model",
        type=str,
        default=None,
        help="Модель LLM для local режима (по умолчанию: из конфига)",
    )
    compare_parser.add_argument(
        "--llm-url",
        type=str,
        default=None,
        help="URL для local LLM (по умолчанию: из конфига)",
    )
    compare_parser.add_argument(
        "--llm-key",
        type=str,
        default=None,
        help="API ключ для api режима (переопределяет конфиг)",
    )
    compare_parser.add_argument(
        "--llm-api-url",
        type=str,
        default=None,
        help="URL для API LLM (по умолчанию: из конфига)",
    )
    compare_parser.add_argument(
        "--llm-site-url",
        type=str,
        default=None,
        help="URL сайта для OpenRouter headers",
    )
    compare_parser.add_argument(
        "--llm-site-name",
        type=str,
        default=None,
        help="Название сайта для OpenRouter headers",
    )

    # Вывод
    compare_parser.add_argument(
        "--no-html",
        action="store_true",
        help="Не генерировать HTML отчёт",
    )
    compare_parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Включить периодическое сохранение промежуточных результатов",
    )
    compare_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Интервал сохранения чекпоинтов (кол-во кандидатов)",
    )
    compare_parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить с partial checkpoint из --out",
    )
    compare_parser.add_argument(
        "--llm-context-size",
        type=int,
        default=1,
        help="Количество сегментов контекста для LLM (prev+next)",
    )
    compare_parser.add_argument(
        "--ref-only-missing",
        action="store_true",
        help="Режим: искать только то, что есть в REF, но отсутствует в STT",
    )
    compare_parser.add_argument(
        "--llm-debug",
        action="store_true",
        help="Сохранять запросы и ответы LLM в debug-лог",
    )
    compare_parser.add_argument(
        "--llm-debug-file",
        type=str,
        default="llm_debug.log",
        help="Путь к человекочитаемому лог-файлу LLM (по умолчанию: llm_debug.log)",
    )
    compare_parser.add_argument(
        "--check-long-segments",
        action="store_true",
        help="Проверять сегменты с длительностью больше порога",
    )
    compare_parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=6.0,
        help="Макс. длительность сегмента в секундах (по умолчанию: 6.0)",
    )
    compare_parser.add_argument(
        "--full-context",
        action="store_true",
        help="Режим полного контекста: загрузить все субтитры в LLM (для моделей 1M+ токенов)",
    )
    compare_parser.add_argument(
        "--full-context-max-pairs",
        type=int,
        default=50,
        help="Макс. пар сегментов в одном batch-запросе (по умолчанию: 50)",
    )
    compare_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Включить подробный вывод",
    )

    # Команда config
    config_parser = subparsers.add_parser(
        "config",
        help="Управление конфигурацией",
        description="Управление конфигурационным файлом",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Подкоманды конфига")

    # config set
    set_parser = config_subparsers.add_parser(
        "set",
        help="Сохранить настройки в конфиг",
        description="Сохранить LLM настройки в конфигурационный файл",
    )
    set_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API ключ для OpenRouter",
    )
    set_parser.add_argument(
        "--api-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="URL API (по умолчанию: https://openrouter.ai/api/v1)",
    )
    set_parser.add_argument(
        "--api-model",
        type=str,
        default="meta-llama/llama-3.2-3b-instruct",
        help="Модель API (по умолчанию: meta-llama/llama-3.2-3b-instruct)",
    )
    set_parser.add_argument(
        "--local-url",
        type=str,
        default="http://localhost:11434",
        help="URL локальной LLM (по умолчанию: http://localhost:11434)",
    )
    set_parser.add_argument(
        "--local-model",
        type=str,
        default="llama3.2",
        help="Модель локальной LLM (по умолчанию: llama3.2)",
    )
    set_parser.add_argument(
        "--site-url",
        type=str,
        default="https://github.com/eXpressionist/SubDiffLLM",
        help="URL сайта для OpenRouter",
    )
    set_parser.add_argument(
        "--site-name",
        type=str,
        default="SubDiff",
        help="Название сайта для OpenRouter",
    )

    # config show
    config_subparsers.add_parser(
        "show",
        help="Показать текущую конфигурацию",
        description="Показать содержимое конфигурационного файла",
    )

    # config clear
    config_subparsers.add_parser(
        "clear",
        help="Очистить конфигурацию",
        description="Удалить конфигурационный файл",
    )

    # Для обратной совместимости - аргументы основного парсера
    # (когда команда не указана, но есть аргументы)
    parser.add_argument(
        "--stt",
        type=str,
        help="Путь к SRT файлу от STT (Whisper и т.д.)",
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="Путь к reference SRT файлу (переведённый/эталонный)",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Путь для выходного JSON отчёта",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,  # Скрываем из help основного парсера
    )
    parser.add_argument(
        "--max-merge",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["off", "auto", "local", "api"],
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-local-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-api-url",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-site-url",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-site-name",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-context-size",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ref-only-missing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-debug",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--llm-debug-file",
        type=str,
        default="llm_debug.log",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check-long-segments",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=6.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--full-context-max-pairs",
        type=int,
        default=50,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


async def run_comparison(config: Config) -> int:
    """
    Выполнить сравнение файлов.

    Args:
        config: Конфигурация.

    Returns:
        Код выхода (0 = успех).
    """
    logger.info(f"Загрузка STT файла: {config.stt_file}")
    try:
        segments_a = parse_srt_file(config.stt_file)
        logger.info(f"  Загружено {len(segments_a)} сегментов из A")
    except Exception as e:
        logger.error(f"Ошибка чтения STT файла: {e}")
        return 1

    logger.info(f"Загрузка reference файла: {config.ref_file}")
    try:
        segments_b = parse_srt_file(config.ref_file)
        logger.info(f"  Загружено {len(segments_b)} сегментов из B")
    except Exception as e:
        logger.error(f"Ошибка чтения reference файла: {e}")
        return 1

    # Детектируем forced-сегменты
    forced_a = set(detect_forced_segments(segments_a))
    forced_b = set(detect_forced_segments(segments_b))
    if forced_a or forced_b:
        logger.info(
            f"  Обнаружено forced-сегментов: A={len(forced_a)}, B={len(forced_b)}"
        )

    # Выравнивание и поиск кандидатов
    logger.info("Выравнивание сегментов и поиск кандидатов...")
    result = align_segments(
        segments_a,
        segments_b,
        time_tol=config.time_tol,
        max_merge=config.max_merge,
        min_score=config.min_score,
    )
    logger.info(f"  Найдено {len(result.candidates)} кандидатов")

    # Проверка длинных сегментов
    long_segment_issues: list[Issue] = []
    if getattr(config, "check_long_segments", False):
        max_duration_ms = int(getattr(config, "max_segment_duration", 6.0) * 1000)
        logger.info(f"Проверка длинных сегментов (>{config.max_segment_duration:.1f} сек)...")
        long_segments = analyze_long_segments(
            segments_a,
            segments_b,
            max_duration_ms=max_duration_ms,
            time_tol_ms=int(config.time_tol * 1000),
        )
        logger.info(f"  Найдено {len(long_segments)} длинных сегментов")

        for i, long_info in enumerate(long_segments):
            issue_id = f"LONG-{i + 1:03d}"
            issue = _create_long_segment_issue(long_info, issue_id)
            long_segment_issues.append(issue)

    # LLM верификация
    llm_client = create_llm_client(
        mode=config.llm_mode,
        local_url=config.llm_url,
        local_model=config.llm_local_model,
        api_url=config.llm_api_url,
        api_key=config.llm_api_key,
        api_model=config.llm_model,
        site_url=config.llm_site_url,
        site_name=config.llm_site_name,
        debug_enabled=config.llm_debug,
        debug_file=config.llm_debug_file,
    )

    llm_available = await llm_client.is_available()
    if config.llm_mode != LLMMode.OFF:
        if llm_available:
            logger.info("LLM доступна, выполняем верификацию кандидатов...")
        else:
            logger.warning("LLM недоступна, используем только эвристики")

    # Устанавливаем обработчик прерывания
    signal.signal(signal.SIGINT, _handle_interrupt)

    # Верифицируем кандидатов.
    # В ref-only режиме расширяем охват до всех выровненных пар.
    issues: list[Issue] = []
    llm_context_size = getattr(config, 'llm_context_size', 1)  # Сколько сегментов контекста
    all_candidates = (
        _build_ref_only_candidates(result.aligned_pairs, result.candidates)
        if getattr(config, "ref_only_missing", False)
        else result.candidates
    )
    total_to_verify = len(all_candidates)
    
    # Full-context режим: batch-верификация
    full_context_enabled = getattr(config, "full_context", False)
    if config.verbose or full_context_enabled:
        logger.info(f"Full-context mode enabled: {full_context_enabled}")
        logger.info(f"LLM available: {llm_available}")
        logger.info(f"LLM mode: {config.llm_mode}")
    
    if full_context_enabled and llm_available and config.llm_mode != LLMMode.OFF:
        logger.info(f"Full-context режим: обработка {total_to_verify} пар batch'ами по {getattr(config, 'full_context_max_pairs', 50)}...")
        max_pairs_per_batch = getattr(config, "full_context_max_pairs", 50)
        all_verdicts: list[Optional[LLMVerdict]] = [None] * total_to_verify
        
        for batch_start in range(0, total_to_verify, max_pairs_per_batch):
            batch_end = min(batch_start + max_pairs_per_batch, total_to_verify)
            batch_candidates = all_candidates[batch_start:batch_end]
            
            pairs_for_batch = []
            for candidate in batch_candidates:
                prev_segment = None
                next_segment = None
                if candidate.a_segment.indices:
                    idx = candidate.a_segment.indices[0]
                    if idx > 0:
                        prev_segment = merge_segments([segments_a[idx - 1]])
                    if idx + len(candidate.a_segment.indices) < len(segments_a):
                        next_segment = merge_segments([segments_a[idx + len(candidate.a_segment.indices)]])
                pairs_for_batch.append((candidate, prev_segment, next_segment))
            
            logger.info(f"  Обработка batch {batch_start+1}-{batch_end} из {total_to_verify}...")
            
            batch_verdicts = await llm_client.verify_batch(pairs_for_batch)
            
            for i, (verdict, candidate) in enumerate(zip(batch_verdicts, batch_candidates)):
                all_verdicts[batch_start + i] = verdict
            
            _save_checkpoint(
                [
                    create_issue_from_candidate(all_candidates[j], f"ISS-{j+1:03d}", v)
                    for j, v in enumerate(all_verdicts[:batch_end])
                    if v or all_candidates[j].is_forced_like
                ],
                Path(config.out_file),
                batch_end,
                total_to_verify,
            )
        
        for i, (verdict, candidate) in enumerate(zip(all_verdicts, all_candidates)):
            if candidate.is_forced_like:
                issues.append(create_issue_from_candidate(candidate, f"ISS-{i+1:03d}", None))
            elif verdict and not _should_skip_trivial_missing_content(candidate, verdict):
                issues.append(create_issue_from_candidate(candidate, f"ISS-{i+1:03d}", verdict))
        
        issues.sort(key=lambda x: x.time_range[0])
        logger.info(f"  Full-context: найдено {len(issues)} проблем")
        
        # Добавляем длинные сегменты в отчёт
        if long_segment_issues:
            issues.extend(long_segment_issues)
            issues.sort(key=lambda x: x.time_range[0])
            logger.info(f"  Добавлено {len(long_segment_issues)} длинных сегментов")
        
        # Генерируем отчёт (для full-context режима)
        logger.info(f"  Итого проблем: {len(issues)}")
        
        # Генерируем отчёт
        logger.info("Генерация отчёта...")
        
        report_config = {
            "time_tol": config.time_tol,
            "max_merge": config.max_merge,
            "min_score": config.min_score,
            "llm_mode": config.llm_mode.value,
            "llm_model": config.llm_model,
            "ref_only_missing": getattr(config, "ref_only_missing", False),
        }
        
        report = generate_report(
            issues=issues,
            stt_file=str(config.stt_file),
            ref_file=str(config.ref_file),
            config=report_config,
        )
        
        # Сохраняем JSON
        out_path = Path(config.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_report_json(report, out_path)
        logger.info(f"  JSON сохранён: {out_path}")
        
        # Сохраняем HTML
        if not config.no_html:
            html_path = out_path.with_suffix(".html")
            save_report_html(report, html_path)
            logger.info(f"  HTML сохранён: {html_path}")
        
        # Выводим сводку
        print_summary(report)
        
        return 0  # Завершаем полную обработку в full-context режиме
    
    else:
        # Обычный режим: поочерёдная верификация
        logger.info("Обычный режим: поочерёдная верификация кандидатов...")
    
    # Общие переменные для обоих режимов
    checkpoint_interval = getattr(config, 'checkpoint_interval', 5)
    use_checkpoint = getattr(config, 'checkpoint', False) or total_to_verify > 20
    resume_from = 0

    if getattr(config, "resume", False):
        out_path = Path(config.out_file)
        try:
            loaded_issues, processed = _load_resume_checkpoint(out_path)
            if processed > 0:
                issues = loaded_issues
                resume_from = min(processed, total_to_verify)
                logger.info(
                    f"Resume: загружен чекпоинт {resume_from}/{total_to_verify}, "
                    f"issues={len(issues)}"
                )
            else:
                logger.warning(
                    "Флаг --resume указан, но partial checkpoint не найден. Запускаем с начала."
                )
        except Exception as e:
            logger.warning(f"Не удалось загрузить checkpoint для resume: {e}")

    if resume_from >= total_to_verify and total_to_verify > 0:
        logger.info("Все кандидаты уже обработаны по checkpoint, пересобираем финальный отчёт...")

    _interrupt_state["issues"] = issues
    _interrupt_state["total"] = total_to_verify
    _interrupt_state["processed"] = resume_from
    _interrupt_state["out_file"] = Path(config.out_file)

    with tqdm(total=total_to_verify, desc="LLM верификация", unit="пар", initial=resume_from) as pbar:
        for i, candidate in enumerate(all_candidates[resume_from:], start=resume_from):
            issue_id = f"ISS-{i + 1:03d}"

            # Для forced сегментов создаём issue сразу
            if candidate.is_forced_like:
                issue = create_issue_from_candidate(candidate, issue_id, llm_verdict=None)
                issues.append(issue)
                pbar.update(1)
                continue

            # Находим контекст (предыдущий и следующий сегменты A)
            prev_segment = None
            next_segment = None
            if llm_context_size > 0 and candidate.a_segment.indices:
                idx = candidate.a_segment.indices[0]
                if idx > 0:
                    # Предыдущий сегмент
                    prev_segment = merge_segments([segments_a[idx - 1]])
                if idx + len(candidate.a_segment.indices) < len(segments_a):
                    # Следующий сегмент
                    next_segment = merge_segments([segments_a[idx + len(candidate.a_segment.indices)]])

            llm_verdict = None
            if llm_available and config.llm_mode != LLMMode.OFF:
                try:
                    llm_verdict = await llm_client.verify(
                        candidate,
                        prev_segment=prev_segment,
                        next_segment=next_segment,
                    )
                except Exception as e:
                    logger.warning(f"Ошибка верификации пары {i+1}: {e}")

            # Создаём issue только если есть подтверждённый вердикт
            if llm_verdict:
                if _should_skip_trivial_missing_content(candidate, llm_verdict):
                    if config.verbose:
                        logger.info(
                            "Пропускаем trivial missing_content (служебные слова) "
                            f"для пары {i+1}"
                        )
                    llm_verdict = None

            if llm_verdict:
                if getattr(config, "ref_only_missing", False) and not _should_keep_issue_in_ref_only_mode(
                    llm_verdict,
                    candidate.is_forced_like,
                ):
                    llm_verdict = None

            if llm_verdict:
                issue = create_issue_from_candidate(candidate, issue_id, llm_verdict)
                issues.append(issue)

            # Обновляем состояние для чекпоинта
            _interrupt_state["issues"] = issues
            _interrupt_state["total"] = total_to_verify
            _interrupt_state["processed"] = i + 1
            _interrupt_state["out_file"] = Path(config.out_file)

            pbar.update(1)

            # Периодическое сохранение чекпоинта
            if use_checkpoint and (i + 1) % checkpoint_interval == 0:
                _save_checkpoint(
                    issues,
                    Path(config.out_file),
                    i + 1,
                    total_to_verify,
                )
                if config.verbose:
                    logger.info(f"Чекпоинт: {i+1}/{total_to_verify}")

    # Сортируем по времени
    issues.sort(key=lambda x: x.time_range[0])

    # Добавляем длинные сегменты в отчёт
    if long_segment_issues:
        issues.extend(long_segment_issues)
        issues.sort(key=lambda x: x.time_range[0])
        logger.info(f"  Добавлено {len(long_segment_issues)} длинных сегментов")

    logger.info(f"  Итого проблем: {len(issues)}")

    # Генерируем отчёт
    logger.info("Генерация отчёта...")

    report_config = {
        "time_tol": config.time_tol,
        "max_merge": config.max_merge,
        "min_score": config.min_score,
        "llm_mode": config.llm_mode.value,
        "llm_model": config.llm_model,
        "ref_only_missing": getattr(config, "ref_only_missing", False),
    }

    report = generate_report(
        issues=issues,
        stt_file=str(config.stt_file),
        ref_file=str(config.ref_file),
        config=report_config,
    )

    # Сохраняем JSON
    out_path = Path(config.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_report_json(report, out_path)
    logger.info(f"  JSON сохранён: {out_path}")

    # Сохраняем HTML
    if not config.no_html:
        html_path = out_path.with_suffix(".html")
        save_report_html(report, html_path)
        logger.info(f"  HTML сохранён: {html_path}")

    # Выводим сводку
    print_summary(report)

    return 0


def print_summary(report) -> None:
    """Вывести сводку по отчёту."""
    print("\n" + "=" * 50)
    print("СВОДКА ПО ОТЧЁТУ")
    print("=" * 50)
    print(f"Всего проблем: {report.summary.total_issues}")
    print(f"  Высокая важность: {report.summary.by_severity.get('high', 0)}")
    print(f"  Средняя важность: {report.summary.by_severity.get('med', 0)}")
    print(f"  Низкая важность: {report.summary.by_severity.get('low', 0)}")
    print("\nПо категориям:")
    for cat, count in report.summary.by_category.items():
        print(f"  {cat}: {count}")
    print("=" * 50)


def main() -> int:
    """Точка входа CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Обработка подкоманд
    if args.command == "config":
        if hasattr(args, 'verbose') and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        return handle_config_command(args)
    elif args.command == "compare" or args.command is None:
        # Если команда не указана, но есть аргументы compare - запускаем сравнение
        if not args.stt or not args.ref or not args.out:
            parser.print_help()
            return 1
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        return run_compare(args)
    else:
        parser.print_help()
        return 1


def handle_config_command(args) -> int:
    """Обработать команду config."""
    if args.config_command == "set":
        # Сохраняем конфиг
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

        # Показываем что сохранено
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

    elif args.config_command == "show":
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
                key_preview = llm["api_key"][:8] + "..." if len(llm["api_key"]) > 8 else llm["api_key"]
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

    elif args.config_command == "clear":
        config_path = get_config_path()
        if config_path.exists():
            config_path.unlink()
            print(f"Конфигурация удалена: {config_path}")
        else:
            print("Конфигурация уже пуста")
        return 0

    else:
        print("Используйте: subs-diff config {set|show|clear}")
        return 1


def run_compare(args) -> int:
    """Запустить сравнение файлов."""
    # Загружаем конфиг из файла
    saved_config = load_config()
    llm_config = saved_config.get("llm", {})

    # Получаем API ключ: аргумент > конфиг > окружение
    llm_api_key = args.llm_key
    if not llm_api_key:
        llm_api_key = get_api_key()

    # Получаем значения из конфига, если не указаны в аргументах
    llm_model = args.llm_model or llm_config.get("api_model", "meta-llama/llama-3.2-3b-instruct")
    llm_local_model = args.llm_local_model or llm_config.get("local_model", "llama3.2")
    llm_url = args.llm_url or llm_config.get("local_url", "http://localhost:11434")
    llm_api_url = args.llm_api_url or llm_config.get("api_url", "https://openrouter.ai/api/v1")
    llm_site_url = args.llm_site_url or llm_config.get("site_url", "https://github.com/eXpressionist/SubDiffLLM")
    llm_site_name = args.llm_site_name or llm_config.get("site_name", "SubDiff")

    # Создаём конфигурацию
    config = Config(
        stt_file=args.stt,
        ref_file=args.ref,
        out_file=args.out,
        time_tol=args.time_tol,
        max_merge=args.max_merge,
        min_score=args.min_score,
        llm_mode=LLMMode(args.llm),
        llm_model=llm_model,
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

    # Запускаем асинхронно
    return asyncio.run(run_comparison(config))


if __name__ == "__main__":
    sys.exit(main())
