"""Comparison pipeline orchestration for SubDiff."""

import logging
import re
import signal
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from subs_diff.align import align_segments, merge_segments
from subs_diff.checkpoint import (
    load_resume_checkpoint as _load_resume_checkpoint,
)
from subs_diff.checkpoint import (
    save_checkpoint as _save_checkpoint,
)
from subs_diff.heuristics import compute_similarity
from subs_diff.llm import create_llm_client
from subs_diff.parser import detect_forced_segments, parse_srt_file
from subs_diff.report import (
    create_issue_from_candidate,
    generate_report,
    save_report_html,
    save_report_json,
)
from subs_diff.segments import analyze_long_segments
from subs_diff.types import (
    Candidate,
    Category,
    Config,
    Issue,
    LLMMode,
    LLMVerdict,
    LongSegmentInfo,
    MergedSegment,
    Report,
    Segment,
    Severity,
    SimilarityMetrics,
    ms_to_srt,
)

logger = logging.getLogger(__name__)

_interrupt_state: dict[str, Any] = {
    "issues": [],
    "total": 0,
    "processed": 0,
    "out_file": None,
}

TRIVIAL_MISSING_TOKENS = {
    "а",
    "и",
    "но",
    "да",
    "же",
    "вот",
    "ну",
    "ли",
    "то",
    "ведь",
    "типа",
    "в",
    "во",
    "на",
    "к",
    "ко",
    "у",
    "с",
    "со",
    "из",
    "от",
    "до",
    "по",
    "за",
    "про",
    "для",
    "при",
    "о",
    "об",
    "обо",
    "под",
    "над",
    "через",
    "между",
    "бы",
    "б",
    "просто",
    "как",
    "так",
    "это",
}


def _handle_interrupt(signum: int, frame: Any) -> None:
    """Handle Ctrl+C by writing the current partial checkpoint."""
    logging.warning("\nПрерывание пользователем...")
    if _interrupt_state["issues"] and _interrupt_state["out_file"]:
        _save_checkpoint(
            _interrupt_state["issues"],
            _interrupt_state["out_file"],
            _interrupt_state["processed"],
            _interrupt_state["total"],
        )
        logging.info(
            "Частичные результаты сохранены в %s",
            _interrupt_state["out_file"],
        )
    sys.exit(1)


def _should_skip_trivial_missing_content(
    candidate: Candidate,
    llm_verdict: LLMVerdict | None,
) -> bool:
    """Skip missing_content issues that only concern small function words."""
    if llm_verdict is None or llm_verdict.category != Category.MISSING_CONTENT:
        return False

    reason = llm_verdict.short_reason.lower()
    quoted_missing = re.search(r"пропущено\s+'([^']+)'", reason)
    if quoted_missing and quoted_missing.group(1).strip().lower() in TRIVIAL_MISSING_TOKENS:
        return True

    topic_missing = re.search(r"пропущен\s+вопрос\s+о\s+([^\s.,!?]+)", reason)
    if topic_missing:
        focus = topic_missing.group(1).strip().lower()
        if len(focus) >= 3:
            root_len = 2 if len(focus) <= 4 else 3
            root = focus[:root_len]
            if (
                root in candidate.a_segment.text.lower()
                and root in candidate.b_segment.text.lower()
            ):
                return True

    a_vocab = {token.lower() for token in candidate.a_segment.tokens}
    missing_tokens = [
        token.lower() for token in candidate.b_segment.tokens if token.lower() not in a_vocab
    ]
    if not missing_tokens:
        return True

    meaningful = [token for token in missing_tokens if token not in TRIVIAL_MISSING_TOKENS]
    return not meaningful


def _should_keep_issue_in_ref_only_mode(
    llm_verdict: LLMVerdict | None,
    is_forced_like: bool,
) -> bool:
    """In ref-only mode keep only missing_content and forced_mismatch issues."""
    if is_forced_like:
        return True
    if llm_verdict is None:
        return False
    return llm_verdict.category in {Category.MISSING_CONTENT, Category.FORCED_MISMATCH}


def _create_long_segment_issue(
    long_info: LongSegmentInfo,
    issue_id: str,
) -> Issue:
    """Create a report issue for an overlong STT segment."""
    seg = long_info.segment
    duration_sec = seg.duration_ms / 1000.0
    ref_text_combined = " ".join(s.text for s in long_info.ref_segments)

    if long_info.split_suggestions:
        suggestions_text = []
        for idx, suggestion in enumerate(long_info.split_suggestions, 1):
            split_time_str = ms_to_srt(suggestion.split_time_ms)
            suggestions_text.append(
                f"  {idx}. Время: {split_time_str}, " f"уверенность: {suggestion.confidence:.0%}"
            )
        evidence = "Предложения разделения:\n" + "\n".join(suggestions_text)
    else:
        evidence = "Точек разделения не найдено " "(нет подходящих сегментов в reference)"

    return Issue(
        issue_id=issue_id,
        time_range=(seg.start_ms, seg.end_ms),
        a_segments=[seg.index],
        b_segments=[s.index for s in long_info.ref_segments],
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
    """Check all aligned pairs in ref-only mode, plus original candidates."""
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
        candidate = Candidate(
            a_segment=merged_a,
            b_segment=merged_b,
            metrics=metrics,
            is_forced_like=False,
        )
        key = (
            tuple(candidate.a_segment.indices),
            tuple(candidate.b_segment.indices),
            candidate.is_forced_like,
        )
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)

    return candidates


def _candidate_context(
    candidate: Candidate,
    segments_a: list[Segment],
    llm_context_size: int,
) -> tuple[MergedSegment | None, MergedSegment | None]:
    """Return previous and next A-side context segments for a candidate."""
    if llm_context_size <= 0 or not candidate.a_segment.indices:
        return None, None

    idx = candidate.a_segment.indices[0]
    prev_segment = merge_segments([segments_a[idx - 1]]) if idx > 0 else None

    next_idx = idx + len(candidate.a_segment.indices)
    next_segment = merge_segments([segments_a[next_idx]]) if next_idx < len(segments_a) else None
    return prev_segment, next_segment


def _report_config(config: Config) -> dict[str, object]:
    return {
        "time_tol": config.time_tol,
        "max_merge": config.max_merge,
        "min_score": config.min_score,
        "llm_mode": config.llm_mode.value,
        "llm_model": config.llm_model,
        "ref_only_missing": getattr(config, "ref_only_missing", False),
    }


def _save_final_report(config: Config, issues: list[Issue]) -> None:
    logger.info("Генерация отчёта...")
    report = generate_report(
        issues=issues,
        stt_file=str(config.stt_file),
        ref_file=str(config.ref_file),
        config=_report_config(config),
    )

    out_path = Path(config.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_report_json(report, out_path)
    logger.info("  JSON сохранён: %s", out_path)

    if not config.no_html:
        html_path = out_path.with_suffix(".html")
        save_report_html(report, html_path)
        logger.info("  HTML сохранён: %s", html_path)

    print_summary(report)


async def run_comparison(config: Config) -> int:
    """Run the comparison pipeline."""
    logger.info("Загрузка STT файла: %s", config.stt_file)
    try:
        segments_a = parse_srt_file(config.stt_file)
        logger.info("  Загружено %s сегментов из A", len(segments_a))
    except Exception as exc:
        logger.error("Ошибка чтения STT файла: %s", exc)
        return 1

    logger.info("Загрузка reference файла: %s", config.ref_file)
    try:
        segments_b = parse_srt_file(config.ref_file)
        logger.info("  Загружено %s сегментов из B", len(segments_b))
    except Exception as exc:
        logger.error("Ошибка чтения reference файла: %s", exc)
        return 1

    forced_a = set(detect_forced_segments(segments_a))
    forced_b = set(detect_forced_segments(segments_b))
    if forced_a or forced_b:
        logger.info(
            "  Обнаружено forced-сегментов: A=%s, B=%s",
            len(forced_a),
            len(forced_b),
        )

    logger.info("Выравнивание сегментов и поиск кандидатов...")
    result = align_segments(
        segments_a,
        segments_b,
        time_tol=config.time_tol,
        max_merge=config.max_merge,
        min_score=config.min_score,
    )
    logger.info("  Найдено %s кандидатов", len(result.candidates))

    long_segment_issues: list[Issue] = []
    if getattr(config, "check_long_segments", False):
        max_duration_ms = int(getattr(config, "max_segment_duration", 6.0) * 1000)
        logger.info(
            "Проверка длинных сегментов (>%.1f сек)...",
            config.max_segment_duration,
        )
        long_segments = analyze_long_segments(
            segments_a,
            segments_b,
            max_duration_ms=max_duration_ms,
            time_tol_ms=int(config.time_tol * 1000),
        )
        logger.info("  Найдено %s длинных сегментов", len(long_segments))
        for idx, long_info in enumerate(long_segments):
            long_segment_issues.append(_create_long_segment_issue(long_info, f"LONG-{idx + 1:03d}"))

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
            logger.info("LLM доступна, " "выполняем верификацию кандидатов...")
        else:
            logger.warning("LLM недоступна, " "используем только эвристики")

    signal.signal(signal.SIGINT, _handle_interrupt)

    issues: list[Issue] = []
    llm_context_size = getattr(config, "llm_context_size", 1)
    all_candidates = (
        _build_ref_only_candidates(result.aligned_pairs, result.candidates)
        if getattr(config, "ref_only_missing", False)
        else result.candidates
    )
    total_to_verify = len(all_candidates)

    full_context_enabled = getattr(config, "full_context", False)
    if config.verbose or full_context_enabled:
        logger.info("Full-context mode enabled: %s", full_context_enabled)
        logger.info("LLM available: %s", llm_available)
        logger.info("LLM mode: %s", config.llm_mode)

    if full_context_enabled and llm_available and config.llm_mode != LLMMode.OFF:
        logger.info(
            "Full-context режим: обработка %s пар batch'ами по %s...",
            total_to_verify,
            getattr(config, "full_context_max_pairs", 50),
        )
        max_pairs_per_batch = getattr(config, "full_context_max_pairs", 50)
        all_verdicts: list[LLMVerdict | None] = [None] * total_to_verify

        for batch_start in range(0, total_to_verify, max_pairs_per_batch):
            batch_end = min(batch_start + max_pairs_per_batch, total_to_verify)
            batch_candidates = all_candidates[batch_start:batch_end]
            pairs_for_batch = [
                (
                    candidate,
                    *_candidate_context(candidate, segments_a, llm_context_size),
                )
                for candidate in batch_candidates
            ]

            logger.info(
                "  Обработка batch %s-%s из %s...",
                batch_start + 1,
                batch_end,
                total_to_verify,
            )
            batch_verdicts = await llm_client.verify_batch(pairs_for_batch)

            for idx, verdict in enumerate(batch_verdicts):
                all_verdicts[batch_start + idx] = verdict

            _save_checkpoint(
                [
                    create_issue_from_candidate(all_candidates[j], f"ISS-{j + 1:03d}", verdict)
                    for j, verdict in enumerate(all_verdicts[:batch_end])
                    if verdict or all_candidates[j].is_forced_like
                ],
                Path(config.out_file),
                batch_end,
                total_to_verify,
            )

        for idx, (verdict, candidate) in enumerate(zip(all_verdicts, all_candidates, strict=False)):
            if candidate.is_forced_like:
                issues.append(create_issue_from_candidate(candidate, f"ISS-{idx + 1:03d}", None))
            elif verdict and not _should_skip_trivial_missing_content(candidate, verdict):
                issues.append(create_issue_from_candidate(candidate, f"ISS-{idx + 1:03d}", verdict))

        issues.sort(key=lambda issue: issue.time_range[0])
        logger.info("  Full-context: найдено %s проблем", len(issues))

        if long_segment_issues:
            issues.extend(long_segment_issues)
            issues.sort(key=lambda issue: issue.time_range[0])
            logger.info(
                "  Добавлено %s длинных сегментов",
                len(long_segment_issues),
            )

        logger.info("  Итого проблем: %s", len(issues))
        _save_final_report(config, issues)
        return 0

    logger.info("Обычный режим: " "поочерёдная верификация кандидатов...")

    checkpoint_interval = getattr(config, "checkpoint_interval", 5)
    use_checkpoint = getattr(config, "checkpoint", False) or total_to_verify > 20
    resume_from = 0

    if getattr(config, "resume", False):
        out_path = Path(config.out_file)
        try:
            loaded_issues, processed = _load_resume_checkpoint(out_path)
            if processed > 0:
                issues = loaded_issues
                resume_from = min(processed, total_to_verify)
                logger.info(
                    "Resume: загружен чекпоинт %s/%s, issues=%s",
                    resume_from,
                    total_to_verify,
                    len(issues),
                )
            else:
                logger.warning(
                    "Флаг --resume указан, но partial checkpoint не найден. " "Запускаем с начала."
                )
        except Exception as exc:
            logger.warning(
                "Не удалось загрузить checkpoint для resume: %s",
                exc,
            )

    if resume_from >= total_to_verify and total_to_verify > 0:
        logger.info(
            "Все кандидаты уже обработаны по checkpoint, " "пересобираем финальный отчёт..."
        )

    _interrupt_state["issues"] = issues
    _interrupt_state["total"] = total_to_verify
    _interrupt_state["processed"] = resume_from
    _interrupt_state["out_file"] = Path(config.out_file)

    with tqdm(
        total=total_to_verify,
        desc="LLM верификация",
        unit="пар",
        initial=resume_from,
    ) as pbar:
        for idx, candidate in enumerate(all_candidates[resume_from:], start=resume_from):
            issue_id = f"ISS-{idx + 1:03d}"

            if candidate.is_forced_like:
                issues.append(create_issue_from_candidate(candidate, issue_id, llm_verdict=None))
                pbar.update(1)
                continue

            prev_segment, next_segment = _candidate_context(candidate, segments_a, llm_context_size)

            llm_verdict = None
            if llm_available and config.llm_mode != LLMMode.OFF:
                try:
                    llm_verdict = await llm_client.verify(
                        candidate,
                        prev_segment=prev_segment,
                        next_segment=next_segment,
                    )
                except Exception as exc:
                    logger.warning(
                        "Ошибка верификации пары %s: %s",
                        idx + 1,
                        exc,
                    )

            if llm_verdict and _should_skip_trivial_missing_content(candidate, llm_verdict):
                if config.verbose:
                    logger.info(
                        "Пропускаем trivial missing_content " "(служебные слова) для пары %s",
                        idx + 1,
                    )
                llm_verdict = None

            if (
                llm_verdict
                and getattr(config, "ref_only_missing", False)
                and not _should_keep_issue_in_ref_only_mode(llm_verdict, candidate.is_forced_like)
            ):
                llm_verdict = None

            if llm_verdict:
                issues.append(create_issue_from_candidate(candidate, issue_id, llm_verdict))

            _interrupt_state["issues"] = issues
            _interrupt_state["total"] = total_to_verify
            _interrupt_state["processed"] = idx + 1
            _interrupt_state["out_file"] = Path(config.out_file)

            pbar.update(1)

            if use_checkpoint and (idx + 1) % checkpoint_interval == 0:
                _save_checkpoint(issues, Path(config.out_file), idx + 1, total_to_verify)
                if config.verbose:
                    logger.info("Чекпоинт: %s/%s", idx + 1, total_to_verify)

    issues.sort(key=lambda issue: issue.time_range[0])

    if long_segment_issues:
        issues.extend(long_segment_issues)
        issues.sort(key=lambda issue: issue.time_range[0])
        logger.info(
            "  Добавлено %s длинных сегментов",
            len(long_segment_issues),
        )

    logger.info("  Итого проблем: %s", len(issues))
    _save_final_report(config, issues)
    return 0


def print_summary(report: Report) -> None:
    """Print a report summary."""
    print("\n" + "=" * 50)
    print("СВОДКА ПО ОТЧЁТУ")
    print("=" * 50)
    print(f"Всего проблем: {report.summary.total_issues}")
    print(f"  Высокая важность: {report.summary.by_severity.get('high', 0)}")
    print(f"  Средняя важность: {report.summary.by_severity.get('med', 0)}")
    print(f"  Низкая важность: {report.summary.by_severity.get('low', 0)}")
    print("\nПо категориям:")
    for category, count in report.summary.by_category.items():
        print(f"  {category}: {count}")
    print("=" * 50)
