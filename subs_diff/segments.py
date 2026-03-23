"""Функции для анализа и разделения длинных сегментов."""

import re
from typing import Optional

from subs_diff.types import (
    Segment,
    LongSegmentInfo,
    SplitSuggestion,
)


DEFAULT_MAX_DURATION_MS = 6000


def find_long_segments(
    segments: list[Segment],
    max_duration_ms: int = DEFAULT_MAX_DURATION_MS,
) -> list[Segment]:
    """
    Найти сегменты, длительность которых превышает порог.

    Args:
        segments: Список сегментов для проверки.
        max_duration_ms: Максимальная допустимая длительность в мс.

    Returns:
        Список длинных сегментов.
    """
    return [seg for seg in segments if seg.duration_ms > max_duration_ms]


def find_ref_segments_in_range(
    ref_segments: list[Segment],
    start_ms: int,
    end_ms: int,
    tolerance_ms: int = 500,
) -> list[Segment]:
    """
    Найти сегменты reference, попадающие во временной диапазон.

    Args:
        ref_segments: Сегменты reference файла.
        start_ms: Начало диапазона.
        end_ms: Конец диапазона.
        tolerance_ms: Допуск по времени.

    Returns:
        Список подходящих сегментов reference.
    """
    result = []
    for seg in ref_segments:
        if seg.end_ms < start_ms - tolerance_ms:
            continue
        if seg.start_ms > end_ms + tolerance_ms:
            continue
        result.append(seg)
    return result


def propose_split_points(
    stt_segment: Segment,
    ref_segments: list[Segment],
    min_split_duration_ms: int = 1500,
) -> list[SplitSuggestion]:
    """
    Предложить точки разделения длинного сегмента на основе reference.

    Алгоритм:
    1. Определяем границы сегментов reference в пределах длинного сегмента
    2. Предлагаем разделение по границам reference сегментов
    3. Проверяем, что обе части будут достаточно длинными

    Args:
        stt_segment: Длинный сегмент STT.
        ref_segments: Соответствующие сегменты reference.
        min_split_duration_ms: Минимальная длительность части после разделения.

    Returns:
        Список предложений разделения.
    """
    if not ref_segments:
        return []

    suggestions = []

    stt_start = stt_segment.start_ms
    stt_end = stt_segment.end_ms

    sorted_refs = sorted(ref_segments, key=lambda s: s.start_ms)

    for i, ref_seg in enumerate(sorted_refs):
        ref_start = ref_seg.start_ms
        ref_end = ref_seg.end_ms

        if ref_start <= stt_start:
            continue
        if ref_start >= stt_end:
            break

        part1_duration = ref_start - stt_start
        part2_duration = stt_end - ref_start

        if part1_duration < min_split_duration_ms:
            continue
        if part2_duration < min_split_duration_ms:
            continue

        text_before_parts = []
        text_after_parts = []

        for other_ref in sorted_refs:
            if other_ref.end_ms <= ref_start:
                text_before_parts.append(other_ref.text)
            elif other_ref.start_ms >= ref_start:
                text_after_parts.append(other_ref.text)

        ref_text_before = " ".join(text_before_parts) if text_before_parts else ""
        ref_text_after = " ".join(text_after_parts) if text_after_parts else ""

        overlap_before = 0
        overlap_after = 0

        for other_ref in sorted_refs:
            if other_ref.end_ms <= ref_start:
                overlap_before += min(other_ref.end_ms, ref_start) - max(other_ref.start_ms, stt_start)
            elif other_ref.start_ms >= ref_start:
                overlap_after += min(other_ref.end_ms, stt_end) - max(other_ref.start_ms, ref_start)

        total_overlap = overlap_before + overlap_after
        stt_duration = stt_segment.duration_ms
        confidence = min(1.0, total_overlap / stt_duration) if stt_duration > 0 else 0.0

        suggestions.append(
            SplitSuggestion(
                split_time_ms=ref_start,
                ref_text_before=ref_text_before,
                ref_text_after=ref_text_after,
                confidence=confidence,
            )
        )

    return suggestions


def analyze_long_segments(
    stt_segments: list[Segment],
    ref_segments: list[Segment],
    max_duration_ms: int = DEFAULT_MAX_DURATION_MS,
    time_tol_ms: int = 500,
) -> list[LongSegmentInfo]:
    """
    Проанализировать все длинные сегменты STT.

    Args:
        stt_segments: Сегменты STT файла.
        ref_segments: Сегменты reference файла.
        max_duration_ms: Порог длительности.
        time_tol_ms: Допуск по времени.

    Returns:
        Список информации о длинных сегментах.
    """
    long_segs = find_long_segments(stt_segments, max_duration_ms)
    results = []

    for long_seg in long_segs:
        ref_in_range = find_ref_segments_in_range(
            ref_segments,
            long_seg.start_ms,
            long_seg.end_ms,
            tolerance_ms=time_tol_ms,
        )

        split_suggestions = propose_split_points(long_seg, ref_in_range)

        results.append(
            LongSegmentInfo(
                segment_index=long_seg.index,
                segment=long_seg,
                ref_segments=ref_in_range,
                split_suggestions=split_suggestions,
            )
        )

    return results


def format_split_suggestion(
    stt_segment: Segment,
    suggestion: SplitSuggestion,
) -> str:
    """
    Отформатировать предложение разделения для вывода.

    Args:
        stt_segment: Длинный сегмент STT.
        suggestion: Предложение разделения.

    Returns:
        Форматированная строка.
    """
    from subs_diff.types import ms_to_srt

    split_time_str = ms_to_srt(suggestion.split_time_ms)
    part1_start = ms_to_srt(stt_segment.start_ms)
    part1_end = ms_to_srt(suggestion.split_time_ms)
    part2_start = ms_to_srt(suggestion.split_time_ms)
    part2_end = ms_to_srt(stt_segment.end_ms)

    lines = [
        f"Предложение разделения в {split_time_str}:",
        f"  Часть 1: {part1_start} --> {part1_end}",
        f"    Reference текст: {suggestion.ref_text_before[:100]}{'...' if len(suggestion.ref_text_before) > 100 else ''}",
        f"  Часть 2: {part2_start} --> {part2_end}",
        f"    Reference текст: {suggestion.ref_text_after[:100]}{'...' if len(suggestion.ref_text_after) > 100 else ''}",
        f"  Уверенность: {suggestion.confidence:.0%}",
    ]
    return "\n".join(lines)


def find_natural_split_points(text: str) -> list[int]:
    """
    Найти естественные точки разделения в тексте.

    Ищет позиции после знаков препинания (точка, восклицательный, вопросительный).

    Args:
        text: Текст для анализа.

    Returns:
        Список позиций для возможного разделения.
    """
    positions = []
    for match in re.finditer(r"[.!?]\s+", text):
        positions.append(match.end())
    return positions
