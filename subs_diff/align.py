"""Выравнивание сегментов и merge операций."""

from dataclasses import dataclass
from typing import Iterable

from subs_diff.types import Segment, MergedSegment, Candidate, SimilarityMetrics
from subs_diff.heuristics import compute_similarity, is_candidate, RareTokenDetector


@dataclass
class AlignmentResult:
    """Результат выравнивания сегментов."""

    candidates: list[Candidate]  # кандидаты на проблемы
    aligned_pairs: list[tuple[MergedSegment, MergedSegment]]  # все выровненные пары
    unmatched_a: list[MergedSegment]  # сегменты A без пары
    unmatched_b: list[MergedSegment]  # сегменты B без пары (возможные forced)


def find_time_window(
    segments: list[Segment],
    start_ms: int,
    end_ms: int,
    time_tol_ms: int,
    strict: bool = True,
) -> list[Segment]:
    """
    Найти сегменты в заданном временном окне с допуском.

    Args:
        segments: Список сегментов для поиска.
        start_ms: Начало окна (мс).
        end_ms: Конец окна (мс).
        time_tol_ms: Допуск по времени (мс).
        strict: Если True, отбрасывать сегменты, у которых start > end_ms + tol

    Returns:
        Список сегментов, попадающих в окно [start_ms - tol, end_ms + tol].
    """
    window_start = start_ms - time_tol_ms
    window_end = end_ms + time_tol_ms

    result = []
    for seg in segments:
        # Сегмент пересекает окно, если:
        # - его начало <= конец окна И
        # - его конец >= начало окна
        if seg.start_ms <= window_end and seg.end_ms >= window_start:
            # Дополнительная проверка: начало сегмента B не должно быть
            # значительно позже конца сегмента A (для избежания некорректных пар)
            if strict and seg.start_ms > window_end:
                continue
            result.append(seg)

    return result


def merge_segments(segments: list[Segment]) -> MergedSegment:
    """
    Объединить несколько сегментов в один.

    Args:
        segments: Список сегментов для объединения.

    Returns:
        MergedSegment с объединёнными текстом и токенами.

    Raises:
        ValueError: Если список сегментов пуст.
    """
    if not segments:
        raise ValueError("Cannot merge empty segments list")

    if len(segments) == 1:
        seg = segments[0]
        return MergedSegment(
            segments=[seg],
            text=seg.text,
            tokens=seg.tokens.copy(),
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            indices=[seg.index],
        )

    # Объединяем тексты с пробелом
    texts = []
    all_tokens = []
    indices = []

    for seg in segments:
        texts.append(seg.text)
        all_tokens.extend(seg.tokens)
        indices.append(seg.index)

    merged_text = " ".join(texts)

    return MergedSegment(
        segments=segments,
        text=merged_text,
        tokens=all_tokens,
        start_ms=segments[0].start_ms,
        end_ms=segments[-1].end_ms,
        indices=indices,
    )


def align_segments(
    segments_a: list[Segment],
    segments_b: list[Segment],
    time_tol: float = 1.0,
    max_merge: int = 5,
    min_score: float = 0.5,
) -> AlignmentResult:
    """
    Выровнять сегменты из двух файлов.

    Алгоритм:
    1. Для каждого сегмента A находим окно кандидатов B по времени
    2. Пробуем разные комбинации merge (1..max_merge) для A и B
    3. Выбираем пару с наилучшим сходством + проверкой таймингов
    4. Возвращаем все пары + unmatched сегменты (forced)

    Args:
        segments_a: Сегменты из файла A (STT).
        segments_b: Сегменты из файла B (reference).
        time_tol: Допуск по времени (секунды).
        max_merge: Максимальное количество сегментов для объединения.
        min_score: Минимальный порог сходства (для информации).

    Returns:
        AlignmentResult с выровненными парами и unmatched сегментами.
    """
    time_tol_ms = int(time_tol * 1000)
    aligned_pairs: list[tuple[MergedSegment, MergedSegment]] = []
    candidates: list[Candidate] = []
    matched_b_indices: set[int] = set()
    matched_a_indices: set[int] = set()

    for i, seg_a in enumerate(segments_a):
        if i in matched_a_indices:
            continue

        # Находим окно кандидатов B
        window_b = find_time_window(
            segments_b, seg_a.start_ms, seg_a.end_ms, time_tol_ms, strict=False
        )
        window_b = [seg for seg in window_b if seg.index not in matched_b_indices]

        if not window_b:
            continue

        # Пробуем разные комбинации merge
        best_match = None
        best_score = -1.0

        for merge_a_count in range(1, min(max_merge + 1, len(segments_a) - i + 1)):
            if i + merge_a_count > len(segments_a):
                break

            segments_a_merge = segments_a[i : i + merge_a_count]

            for merge_b_count in range(1, min(max_merge + 1, len(window_b) + 1)):
                for b_start in range(len(window_b) - merge_b_count + 1):
                    segments_b_merge = window_b[b_start : b_start + merge_b_count]

                    merged_a = merge_segments(segments_a_merge)
                    merged_b = merge_segments(segments_b_merge)

                    # Жёсткие ограничения по таймингам, чтобы не захватывать лишние реплики.
                    start_diff = abs(merged_b.start_ms - merged_a.start_ms)
                    end_diff = abs(merged_b.end_ms - merged_a.end_ms)
                    max_allowed_diff = int(time_tol_ms * 1.5)
                    if start_diff > max_allowed_diff or end_diff > max_allowed_diff:
                        continue

                    overlap_start = max(merged_a.start_ms, merged_b.start_ms)
                    overlap_end = min(merged_a.end_ms, merged_b.end_ms)
                    overlap_ms = max(0, overlap_end - overlap_start)

                    # Если нет пересечения и зазор большой, считаем пару некорректной.
                    if overlap_ms == 0:
                        gap_ms = max(
                            merged_b.start_ms - merged_a.end_ms,
                            merged_a.start_ms - merged_b.end_ms,
                        )
                        if gap_ms > int(time_tol_ms * 0.5):
                            continue

                    metrics = compute_similarity(merged_a, merged_b)
                    union_ms = max(merged_a.end_ms, merged_b.end_ms) - min(
                        merged_a.start_ms, merged_b.start_ms
                    )
                    timing_iou = (overlap_ms / union_ms) if union_ms > 0 else 0.0
                    scored_similarity = metrics.overall_similarity * (0.7 + 0.3 * timing_iou)

                    if scored_similarity > best_score:
                        best_score = scored_similarity
                        best_match = (merged_a, merged_b, metrics)

        if best_match:
            merged_a, merged_b, metrics = best_match
            aligned_pairs.append((merged_a, merged_b))
            if is_candidate(
                metrics, 
                min_score=min_score,
                a_tokens=merged_a.tokens,
                b_tokens=merged_b.tokens,
            ):
                candidates.append(
                    Candidate(
                        a_segment=merged_a,
                        b_segment=merged_b,
                        metrics=metrics,
                        is_forced_like=False,
                    )
                )

            for idx in merged_a.indices:
                matched_a_indices.add(idx)
            for idx in merged_b.indices:
                matched_b_indices.add(idx)

    # Unmatched сегменты B — это forced (текст в reference без соответствия в STT)
    unmatched_b: list[MergedSegment] = []
    for j, seg_b in enumerate(segments_b):
        if j not in matched_b_indices:
            merged_b = merge_segments([seg_b])
            unmatched_b.append(merged_b)

    # Unmatched сегменты A
    unmatched_a: list[MergedSegment] = []
    for i, seg_a in enumerate(segments_a):
        if i not in matched_a_indices:
            unmatched_a.append(merge_segments([seg_a]))

    # Добавляем candidates из unmatched B.
    # Важно: если в A есть близкий по времени сегмент, не считаем это forced автоматически.
    for merged_b in unmatched_b:
        best_temporal_a = None
        best_overlap_ms = -1
        best_timing_delta = 10**12

        for seg_a in segments_a:
            overlap_start = max(seg_a.start_ms, merged_b.start_ms)
            overlap_end = min(seg_a.end_ms, merged_b.end_ms)
            overlap_ms = max(0, overlap_end - overlap_start)
            timing_delta = abs(seg_a.start_ms - merged_b.start_ms) + abs(
                seg_a.end_ms - merged_b.end_ms
            )

            # Если сегменты не пересекаются и зазор большой, этот кандидат нам не подходит.
            if overlap_ms == 0:
                gap_ms = max(
                    seg_a.start_ms - merged_b.end_ms,
                    merged_b.start_ms - seg_a.end_ms,
                )
                if gap_ms > int(time_tol_ms * 0.5):
                    continue

            if overlap_ms > best_overlap_ms or (
                overlap_ms == best_overlap_ms and timing_delta < best_timing_delta
            ):
                best_overlap_ms = overlap_ms
                best_timing_delta = timing_delta
                best_temporal_a = seg_a

        if best_temporal_a is not None:
            merged_a = merge_segments([best_temporal_a])
            metrics = compute_similarity(merged_a, merged_b)
            if is_candidate(
                metrics, 
                min_score=min_score,
                a_tokens=merged_a.tokens,
                b_tokens=merged_b.tokens,
            ):
                candidates.append(
                    Candidate(
                        a_segment=merged_a,
                        b_segment=merged_b,
                        metrics=metrics,
                        is_forced_like=False,
                    )
                )
            continue

        candidates.append(
            Candidate(
                a_segment=MergedSegment(
                    segments=[], text="", tokens=[], start_ms=0, end_ms=0, indices=[]
                ),
                b_segment=merged_b,
                metrics=SimilarityMetrics(
                    jaccard=0.0, char_3gram=0.0, levenshtein=0.0,
                    length_ratio=0.0, rare_token_overlap=0.0, rare_token_missing=1.0,
                ),
                is_forced_like=True,
            )
        )

    return AlignmentResult(
        candidates=candidates,
        aligned_pairs=aligned_pairs,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
    )


def _detect_forced_in_pair(a: MergedSegment, b: MergedSegment) -> bool:
    """
    Детектировать, похожа ли пара на forced-сегмент.

    Forced-сегменты часто:
    - Очень короткие
    - Содержат смешение алфавитов
    - Содержат только заглавные буквы

    Args:
        a: Сегмент A.
        b: Сегмент B.

    Returns:
        True, если пара похожа на forced.
    """
    import re

    # Проверяем длительность
    if a.duration_ms < 1500 and b.duration_ms < 1500:
        # Короткие сегменты — проверяем на все заглавные
        a_all_caps = a.text.isupper() or bool(re.match(r"^[А-ЯЁ\s]+$", a.text))
        b_all_caps = b.text.isupper() or bool(re.match(r"^[А-ЯЁ\s]+$", b.text))
        if a_all_caps or b_all_caps:
            return True

    # Проверяем на смешение алфавитов
    cyrillic = re.compile(r"[а-яёА-ЯЁ]")
    latin = re.compile(r"[a-zA-Z]")

    for seg in [a, b]:
        has_cyrillic = bool(cyrillic.search(seg.text))
        has_latin = bool(latin.search(seg.text))
        if has_cyrillic and has_latin:
            return True

    return False


def stable_align(
    segments_a: list[Segment],
    segments_b: list[Segment],
    time_tol: float = 1.0,
    max_merge: int = 5,
) -> list[tuple[list[int], list[int]]]:
    """
    Стабильное выравнивание с возвратом индексов выровненных пар.

    Возвращает список пар (индексы A, индексы B), отсортированный по времени.

    Args:
        segments_a: Сегменты из файла A.
        segments_b: Сегменты из файла B.
        time_tol: Допуск по времени (секунды).
        max_merge: Максимальное количество сегментов для объединения.

    Returns:
        Список пар (a_indices, b_indices).
    """
    result = align_segments(segments_a, segments_b, time_tol=time_tol, max_merge=max_merge)

    pairs = []
    for merged_a, merged_b in result.aligned_pairs:
        pairs.append((merged_a.indices, merged_b.indices))

    # Сортируем по времени начала A
    pairs.sort(key=lambda p: segments_a[p[0][0]].start_ms if p[0] else 0)

    return pairs
