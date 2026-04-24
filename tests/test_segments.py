"""Тесты для модуля segments."""

from subs_diff.segments import (
    analyze_long_segments,
    find_long_segments,
    find_natural_split_points,
    find_ref_segments_in_range,
    propose_split_points,
)
from subs_diff.types import LongSegmentInfo, Segment, SplitSuggestion


def make_segment(index: int, start_ms: int, end_ms: int, text: str) -> Segment:
    """Создать тестовый сегмент."""
    return Segment(
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        tokens=text.lower().split(),
    )


class TestFindLongSegments:
    """Тесты для find_long_segments."""

    def test_empty_list(self):
        """Пустой список возвращает пустой результат."""
        result = find_long_segments([])
        assert result == []

    def test_no_long_segments(self):
        """Нет длинных сегментов."""
        segments = [
            make_segment(0, 0, 3000, "Короткий"),
            make_segment(1, 3000, 6000, "Тоже короткий"),
        ]
        result = find_long_segments(segments, max_duration_ms=6000)
        assert result == []

    def test_find_one_long_segment(self):
        """Найти один длинный сегмент."""
        segments = [
            make_segment(0, 0, 3000, "Короткий"),
            make_segment(1, 3000, 10000, "Длинный сегмент"),
            make_segment(2, 10000, 13000, "Короткий"),
        ]
        result = find_long_segments(segments, max_duration_ms=6000)
        assert len(result) == 1
        assert result[0].index == 1

    def test_find_multiple_long_segments(self):
        """Найти несколько длинных сегментов."""
        segments = [
            make_segment(0, 0, 8000, "Длинный 1"),
            make_segment(1, 8000, 11000, "Короткий"),
            make_segment(2, 11000, 20000, "Длинный 2"),
        ]
        result = find_long_segments(segments, max_duration_ms=6000)
        assert len(result) == 2
        assert [s.index for s in result] == [0, 2]

    def test_custom_threshold(self):
        """Кастомный порог."""
        segments = [
            make_segment(0, 0, 5000, "5 секунд"),
            make_segment(1, 5000, 12000, "7 секунд"),
        ]
        result_default = find_long_segments(segments)
        result_custom = find_long_segments(segments, max_duration_ms=4000)
        assert len(result_default) == 1
        assert len(result_custom) == 2


class TestFindRefSegmentsInRange:
    """Тесты для find_ref_segments_in_range."""

    def test_empty_ref(self):
        """Пустой reference возвращает пустой результат."""
        result = find_ref_segments_in_range([], 0, 10000)
        assert result == []

    def test_find_overlapping(self):
        """Найти пересекающиеся сегменты."""
        ref_segments = [
            make_segment(0, 0, 3000, "Первый"),
            make_segment(1, 3000, 6000, "Второй"),
            make_segment(2, 6000, 9000, "Третий"),
            make_segment(3, 9000, 12000, "Четвёртый"),
        ]
        result = find_ref_segments_in_range(ref_segments, 2000, 8000)
        assert len(result) == 3
        assert [s.index for s in result] == [0, 1, 2]

    def test_find_exact_match(self):
        """Точное совпадение границ."""
        ref_segments = [
            make_segment(0, 5000, 7000, "Точный"),
        ]
        result = find_ref_segments_in_range(ref_segments, 5000, 7000)
        assert len(result) == 1

    def test_tolerance(self):
        """Допуск по времени - сегмент за пределами окна с tolerance."""
        ref_segments = [
            make_segment(0, 0, 3000, "Ранний"),
            make_segment(1, 15000, 18000, "Поздний"),
        ]
        result = find_ref_segments_in_range(ref_segments, 5000, 10000, tolerance_ms=1000)
        assert len(result) == 0


class TestProposeSplitPoints:
    """Тесты для propose_split_points."""

    def test_no_ref_segments(self):
        """Нет reference сегментов - нет предложений."""
        stt_segment = make_segment(0, 0, 10000, "Длинный сегмент STT")
        result = propose_split_points(stt_segment, [])
        assert result == []

    def test_single_split_point(self):
        """Одна точка разделения."""
        stt_segment = make_segment(0, 0, 10000, "Длинный сегмент STT")
        ref_segments = [
            make_segment(0, 0, 4000, "Первая часть"),
            make_segment(1, 4000, 10000, "Вторая часть"),
        ]
        result = propose_split_points(stt_segment, ref_segments)
        assert len(result) >= 1
        assert result[0].split_time_ms == 4000

    def test_minimum_split_duration(self):
        """Слишком короткие части не предлагаются."""
        stt_segment = make_segment(0, 0, 10000, "Длинный")
        ref_segments = [
            make_segment(0, 0, 500, "Очень короткий"),
            make_segment(1, 500, 10000, "Почти всё"),
        ]
        result = propose_split_points(stt_segment, ref_segments, min_split_duration_ms=1000)
        assert len(result) == 0

    def test_multiple_split_points(self):
        """Несколько точек разделения."""
        stt_segment = make_segment(0, 0, 15000, "Очень длинный сегмент")
        ref_segments = [
            make_segment(0, 0, 5000, "Часть 1"),
            make_segment(1, 5000, 10000, "Часть 2"),
            make_segment(2, 10000, 15000, "Часть 3"),
        ]
        result = propose_split_points(stt_segment, ref_segments)
        assert len(result) >= 2


class TestAnalyzeLongSegments:
    """Тесты для analyze_long_segments."""

    def test_empty_stt(self):
        """Пустой STT возвращает пустой результат."""
        result = analyze_long_segments([], [])
        assert result == []

    def test_analyze_single_long(self):
        """Анализ одного длинного сегмента."""
        stt_segments = [
            make_segment(0, 0, 10000, "Длинный сегмент STT"),
        ]
        ref_segments = [
            make_segment(0, 0, 5000, "Первая половина"),
            make_segment(1, 5000, 10000, "Вторая половина"),
        ]
        result = analyze_long_segments(stt_segments, ref_segments)
        assert len(result) == 1
        assert result[0].segment_index == 0
        assert len(result[0].ref_segments) == 2
        assert len(result[0].split_suggestions) >= 1

    def test_mixed_lengths(self):
        """Смешанные длины сегментов."""
        stt_segments = [
            make_segment(0, 0, 3000, "Короткий"),
            make_segment(1, 3000, 12000, "Длинный"),
            make_segment(2, 12000, 15000, "Короткий"),
        ]
        ref_segments = [
            make_segment(0, 0, 3000, "Ref 1"),
            make_segment(1, 3000, 7000, "Ref 2"),
            make_segment(2, 7000, 12000, "Ref 3"),
            make_segment(3, 12000, 15000, "Ref 4"),
        ]
        result = analyze_long_segments(stt_segments, ref_segments)
        assert len(result) == 1
        assert result[0].segment_index == 1

    def test_overlong_whisper_segment_still_reported_with_split_suggestions(self):
        stt_segments = [
            make_segment(0, 0, 20000, "РћС‡РµРЅСЊ РґР»РёРЅРЅС‹Р№ СЃРµРіРјРµРЅС‚ Whisper"),
        ]
        ref_segments = [
            make_segment(0, 0, 5000, "Р§Р°СЃС‚СЊ 1"),
            make_segment(1, 5000, 10000, "Р§Р°СЃС‚СЊ 2"),
            make_segment(2, 10000, 15000, "Р§Р°СЃС‚СЊ 3"),
            make_segment(3, 15000, 20000, "Р§Р°СЃС‚СЊ 4"),
        ]

        result = analyze_long_segments(stt_segments, ref_segments, max_duration_ms=6000)

        assert len(result) == 1
        assert result[0].segment.duration_ms == 20000
        assert len(result[0].ref_segments) == 4
        assert len(result[0].split_suggestions) >= 3


class TestFindNaturalSplitPoints:
    """Тесты для find_natural_split_points."""

    def test_no_punctuation(self):
        """Нет знаков препинания - нет точек."""
        result = find_natural_split_points("Простой текст без знаков")
        assert result == []

    def test_single_sentence(self):
        """Одно предложение."""
        result = find_natural_split_points("Одно предложение. ")
        assert len(result) == 1

    def test_multiple_sentences(self):
        """Несколько предложений."""
        result = find_natural_split_points("Первое. Второе! Третье? ")
        assert len(result) == 3

    def test_no_trailing_space(self):
        """Без пробела после знака."""
        result = find_natural_split_points("Текст.")
        assert len(result) == 0


class TestSplitSuggestionDataclass:
    """Тесты для dataclass SplitSuggestion."""

    def test_create_suggestion(self):
        """Создание предложения."""
        sugg = SplitSuggestion(
            split_time_ms=5000,
            ref_text_before="Первая часть",
            ref_text_after="Вторая часть",
            confidence=0.85,
        )
        assert sugg.split_time_ms == 5000
        assert sugg.confidence == 0.85


class TestLongSegmentInfoDataclass:
    """Тесты для dataclass LongSegmentInfo."""

    def test_create_info(self):
        """Создание информации о сегменте."""
        seg = make_segment(0, 0, 10000, "Длинный")
        info = LongSegmentInfo(
            segment_index=0,
            segment=seg,
            ref_segments=[],
            split_suggestions=[],
        )
        assert info.segment_index == 0
        assert info.segment.duration_ms == 10000
