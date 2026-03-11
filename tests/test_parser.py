"""Тесты для парсера SRT."""

import pytest
from pathlib import Path

from subs_diff.parser import (
    parse_srt,
    parse_srt_file,
    tokenize,
    normalize_text,
    detect_forced_segments,
)
from subs_diff.types import srt_to_ms, ms_to_srt


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSrtToMs:
    """Тесты конвертации SRT времени в миллисекунды."""

    def test_basic_conversion(self):
        assert srt_to_ms("00:00:01,000") == 1000
        assert srt_to_ms("00:00:05,500") == 5500
        assert srt_to_ms("00:01:00,000") == 60000

    def test_with_dots(self):
        # SRT может использовать точки вместо запятых
        assert srt_to_ms("00:00:01.000") == 1000
        assert srt_to_ms("00:00:05.500") == 5500

    def test_hours(self):
        assert srt_to_ms("01:00:00,000") == 3600000
        assert srt_to_ms("02:30:00,000") == 9000000


class TestMsToSrt:
    """Тесты конвертации миллисекунд в SRT формат."""

    def test_basic_conversion(self):
        assert ms_to_srt(1000) == "00:00:01,000"
        assert ms_to_srt(5500) == "00:00:05,500"
        assert ms_to_srt(60000) == "00:01:00,000"

    def test_hours(self):
        assert ms_to_srt(3600000) == "01:00:00,000"
        assert ms_to_srt(9000000) == "02:30:00,000"

    def test_roundtrip(self):
        # Проверяем обратимость конвертации
        time_str = "00:00:01,500"
        assert ms_to_srt(srt_to_ms(time_str)) == time_str


class TestTokenize:
    """Тесты токенизации."""

    def test_simple_text(self):
        tokens = tokenize("Привет мир")
        assert tokens == ["Привет", "мир"]

    def test_with_punctuation(self):
        tokens = tokenize("Привет, мир! Как дела?")
        assert "Привет" in tokens
        assert "мир" in tokens
        assert "Как" in tokens
        assert "дела" in tokens

    def test_with_numbers(self):
        tokens = tokenize("Дом номер 42 на улице")
        assert "42" in tokens

    def test_compound_words(self):
        tokens = tokenize("Бейкер-стрит, 221Б")
        assert any("Бейкер" in t for t in tokens)


class TestNormalizeText:
    """Тесты нормализации текста."""

    def test_lowercase(self):
        assert normalize_text("Привет МИР") == "привет мир"

    def test_multiple_spaces(self):
        assert normalize_text("Привет    мир") == "привет мир"

    def test_strip_punctuation(self):
        result = normalize_text("Привет, мир!")
        # Запятая внутри сохраняется, восклицательный знак в конце удаляется
        assert result == "привет, мир"

    def test_preserve_hyphens(self):
        result = normalize_text("Бейкер-стрит")
        assert "бейкер-стрит" in result


class TestParseSrt:
    """Тесты парсинга SRT содержимого."""

    def test_parse_simple(self):
        content = """1
00:00:01,000 --> 00:00:04,000
Привет, мир!

2
00:00:05,000 --> 00:00:08,000
Как дела?
"""
        segments = parse_srt(content)
        assert len(segments) == 2
        assert segments[0].text == "Привет, мир!"
        assert segments[1].text == "Как дела?"

    def test_parse_multiline(self):
        content = """1
00:00:01,000 --> 00:00:04,000
Первая строка
Вторая строка
"""
        segments = parse_srt(content)
        assert len(segments) == 1
        assert "Первая строка" in segments[0].text
        assert "Вторая строка" in segments[0].text

    def test_parse_with_tags(self):
        content = """1
00:00:01,000 --> 00:00:04,000
<i>Привет</i>, <b>мир</b>!
"""
        segments = parse_srt(content)
        assert len(segments) == 1
        # Теги должны быть удалены
        assert "<i>" not in segments[0].text

    def test_parse_empty(self):
        segments = parse_srt("")
        assert len(segments) == 0

    def test_parse_invalid(self):
        content = """Это не SRT формат
просто текст
"""
        segments = parse_srt(content)
        assert len(segments) == 0


class TestParseSrtFile:
    """Тесты парсинга SRT файлов."""

    def test_parse_fixture_file(self):
        filepath = FIXTURES_DIR / "sample_a.srt"
        segments = parse_srt_file(filepath)
        assert len(segments) == 10
        assert "Шерлок Холмс" in segments[0].text

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_srt_file("/nonexistent/path.srt")


class TestDetectForcedSegments:
    """Тесты детекции forced-сегментов."""

    def test_all_caps_short(self):
        from subs_diff.types import Segment

        segments = [
            Segment(
                index=0,
                start_ms=0,
                end_ms=1000,  # 1 секунда
                text="СРОЧНО",
                tokens=["СРОЧНО"],
            )
        ]
        forced = detect_forced_segments(segments)
        assert 0 in forced

    def test_mixed_alphabets(self):
        from subs_diff.types import Segment

        segments = [
            Segment(
                index=0,
                start_ms=0,
                end_ms=3000,
                text="Привет Hello мир",
                tokens=["Привет", "Hello", "мир"],
            )
        ]
        forced = detect_forced_segments(segments)
        assert 0 in forced

    def test_normal_text(self):
        from subs_diff.types import Segment

        segments = [
            Segment(
                index=0,
                start_ms=0,
                end_ms=3000,
                text="Обычный текст субтитра",
                tokens=["Обычный", "текст", "субтитра"],
            )
        ]
        forced = detect_forced_segments(segments)
        assert len(forced) == 0
