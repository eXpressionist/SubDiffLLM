"""Парсер файлов формата SRT."""

import re
from collections.abc import Iterable
from pathlib import Path

from subs_diff.types import Segment, srt_to_ms

# Альтернативный паттерн для более гибкого парсинга
SRT_TIMESTAMP_PATTERN = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)

# Паттерн для удаления тегов форматирования SRT
SRT_TAG_PATTERN = re.compile(r"<[^>]+>")

# Паттерн для токенизации (слова, числа, составные слова)
TOKEN_PATTERN = re.compile(r"\b[\w\-']+\b|\d+")


def parse_srt(content: str) -> list[Segment]:
    """
    Распарсить содержимое SRT-файла в список сегментов.

    Args:
        content: Содержимое SRT-файла как строка.

    Returns:
        Список Segment, отсортированный по времени начала.
    """
    segments = []

    # Нормализуем переносы строк
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Разбиваем на блоки по пустым строкам
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        segment = _parse_block(block)
        if segment:
            segments.append(segment)

    # Сортируем по времени начала
    segments.sort(key=lambda s: s.start_ms)

    return segments


def _parse_block(block: str) -> Segment | None:
    """
    Распарсить один блок SRT.

    Args:
        block: Строка блока SRT.

    Returns:
        Segment или None, если блок не распарсился.
    """
    lines = block.split("\n")

    # Пытаемся найти строку с таймингом
    timestamp_line_idx = None
    timestamp_match = None

    for i, line in enumerate(lines):
        match = SRT_TIMESTAMP_PATTERN.search(line)
        if match:
            timestamp_line_idx = i
            timestamp_match = match
            break

    if timestamp_line_idx is None or timestamp_match is None:
        return None

    # Парсим тайминг
    start_str = timestamp_match.group(1)
    end_str = timestamp_match.group(2)
    start_ms = srt_to_ms(start_str)
    end_ms = srt_to_ms(end_str)

    # Извлекаем текст (все строки после тайминга)
    text_lines = lines[timestamp_line_idx + 1 :]
    text = "\n".join(text_lines).strip()

    # Удаляем теги форматирования
    text = SRT_TAG_PATTERN.sub("", text)

    # Если текст пустой, пропускаем
    if not text:
        return None

    # Токенизируем
    tokens = tokenize(text)

    # Индекс будет установлен позже
    return Segment(
        index=0,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        tokens=tokens,
    )


def parse_srt_file(filepath: str | Path) -> list[Segment]:
    """
    Распарсить SRT-файл по пути.

    Args:
        filepath: Путь к SRT-файлу.

    Returns:
        Список Segment.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если файл не содержит валидных SRT-данных.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"SRT file not found: {path}")

    content = path.read_text(encoding="utf-8-sig")
    segments = parse_srt(content)

    if not segments:
        raise ValueError(f"No valid SRT segments found in {path}")

    # Пересоздаём сегменты с правильными индексами (Segment — frozen dataclass)
    segments = [
        Segment(index=i, start_ms=seg.start_ms, end_ms=seg.end_ms, text=seg.text, tokens=seg.tokens)
        for i, seg in enumerate(segments)
    ]

    return segments


def tokenize(text: str) -> list[str]:
    """
    Токенизировать текст.

    Args:
        text: Текст для токенизации.

    Returns:
        Список токенов.
    """
    # Находим все токены
    tokens = TOKEN_PATTERN.findall(text)
    return [t for t in tokens if t.strip()]


def normalize_text(text: str) -> str:
    """
    Нормализовать текст для сравнения.

    - Приводит к нижнему регистру
    - Удаляет лишнюю пунктуацию
    - Сжимает множественные пробелы
    - Сохраняет важные символы (дефисы в составных словах)

    Args:
        text: Исходный текст.

    Returns:
        Нормализованный текст.
    """
    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем теги
    text = SRT_TAG_PATTERN.sub("", text)

    # Заменяем множественные пробелы и переносы строк на один пробел
    text = re.sub(r"\s+", " ", text)

    # Удаляем пунктуацию на концах, но сохраняем внутреннюю
    text = text.strip(".,!?;:'\"—–-()[]{}")

    return text.strip()


def extract_text_only(content: str) -> str:
    """
    Извлечь только текст из SRT-содержимого (без таймингов).

    Полезно для быстрой предобработки.

    Args:
        content: Содержимое SRT-файла.

    Returns:
        Только текст, объединённый пробелами.
    """
    segments = parse_srt(content)
    return " ".join(seg.text for seg in segments)


def detect_forced_segments(segments: Iterable[Segment]) -> list[int]:
    """
    Детектировать возможные forced-сегменты (надписи на экране, чужой язык).

    Эвристики:
    - Очень короткие сегменты (< 1 сек) с заглавными буквами
    - Сегменты со смешением алфавитов (кириллица + латиница)
    - Сегменты с необычными символами

    Args:
        segments: Итератор сегментов.

    Returns:
        Список индексов сегментов, похожих на forced.
    """
    forced_indices = []

    # Паттерны для детекции
    cyrillic_pattern = re.compile(r"[а-яёА-ЯЁ]")
    latin_pattern = re.compile(r"[a-zA-Z]")
    all_caps_pattern = re.compile(r"^[^а-яёa-z]*[А-ЯЁA-Z]{2,}[^а-яёa-z]*$")

    for seg in segments:
        is_forced = False

        # Проверка на все заглавные (для коротких сегментов)
        if seg.duration_ms < 2000 and all_caps_pattern.match(seg.text.strip()):
            is_forced = True

        # Проверка на смешение алфавитов
        has_cyrillic = bool(cyrillic_pattern.search(seg.text))
        has_latin = bool(latin_pattern.search(seg.text))
        if has_cyrillic and has_latin:
            # Дополнительная проверка: если есть слова целиком на латинице
            for token in seg.tokens:
                if latin_pattern.match(token) and len(token) > 2:
                    is_forced = True
                    break

        if is_forced:
            forced_indices.append(seg.index)

    return forced_indices
