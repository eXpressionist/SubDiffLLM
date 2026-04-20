"""Эвристики для сравнения сегментов: метрики сходства, детектор редких токенов."""

import re
from collections import Counter
from collections.abc import Iterable

from rapidfuzz import fuzz

from subs_diff.parser import normalize_text
from subs_diff.types import MergedSegment, SimilarityMetrics

# Паттерны для детекции редких токенов
RARE_TOKEN_PATTERNS = [
    re.compile(r"\d"),  # содержит цифры
    re.compile(r"-"),  # содержит дефисы
    re.compile(r"[а-яёА-ЯЁ][a-zA-Z]|[a-zA-Z][а-яёА-ЯЁ]"),  # смешение кириллицы и латиницы
    re.compile(r"^[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)+$"),  # несколько слов с заглавной (имена)
]

# Минимальная длина для рассмотрения токена как "редкого"
MIN_RARE_TOKEN_LENGTH = 4


class RareTokenDetector:
    """Детектор редких токенов для приоритезации расхождений."""

    def __init__(self, segments: Iterable[Iterable[str]] | None = None):
        """
        Инициализировать детектор.

        Args:
            segments: Опционально, коллекция токенов из всех сегментов
                     для вычисления частот.
        """
        self._token_frequencies: Counter[str] = Counter()
        self._total_tokens = 0

        if segments:
            for seg_tokens in segments:
                tokens = list(seg_tokens)
                self._token_frequencies.update(tokens)
                self._total_tokens += len(tokens)

    def is_rare(self, token: str) -> bool:
        """
        Проверить, является ли токен редким.

        Критерии редкости:
        - Длина >= MIN_RARE_TOKEN_LENGTH
        - Содержит цифры, дефисы, или смешение алфавитов
        - Начинается с заглавной буквы (в оригинале)
        - Низкая частота в корпусе (если данные доступны)

        Args:
            token: Токен для проверки.

        Returns:
            True, если токен редкий.
        """
        if len(token) < MIN_RARE_TOKEN_LENGTH:
            return False

        # Проверяем паттерны
        for pattern in RARE_TOKEN_PATTERNS:
            if pattern.search(token):
                return True

        # Проверяем частоту (если данные доступны)
        if self._total_tokens > 0:
            freq = self._token_frequencies.get(token.lower(), 0)
            relative_freq = freq / self._total_tokens if self._total_tokens > 0 else 0
            # Токен редкий, если встречается реже 0.1% от всех токенов
            if relative_freq < 0.001 and freq < 5:
                return True

        return False

    def extract_rare_tokens(self, tokens: list[str]) -> list[str]:
        """
        Извлечь редкие токены из списка.

        Args:
            tokens: Список токенов.

        Returns:
            Список редких токенов.
        """
        return [t for t in tokens if self.is_rare(t)]


def compute_similarity(a: MergedSegment, b: MergedSegment) -> SimilarityMetrics:
    """
    Вычислить метрики сходства между двумя сегментами.

    Args:
        a: Сегмент A.
        b: Сегмент B.

    Returns:
        SimilarityMetrics с вычисленными значениями.
    """
    # Нормализуем тексты
    a_norm = normalize_text(a.text)
    b_norm = normalize_text(b.text)

    # Токены (нормализованные)
    a_tokens = {t.lower() for t in a.tokens}
    b_tokens = {t.lower() for t in b.tokens}

    # Jaccard similarity по токенам
    jaccard = _jaccard_similarity(a_tokens, b_tokens)

    # Character 3-gram similarity
    char_3gram = _char_ngram_similarity(a_norm, b_norm, n=3)

    # Levenshtein ratio (через rapidfuzz)
    levenshtein = fuzz.ratio(a_norm, b_norm) / 100.0

    # Отношение длин
    len_a = len(a_norm)
    len_b = len(b_norm)
    if len_a == 0 and len_b == 0:
        length_ratio = 1.0
    elif len_a == 0 or len_b == 0:
        length_ratio = 0.0
    else:
        length_ratio = min(len_a, len_b) / max(len_a, len_b)

    # Rare token overlap
    rare_detector = RareTokenDetector()
    a_rare = set(rare_detector.extract_rare_tokens(a.tokens))
    b_rare = set(rare_detector.extract_rare_tokens(b.tokens))

    if len(a_rare) == 0 and len(b_rare) == 0:
        rare_token_overlap = 1.0
        rare_token_missing = 0.0
    elif len(a_rare) == 0:
        rare_token_overlap = 0.0
        rare_token_missing = 1.0
    elif len(b_rare) == 0:
        rare_token_overlap = 1.0
        rare_token_missing = 0.0
    else:
        rare_intersection = a_rare & b_rare
        rare_token_overlap = len(rare_intersection) / max(len(a_rare), len(b_rare))
        rare_token_missing = len(b_rare - a_rare) / len(b_rare)

    return SimilarityMetrics(
        jaccard=jaccard,
        char_3gram=char_3gram,
        levenshtein=levenshtein,
        length_ratio=length_ratio,
        rare_token_overlap=rare_token_overlap,
        rare_token_missing=rare_token_missing,
    )


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Вычислить Jaccard similarity между двумя множествами."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _char_ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """
    Вычислить similarity на основе character n-gram.

    Использует rapidfuzz для эффективности.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # fuzz.ratio использует алгоритм, похожий на n-gram similarity
    return fuzz.ratio(s1, s2) / 100.0


def is_candidate(
    metrics: SimilarityMetrics,
    min_score: float = 0.5,
    rare_token_threshold: float = 0.5,
    a_tokens: list[str] | None = None,
    b_tokens: list[str] | None = None,
) -> bool:
    """
    Определить, является ли пара сегментов кандидатом на проблему.

    Кандидат если:
    - Общее сходство ниже порога
    - ИЛИ редкие токены сильно расходятся
    - В B есть редкие токены, отсутствующие в A
    - ИЛИ разные имена/названия (semantic mismatch)
    - ИЛИ разные числа
    - ИЛИ антонимы

    Args:
        metrics: Метрики сходства.
        min_score: Минимальный порог общего сходства.
        rare_token_threshold: Порог для rare_token_missing.
        a_tokens: Токены из A (для семантических проверок).
        b_tokens: Токены из B (для семантических проверок).

    Returns:
        True, если это кандидат на проблему.
    """
    # Низкое общее сходство
    if metrics.overall_similarity < min_score:
        return True

    # ASR-подобное искажение: мало общих токенов, но строки формально "похожи"
    # (например, кривая распознавалка с подстановкой фрагментов).
    # Такие пары лучше обязательно отправлять в LLM.
    if metrics.jaccard < 0.42 and metrics.levenshtein < 0.72:
        return True

    # Редкие токены расходятся
    if metrics.rare_token_overlap < rare_token_threshold:
        return True

    # В B есть редкие токены, которых нет в A
    if metrics.rare_token_missing > rare_token_threshold:
        return True

    # Семантические проверки (если переданы токены)
    if a_tokens and b_tokens:
        # Разные имена/названия — высокий приоритет
        if has_different_entities(a_tokens, b_tokens):
            return True

        # Разные числа
        if has_different_numbers(a_tokens, b_tokens):
            return True

        # Антонимы (противоположные по смыслу слова)
        if has_antonyms(a_tokens, b_tokens):
            return True

    return False


# Антонимы (пары противоположных слов)
ANTONYMS = {
    # Русские
    ("да", "нет"),
    ("можно", "нельзя"),
    ("надо", "не надо"),
    ("хорошо", "плохо"),
    ("большой", "маленький"),
    ("много", "мало"),
    ("быстро", "медленно"),
    ("рано", "поздно"),
    ("вчера", "завтра"),
    ("утро", "вечер"),
    ("день", "ночь"),
    ("лето", "зима"),
    ("начало", "конец"),
    ("первый", "последний"),
    ("старый", "новый"),
    ("открыть", "закрыть"),
    ("войти", "выйти"),
    ("прийти", "уйти"),
    ("взять", "дать"),
    ("купить", "продать"),
    ("найти", "потерять"),
    ("жить", "умереть"),
    ("любить", "ненавидеть"),
    ("друг", "враг"),
    ("восток", "запад"),
    ("север", "юг"),
    ("лево", "право"),
    ("верх", "низ"),
    ("вперёд", "назад"),
    ("туда", "сюда"),
    ("ист", "вест"),
    ("ист-энд", "вест-энд"),  # районы Лондона
    # Английские
    ("yes", "no"),
    ("can", "cannot"),
    ("good", "bad"),
    ("big", "small"),
    ("many", "few"),
    ("fast", "slow"),
    ("early", "late"),
    ("yesterday", "tomorrow"),
    ("morning", "evening"),
    ("day", "night"),
    ("summer", "winter"),
    ("begin", "end"),
    ("first", "last"),
    ("old", "new"),
    ("open", "close"),
    ("enter", "exit"),
    ("come", "go"),
    ("take", "give"),
    ("buy", "sell"),
    ("find", "lose"),
    ("live", "die"),
    ("love", "hate"),
    ("friend", "enemy"),
    ("east", "west"),
    ("north", "south"),
    ("left", "right"),
    ("up", "down"),
    ("forward", "backward"),
}

# Разворачиваем в словарь для быстрого поиска
ANTONYM_MAP = {}
for w1, w2 in ANTONYMS:
    ANTONYM_MAP[w1.lower()] = w2.lower()
    ANTONYM_MAP[w2.lower()] = w1.lower()


def has_different_entities(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """
    Проверить, есть ли разные именованные сущности в A и B.

    Считаем что имена разные, если:
    - Оба сегмента содержат имена с заглавной буквы
    - Имена не совпадают

    Args:
        a_tokens: Токены из A.
        b_tokens: Токены из B.

    Returns:
        True, если есть разные имена/названия.
    """
    entities_a = set(detect_named_entities(a_tokens))
    entities_b = set(detect_named_entities(b_tokens))

    # Если в обоих есть сущности
    if entities_a and entities_b:
        # Приводим к нижнему регистру для сравнения
        entities_a_lower = {e.lower() for e in entities_a}
        entities_b_lower = {e.lower() for e in entities_b}

        # Если множества не пересекаются — разные имена
        if not entities_a_lower.intersection(entities_b_lower):
            return True

        # Если есть непересекающиеся элементы — тоже кандидат
        diff_a = entities_a_lower - entities_b_lower
        diff_b = entities_b_lower - entities_a_lower

        # Игнорируем служебные слова
        trivial = {"да", "нет", "вот", "это", "как", "там", "тут", "тебя", "меня"}
        diff_a -= trivial
        diff_b -= trivial

        if diff_a or diff_b:
            return True

    return False


def has_different_numbers(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """
    Проверить, есть ли разные числа в A и B.

    Args:
        a_tokens: Токены из A.
        b_tokens: Токены из B.

    Returns:
        True, если числа отличаются.
    """

    def extract_numbers(tokens: list[str]) -> set[str]:
        numbers: set[str] = set()
        for token in tokens:
            # Чистые числа
            if token.isdigit():
                numbers.add(token)
            # Числа с суффиксами (1-й, 2-го и т.п.)
            elif re.match(r"^\d+[-\w]*$", token):
                number_match = re.match(r"^\d+", token)
                if number_match:
                    numbers.add(number_match.group())
        return numbers

    nums_a = extract_numbers(a_tokens)
    nums_b = extract_numbers(b_tokens)

    if nums_a and nums_b:
        # Если множества чисел не равны — кандидат
        if nums_a != nums_b:
            return True

    return False


def has_antonyms(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """
    Проверить, есть ли антонимы между A и B.

    Антонимы — слова с противоположным смыслом (да/нет, восток/запад).

    Args:
        a_tokens: Токены из A.
        b_tokens: Токены из B.

    Returns:
        True, если найдены антонимы.
    """
    a_lower = {t.lower() for t in a_tokens}
    b_lower = {t.lower() for t in b_tokens}

    for token_a in a_lower:
        if token_a in ANTONYM_MAP:
            antonym = ANTONYM_MAP[token_a]
            if antonym in b_lower:
                return True

    return False


def detect_named_entities(tokens: list[str]) -> list[str]:
    """
    Детектировать возможные именованные сущности в токенах.

    Эвристики:
    - Слова с заглавной буквы (в оригинале)
    - Составные слова с дефисом
    - Слова, содержащие цифры

    Args:
        tokens: Список токенов (оригинальные, не нормализованные).

    Returns:
        Список возможных именованных сущностей.
    """
    entities = []

    # Паттерн для имён/названий
    name_pattern = re.compile(r"^[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?$")
    latin_name_pattern = re.compile(r"^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$")

    for token in tokens:
        # Проверяем паттерны имён
        if name_pattern.match(token) or latin_name_pattern.match(token):
            entities.append(token)
            continue

        # Проверяем на смешение алфавитов (транслитерация)
        has_cyrillic = bool(re.search(r"[а-яёА-ЯЁ]", token))
        has_latin = bool(re.search(r"[a-zA-Z]", token))
        if has_cyrillic and has_latin:
            entities.append(token)
            continue

        # Содержит цифры (но не чисто цифры)
        if re.search(r"\d", token) and not token.isdigit():
            entities.append(token)

    return entities


def find_missing_content(
    a_tokens: list[str], b_tokens: list[str], min_length: int = 4
) -> list[str]:
    """
    Найти токены из B, отсутствующие в A.

    Игнорирует служебные части речи (предлоги, союзы, артикли).

    Args:
        a_tokens: Токены из A.
        b_tokens: Токены из B.
        min_length: Минимальная длина токена для учёта.

    Returns:
        Список отсутствующих значимых токенов.
    """
    # Служебные части речи (русский + английский)
    stopwords = {
        # Русские
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "но",
        "вы",
        "бы",
        "её",
        "ее",
        "их",
        "же",
        "за",
        "из",
        "или",
        "мне",
        "ми",
        "мы",
        "ни",
        "о",
        "об",
        "от",
        "по",
        "под",
        "при",
        "про",
        "ту",
        "ты",
        "у",
        "эти",
        "этот",
        "та",
        "тот",
        "тех",
        "том",
        "сам",
        "там",
        "над",
        "без",
        "для",
        "всё",
        "весь",
        "вся",
        "все",
        "всех",
        "всем",
        "всему",
        "всего",
        # Английские
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "his",
        "her",
        "their",
        "my",
        "your",
        "our",
    }

    a_lower = {t.lower() for t in a_tokens}
    missing = []

    for token in b_tokens:
        if len(token) < min_length:
            continue
        if token.lower() in stopwords:
            continue
        if token.lower() not in a_lower:
            missing.append(token)

    return missing


def compress_text(text: str, max_chars: int = 500) -> str:
    """
    Сжать текст для отправки в LLM.

    - Удаляет повторы пробелов
    - Обрезает до max_chars
    - Сохраняет начало и конец при обрезке

    Args:
        text: Исходный текст.
        max_chars: Максимальная длина.

    Returns:
        Сжатый текст.
    """
    # Удаляем повторы пробелов
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if len(text) <= max_chars:
        return text

    # Обрезаем с индикацией
    keep_from_start = max_chars // 2 - 2
    keep_from_end = max_chars // 2 - 3
    return text[:keep_from_start] + " [...] " + text[-keep_from_end:]
