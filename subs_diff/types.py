"""Типы данных для SubDiff."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """Уровень важности проблемы."""

    LOW = "low"
    MED = "med"
    HIGH = "high"


class Category(str, Enum):
    """Категория проблемы."""

    MISSING_CONTENT = "missing_content"
    WRONG_NAMED_ENTITY = "wrong_named_entity"
    TERMINOLOGY = "terminology"
    FORCED_MISMATCH = "forced_mismatch"
    LONG_SEGMENT = "long_segment"
    OTHER = "other"


class LLMMode(str, Enum):
    """Режим работы с LLM."""

    OFF = "off"
    AUTO = "auto"
    LOCAL = "local"
    API = "api"


@dataclass(frozen=True)
class Segment:
    """Один сегмент субтитров (одна реплика)."""

    index: int  # порядковый номер в исходном файле
    start_ms: int  # время начала в миллисекундах
    end_ms: int  # время окончания в миллисекундах
    text: str  # оригинальный текст
    tokens: list[str] = field(default_factory=list)  # токенизированный текст

    @property
    def duration_ms(self) -> int:
        """Длительность сегмента в миллисекундах."""
        return self.end_ms - self.start_ms

    def to_time_string(self) -> str:
        """Конвертировать время в формат SRT (HH:MM:SS,mmm)."""
        return f"{ms_to_srt(self.start_ms)} --> {ms_to_srt(self.end_ms)}"


@dataclass
class MergedSegment:
    """Объединённый сегмент (несколько исходных сегментов)."""

    segments: list[Segment]  # исходные сегменты
    text: str  # объединённый текст
    tokens: list[str] = field(default_factory=list)  # объединённые токены
    start_ms: int = 0  # время начала первого сегмента
    end_ms: int = 0  # время окончания последнего сегмента
    indices: list[int] = field(default_factory=list)  # индексы исходных сегментов

    @property
    def duration_ms(self) -> int:
        """Длительность сегмента в миллисекундах."""
        return self.end_ms - self.start_ms

    @classmethod
    def from_segments(cls, segments: list[Segment]) -> "MergedSegment":
        """Создать объединённый сегмент из списка сегментов."""
        if not segments:
            raise ValueError("Segments list cannot be empty")

        text = " ".join(seg.text for seg in segments)
        tokens = []
        for seg in segments:
            tokens.extend(seg.tokens)

        return cls(
            segments=segments,
            text=text,
            tokens=tokens,
            start_ms=segments[0].start_ms,
            end_ms=segments[-1].end_ms,
            indices=[seg.index for seg in segments],
        )


@dataclass
class SimilarityMetrics:
    """Метрики сходства между двумя сегментами."""

    jaccard: float  # Jaccard similarity по токенам
    char_3gram: float  # Character 3-gram similarity
    levenshtein: float  # Levenshtein ratio (0-1)
    length_ratio: float  # Отношение длин (меньшая/большая)
    rare_token_overlap: float  # Доля общих редких токенов
    rare_token_missing: float  # Доля редких токенов B, отсутствующих в A

    @property
    def overall_similarity(self) -> float:
        """Общая метрика сходства (взвешенная средняя)."""
        # Приоритет: rare_token_overlap > levenshtein > char_3gram > jaccard
        return (
            0.35 * self.rare_token_overlap
            + 0.30 * self.levenshtein
            + 0.20 * self.char_3gram
            + 0.15 * self.jaccard
        )


@dataclass
class Candidate:
    """Кандидат на проблему (расхождение между A и B)."""

    a_segment: MergedSegment  # сегмент(ы) из файла A
    b_segment: MergedSegment  # сегмент(ы) из файла B
    metrics: SimilarityMetrics  # метрики сходства
    is_forced_like: bool = False  # похоже на forced-сегмент


@dataclass
class LLMVerdict:
    """Результат верификации кандидата через LLM."""

    severity: Severity
    category: Category
    short_reason: str
    evidence: str
    confidence: float  # 0.0 - 1.0
    suggested_fix: Optional[str] = None
    is_forced: bool = False


@dataclass
class Issue:
    """Проблема в субтитрах."""

    issue_id: str
    time_range: tuple[int, int]  # (start_ms, end_ms)
    a_segments: list[int]  # индексы сегментов в A
    b_segments: list[int]  # индексы сегментов в B
    severity: Severity
    category: Category
    short_reason: str
    evidence: str
    forced_detected: bool
    llm_verdict: Optional[LLMVerdict] = None
    a_text: str = ""  # текст из A (для отчёта)
    b_text: str = ""  # текст из B (для отчёта)


@dataclass
class ReportMetadata:
    """Метаданные отчёта."""

    generated_at: str
    stt_file: str
    ref_file: str
    config: dict


@dataclass
class ReportSummary:
    """Сводка по отчёту."""

    total_issues: int
    by_severity: dict[str, int]
    by_category: dict[str, int]


@dataclass
class Report:
    """Полный отчёт о сравнении."""

    metadata: ReportMetadata
    issues: list[Issue]
    summary: ReportSummary


@dataclass
class SplitSuggestion:
    """Предложение разделения длинного сегмента."""

    split_time_ms: int  # время разделения в мс
    ref_text_before: str  # текст reference до точки разделения
    ref_text_after: str  # текст reference после точки разделения
    confidence: float  # уверенность в предложении (0.0-1.0)


@dataclass
class LongSegmentInfo:
    """Информация о длинном сегменте."""

    segment_index: int  # индекс сегмента в STT
    segment: Segment  # сам сегмент
    ref_segments: list["Segment"]  # соответствующие сегменты из reference
    split_suggestions: list[SplitSuggestion]  # предложения разделения


@dataclass
class Config:
    """Конфигурация приложения."""

    stt_file: str = ""  # путь к STT файлу
    ref_file: str = ""  # путь к reference файлу
    out_file: str = ""  # путь для выходного JSON
    time_tol: float = 1.0  # допуск по времени (секунды)
    max_merge: int = 5  # макс. соседних сегментов для склейки
    llm_mode: LLMMode = LLMMode.AUTO
    min_score: float = 0.5  # минимальный порог сходства
    llm_model: str = "meta-llama/llama-3.2-3b-instruct"  # модель для api
    llm_local_model: str = "llama3.2"  # модель для local
    llm_url: str = "http://localhost:11434"  # URL для local
    llm_api_url: str = "https://openrouter.ai/api/v1"  # URL для API
    llm_api_key: Optional[str] = None  # API ключ для api
    llm_site_url: str = "https://github.com/eXpressionist/SubDiffLLM"  # URL сайта для OpenRouter
    llm_site_name: str = "SubDiff"  # Название сайта для OpenRouter
    no_html: bool = False  # не генерировать HTML
    checkpoint: bool = False  # включать периодическое сохранение
    checkpoint_interval: int = 5  # интервал сохранения чекпоинтов
    resume: bool = False  # продолжать с partial checkpoint
    ref_only_missing: bool = False  # режим: искать только то, что есть в REF и отсутствует в STT
    verbose: bool = False  # подробный вывод
    llm_context_size: int = 1  # количество сегментов контекста для LLM
    llm_debug: bool = False  # сохранять запросы/ответы LLM в файл
    llm_debug_file: str = "llm_debug.log"  # путь к debug-логу LLM
    check_long_segments: bool = False  # проверять длинные сегменты
    max_segment_duration: float = 6.0  # макс. длительность сегмента (секунды)
    full_context: bool = False  # режим полного контекста (загрузить все субтитры)
    full_context_max_pairs: int = 50  # макс. пар для batch-запроса в full_context режиме


def ms_to_srt(ms: int) -> str:
    """Конвертировать миллисекунды в формат SRT (HH:MM:SS,mmm)."""
    total_seconds = ms // 1000
    milliseconds = ms % 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def srt_to_ms(time_str: str) -> int:
    """Конвертировать строку времени SRT в миллисекунды."""
    # Формат: HH:MM:SS,mmm или HH:MM:SS.mmm
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split(".")
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
