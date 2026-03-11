"""SubDiff — клиент для сравнения двух файлов субтитров в формате .srt."""

__version__ = "0.1.0"

from subs_diff.types import (
    Segment,
    Candidate,
    Issue,
    Severity,
    Category,
    LLMVerdict,
    Report,
    Config,
)

__all__ = [
    "Segment",
    "Candidate",
    "Issue",
    "Severity",
    "Category",
    "LLMVerdict",
    "Report",
    "Config",
]
