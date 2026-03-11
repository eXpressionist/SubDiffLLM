"""Тесты фильтрации trivial missing_content."""

from subs_diff.align import merge_segments
from subs_diff.cli import (
    _should_skip_trivial_missing_content,
    _should_keep_issue_in_ref_only_mode,
)
from subs_diff.types import Segment, Candidate, SimilarityMetrics, LLMVerdict, Severity, Category


def _dummy_candidate(a_tokens: list[str], b_tokens: list[str]) -> Candidate:
    a = merge_segments([Segment(index=0, start_ms=0, end_ms=1000, text=" ".join(a_tokens), tokens=a_tokens)])
    b = merge_segments([Segment(index=0, start_ms=0, end_ms=1000, text=" ".join(b_tokens), tokens=b_tokens)])
    metrics = SimilarityMetrics(
        jaccard=0.0,
        char_3gram=0.0,
        levenshtein=0.0,
        length_ratio=1.0,
        rare_token_overlap=1.0,
        rare_token_missing=0.0,
    )
    return Candidate(a_segment=a, b_segment=b, metrics=metrics, is_forced_like=False)


def test_skip_when_only_trivial_tokens_missing():
    candidate = _dummy_candidate(
        ["Нападающая", "Молодец", "Классный", "последний", "удар"],
        ["Ну", "у", "тебя", "был", "отличный", "последний", "удар"],
    )
    verdict = LLMVerdict(
        severity=Severity.MED,
        category=Category.MISSING_CONTENT,
        short_reason="Пропущено 'Ну' в A",
        evidence="",
        confidence=0.8,
    )
    assert _should_skip_trivial_missing_content(candidate, verdict)


def test_do_not_skip_when_meaningful_token_missing():
    candidate = _dummy_candidate(
        ["последний", "удар"],
        ["последний", "удар", "пистолет"],
    )
    verdict = LLMVerdict(
        severity=Severity.MED,
        category=Category.MISSING_CONTENT,
        short_reason="Пропущено важное слово",
        evidence="",
        confidence=0.8,
    )
    assert not _should_skip_trivial_missing_content(candidate, verdict)


def test_skip_false_missing_question_about_existing_topic():
    candidate = _dummy_candidate(
        ["Нравятся", "усы", "Да", "что", "за", "вид"],
        ["Тебе", "нравятся", "мои", "усы", "Ага", "Что", "с", "видом"],
    )
    verdict = LLMVerdict(
        severity=Severity.MED,
        category=Category.MISSING_CONTENT,
        short_reason="Пропущен вопрос о усах",
        evidence="",
        confidence=0.9,
    )
    assert _should_skip_trivial_missing_content(candidate, verdict)


def test_ref_only_mode_keeps_only_missing_and_forced():
    missing_verdict = LLMVerdict(
        severity=Severity.MED,
        category=Category.MISSING_CONTENT,
        short_reason="x",
        evidence="",
        confidence=0.7,
    )
    name_verdict = LLMVerdict(
        severity=Severity.MED,
        category=Category.WRONG_NAMED_ENTITY,
        short_reason="x",
        evidence="",
        confidence=0.7,
    )

    assert _should_keep_issue_in_ref_only_mode(missing_verdict, is_forced_like=False)
    assert not _should_keep_issue_in_ref_only_mode(name_verdict, is_forced_like=False)
    assert _should_keep_issue_in_ref_only_mode(None, is_forced_like=True)
