"""Tests for LLM response parsing."""

import asyncio

from subs_diff import llm
from subs_diff.align import merge_segments
from subs_diff.llm import FULL_CONTEXT_SYSTEM_PROMPT, APILLLMClient, LocalLLMClient
from subs_diff.types import Candidate, Category, Segment, Severity, SimilarityMetrics


def _candidate() -> Candidate:
    a = merge_segments(
        [
            Segment(
                index=0,
                start_ms=0,
                end_ms=1000,
                text="Sherlock Holmes",
                tokens=["Sherlock", "Holmes"],
            )
        ]
    )
    b = merge_segments(
        [
            Segment(
                index=0,
                start_ms=0,
                end_ms=1000,
                text="Sherlock Stout",
                tokens=["Sherlock", "Stout"],
            )
        ]
    )
    metrics = SimilarityMetrics(
        jaccard=0.0,
        char_3gram=0.0,
        levenshtein=0.0,
        length_ratio=1.0,
        rare_token_overlap=0.0,
        rare_token_missing=1.0,
    )
    return Candidate(a_segment=a, b_segment=b, metrics=metrics)


def test_local_parser_extracts_fenced_json_object():
    client = LocalLLMClient()

    verdict = client._parse_response("""Reasoning text.
```json
{"has_error": true, "severity": "high", "category": "wrong_named_entity",
 "short_reason": "Wrong name", "evidence": "A vs B", "confidence": 0.9}
```
""")

    assert verdict is not None
    assert verdict.severity == Severity.HIGH
    assert verdict.category == Category.WRONG_NAMED_ENTITY
    assert verdict.short_reason == "Wrong name"
    assert verdict.confidence == 0.9


def test_api_parser_skips_malformed_braces_before_json():
    client = APILLLMClient(api_key="test")

    verdict = client._parse_response("""Model note with {not json}.
{"has_error": true, "severity": "med", "category": "missing_content",
 "short_reason": "Missing item", "evidence": "B has extra item"}
""")

    assert verdict is not None
    assert verdict.severity == Severity.MED
    assert verdict.category == Category.MISSING_CONTENT
    assert verdict.short_reason == "Missing item"


def test_api_parser_returns_none_for_no_json():
    client = APILLLMClient(api_key="test")

    assert client._parse_response("No issue found.") is None


def test_batch_parser_extracts_fenced_json_array():
    client = APILLLMClient(api_key="test")

    verdicts = client._parse_batch_response(
        """```json
[
  {"id": 1, "has_error": false},
  {
    "id": 2,
    "has_error": true,
    "severity": "low",
    "category": "other",
    "reason": "Changed meaning"
  }
]
```""",
        expected_count=2,
    )

    assert verdicts[0] is None
    assert verdicts[1] is not None
    assert verdicts[1].severity == Severity.LOW
    assert verdicts[1].category == Category.OTHER
    assert verdicts[1].short_reason == "Changed meaning"


def test_verify_batch_uses_full_context_system_prompt(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "[]"}}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeAsyncClient)

    client = APILLLMClient(api_key="test")
    asyncio.run(client.verify_batch([(_candidate(), None, None)]))

    assert captured["payload"]["messages"][0]["content"] == FULL_CONTEXT_SYSTEM_PROMPT
