"""LLM адаптер для верификации кандидатов."""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from subs_diff.heuristics import compress_text, detect_named_entities
from subs_diff.types import (
    Candidate,
    Category,
    LLMMode,
    LLMVerdict,
    MergedSegment,
    Severity,
)

logger = logging.getLogger(__name__)


def _extract_json_value(response_text: str, expected_type: type[Any]) -> Any | None:
    """Extract the first JSON value of expected_type from raw model text."""
    if not response_text:
        return None

    starts = "{" if expected_type is dict else "["
    decoder = json.JSONDecoder()
    for idx, char in enumerate(response_text):
        if char != starts:
            continue
        try:
            value, _ = decoder.raw_decode(response_text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, expected_type):
            return value
    return None


def _verdict_from_mapping(
    data: dict[str, Any],
    *,
    reason_key: str = "short_reason",
    default_severity: Severity = Severity.LOW,
    default_confidence: float = 0.5,
    max_reason_chars: int = 100,
) -> LLMVerdict | None:
    """Convert a parsed LLM JSON object into a verdict."""
    if not data.get("has_error", False):
        return None

    severity_map = {
        "low": Severity.LOW,
        "med": Severity.MED,
        "high": Severity.HIGH,
    }
    severity_str = str(data.get("severity", default_severity.value)).lower()
    severity = severity_map.get(severity_str, default_severity)

    category_map = {
        "missing_content": Category.MISSING_CONTENT,
        "wrong_named_entity": Category.WRONG_NAMED_ENTITY,
        "terminology": Category.TERMINOLOGY,
        "forced_mismatch": Category.FORCED_MISMATCH,
        "other": Category.OTHER,
    }
    category_str = str(data.get("category", "other")).lower()
    category = category_map.get(category_str, Category.OTHER)

    reason = (
        data.get(reason_key) or data.get("short_reason") or data.get("reason") or "Unknown issue"
    )
    evidence = data.get("evidence", "")

    return LLMVerdict(
        severity=severity,
        category=category,
        short_reason=str(reason)[:max_reason_chars],
        evidence=str(evidence)[:100],
        confidence=float(data.get("confidence", default_confidence)),
        suggested_fix=data.get("suggested_fix"),
        is_forced=bool(data.get("is_forced", False)),
    )


def _write_llm_debug_event(
    debug_enabled: bool,
    debug_file: str,
    event: dict[str, Any],
) -> None:
    """Сохранить debug-событие LLM в человекочитаемом виде."""
    if not debug_enabled:
        return

    try:
        path = Path(debug_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        event_with_ts = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with path.open("a", encoding="utf-8") as f:
            provider = event_with_ts.get("provider", "unknown")
            stage = event_with_ts.get("stage", "event")
            f.write("=" * 90 + "\n")
            f.write(f"[{event_with_ts['timestamp']}] {provider.upper()} {stage.upper()}\n")
            pretty = json.dumps(event_with_ts, ensure_ascii=False, indent=2)
            f.write(pretty + "\n")
    except Exception as e:
        logger.debug(f"Failed to write LLM debug log: {e}")


# Системный промпт для LLM — строгий, без перевода
SYSTEM_PROMPT = """Ты проверяешь качество субтитров.
Сравни STT-субтитры (A) с reference-субтитрами (B).

КРИТИЧЕСКИ ВАЖНО:
1) НИЧЕГО НЕ ПЕРЕВОДИ.
2) Сравнивай тексты строго в исходном виде.
3) Не перефразируй A и B в ответе.
4) Если вход на русском — reason/evidence тоже на русском.

Проверь ПОСЛЕДОВАТЕЛЬНО:

1. ИМЕНА/НАЗВАНИЯ: Одинаковые ли имена, фамилии, географические названия?
   - Если имя в A отличается от B → wrong_named_entity

2. ТЕРМИНОЛОГИЯ: Одинаковая ли специальная терминология?
   - Если термин в A отличается от B → terminology

3. СМЫСЛ: Передают ли A и B одинаковый смысл?
   - Если смысл противоположный → other (high severity)
   - Если пропущен важный элемент смысла → missing_content

4. СОДЕРЖАНИЕ: Есть ли в B важная информация, которой нет в A?
   - Только если это важный элемент сюжета/диалога, не просто слово

НЕ считать ошибкой:
- другое выражение той же мысли (синонимы)
- грамматические различия
- служебные слова (частицы, предлоги)
- порядок слов

Отвечай ТОЛЬКО валидным JSON.
Если ошибки нет:
{"has_error": false}

Если ошибка есть:
{
  "has_error": true,
  "severity": "low" | "med" | "high",
  "category": "wrong_named_entity" | "terminology" |
              "missing_content" | "forced_mismatch" | "other",
  "short_reason": "Коротко, до 60 символов",
  "evidence": "A: '...' vs B: '...'",
  "confidence": 0.0-1.0
}

Будь консервативен: при сомнении ставь has_error=false."""

# Промпт для сравнения текстов с контекстом
COMPARISON_PROMPT_TEMPLATE = """Сравни эти сегменты субтитров без перевода.

КОНТЕКСТ (предыдущий сегмент A): {prev_context}
КОНТЕКСТ (следующий сегмент A): {next_context}

{forced_guidance}

STT (A): {a_text}
Reference (B): {b_text}

СУЩНОСТИ:
- В A: {entities_a}
- В B: {entities_b}

Проверь:
1) Совпадают ли имена/названия?
2) Совпадает ли смысл (не слова, а содержание)?
3) Есть ли пропуск важной информации?

Ответь JSON."""

# Системный промпт для full-context режима
FULL_CONTEXT_SYSTEM_PROMPT = """Ты проверяешь качество субтитров.
Тебе даны пары STT-субтитров (A) и reference-субтитров (B).

КРИТИЧЕСКИ ВАЖНО:
1) НИЧЕГО НЕ ПЕРЕВОДИ.
2) Сравнивай тексты строго в исходном виде.
3) Не перефразируй A и B в ответе.
4) Если вход на русском — отвечай на русском.

Ищи ТОЛЬКО серьёзные ошибки.

НЕ считать ошибкой:
- другое выражение той же мысли;
- грамматические различия;
- синонимы;
- смену порядка слов.

Считать ошибкой ТОЛЬКО:
- неверные имена/названия;
- неверную терминологию;
- пропуск важного смысла;
- противоположный смысл.

Отвечай ТОЛЬКО валидным JSON-массивом.
Для каждой пары верни объект с полями:
- id: номер пары из входа
- has_error: true/false
- severity: "low"/"med"/"high" (если has_error=true)
- category: "wrong_named_entity"/"terminology"/"missing_content"/"other" (если has_error=true)
- reason: краткое описание (если has_error=true)

Пример ответа:
[
  {"id": 1, "has_error": false},
  {"id": 2, "has_error": true, "severity": "med",
   "category": "missing_content", "reason": "Пропущено имя"}
]

Будь консервативен: при сомнении ставь has_error=false."""

FULL_CONTEXT_PROMPT_TEMPLATE = """Проверь эти пары субтитров на ошибки.

{pairs_text}

Верни JSON-массив с результатом для каждой пары."""


FORCED_LIKE_GUIDANCE = """
REFERENCE-ONLY candidate: alignment did not find a direct STT pair.
Set has_error=true only if Reference is genuinely not covered by STT
or the neighboring STT context.
Do not flag dubbing/paraphrase, titles, short interjections,
or meaning already covered by nearby STT.
"""


class LLMCache:
    """Кэш для LLM-ответов по хэшу текстов."""

    def __init__(self) -> None:
        self._cache: dict[str, LLMVerdict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, a_text: str, b_text: str) -> str:
        """Создать ключ кэша из текстов."""
        combined = f"{a_text}|||{b_text}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, a_text: str, b_text: str) -> LLMVerdict | None:
        """Получить из кэша."""
        key = self._make_key(a_text, b_text)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, a_text: str, b_text: str, verdict: LLMVerdict) -> None:
        """Сохранить в кэш."""
        key = self._make_key(a_text, b_text)
        self._cache[key] = verdict

    @property
    def stats(self) -> dict[str, int]:
        """Статистика кэша."""
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


class LLMClient(ABC):
    """Базовый класс для LLM-клиентов."""

    @abstractmethod
    async def verify(
        self,
        candidate: Candidate,
        prev_segment: MergedSegment | None = None,
        next_segment: MergedSegment | None = None,
    ) -> LLMVerdict | None:
        """
        Верифицировать кандидата через LLM.

        Args:
            candidate: Кандидат на проблему.
            prev_segment: Предыдущий сегмент для контекста.
            next_segment: Следующий сегмент для контекста.

        Returns:
            LLMVerdict или None, если LLM не смогла определить.
        """
        pass

    async def verify_batch(
        self,
        pairs: list[tuple[Candidate, MergedSegment | None, MergedSegment | None]],
    ) -> list[LLMVerdict | None]:
        """
        Верифицировать несколько кандидатов в одном запросе (для full-context режима).

        Args:
            pairs: Список кортежей (candidate, prev_segment, next_segment).

        Returns:
            Список LLMVerdict или None для каждой пары.
        """
        return [await self.verify(c, p, n) for c, p, n in pairs]

    @abstractmethod
    async def is_available(self) -> bool:
        """Проверить доступность LLM."""
        pass


class OffLLMClient(LLMClient):
    """Клиент, который всегда возвращает None (LLM отключена)."""

    async def verify(
        self,
        candidate: Candidate,
        prev_segment: MergedSegment | None = None,
        next_segment: MergedSegment | None = None,
    ) -> LLMVerdict | None:
        return None

    async def is_available(self) -> bool:
        return False


class LocalLLMClient(LLMClient):
    """Клиент для локальной LLM (Ollama, llama.cpp)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 30.0,
        cache: LLMCache | None = None,
        debug_enabled: bool = False,
        debug_file: str = "llm_debug.log",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache = cache or LLMCache()
        self.debug_enabled = debug_enabled
        self.debug_file = debug_file
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Проверить доступность локальной LLM."""
        if self._available is not None:
            return self._available

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Для Ollama проверяем /api/tags
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    self._available = True
                    return True

                # Для llama.cpp сервера проверяем /completion
                response = await client.get(f"{self.base_url}/")
                self._available = response.status_code == 200
                return self._available
        except Exception:
            self._available = False
            return False

    async def verify(
        self,
        candidate: Candidate,
        prev_segment: MergedSegment | None = None,
        next_segment: MergedSegment | None = None,
    ) -> LLMVerdict | None:
        """
        Верифицировать кандидата через локальную LLM.

        Args:
            candidate: Кандидат на проблему.
            prev_segment: Предыдущий сегмент A для контекста.
            next_segment: Следующий сегмент A для контекста.

        Returns:
            LLMVerdict или None, если LLM не смогла определить.
        """
        a_text = compress_text(candidate.a_segment.text, max_chars=500)
        b_text = compress_text(candidate.b_segment.text, max_chars=500)

        # Проверяем кэш
        cached = self.cache.get(a_text, b_text)
        if cached:
            logger.debug("Cache hit for LLM verification")
            return cached

        # Подготовка контекста
        prev_context = compress_text(prev_segment.text, max_chars=200) if prev_segment else "(none)"
        next_context = compress_text(next_segment.text, max_chars=200) if next_segment else "(none)"
        entities_a = detect_named_entities(candidate.a_segment.tokens)
        entities_b = detect_named_entities(candidate.b_segment.tokens)

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            prev_context=prev_context,
            next_context=next_context,
            forced_guidance=FORCED_LIKE_GUIDANCE if candidate.is_forced_like else "",
            a_text=a_text,
            b_text=b_text,
            entities_a=", ".join(entities_a[:8]) if entities_a else "(none)",
            entities_b=", ".join(entities_b[:8]) if entities_b else "(none)",
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Формат запроса для Ollama
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "max_tokens": 256,
                    },
                }
                _write_llm_debug_event(
                    self.debug_enabled,
                    self.debug_file,
                    {
                        "provider": "local",
                        "stage": "request",
                        "url": f"{self.base_url}/api/generate",
                        "payload": payload,
                    },
                )

                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                try:
                    response_payload = response.json()
                except Exception:
                    response_payload = {"raw_text": response.text}
                _write_llm_debug_event(
                    self.debug_enabled,
                    self.debug_file,
                    {
                        "provider": "local",
                        "stage": "response",
                        "status_code": response.status_code,
                        "payload": response_payload,
                    },
                )

                if response.status_code != 200:
                    logger.warning(f"LLM returned status {response.status_code}")
                    return None

                result = response.json()
                response_text = result.get("response", "")

                verdict = self._parse_response(response_text)
                if verdict:
                    self.cache.set(a_text, b_text, verdict)

                return verdict

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return None

    def _parse_response(self, response_text: str) -> LLMVerdict | None:
        """Распарсить JSON-ответ от LLM."""
        if not response_text:
            logger.debug("Empty response text from LLM")
            return None

        data = _extract_json_value(response_text, dict)
        if data is None:
            logger.debug("No JSON object found in LLM response, assuming no error")
            return None

        try:
            return _verdict_from_mapping(data, max_reason_chars=60)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None


class APILLLMClient(LLMClient):
    """Клиент для API LLM (OpenRouter-compatible)."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "meta-llama/llama-3.2-3b-instruct",
        api_key: str | None = None,
        timeout: float = 30.0,
        cache: LLMCache | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        debug_enabled: bool = False,
        debug_file: str = "llm_debug.log",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.cache = cache or LLMCache()
        self.site_url = site_url
        self.site_name = site_name
        self.debug_enabled = debug_enabled
        self.debug_file = debug_file
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Проверить доступность API."""
        if self._available is not None:
            return self._available

        if not self.api_key:
            self._available = False
            return False

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                # OpenRouter: проверяем доступность через простой запрос
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                )
                self._available = response.status_code == 200
                return self._available
        except Exception:
            self._available = False
            return False

    async def verify(
        self,
        candidate: Candidate,
        prev_segment: MergedSegment | None = None,
        next_segment: MergedSegment | None = None,
    ) -> LLMVerdict | None:
        """Верифицировать кандидата через API LLM."""
        a_text = compress_text(candidate.a_segment.text, max_chars=500)
        b_text = compress_text(candidate.b_segment.text, max_chars=500)

        # Проверяем кэш
        cached = self.cache.get(a_text, b_text)
        if cached:
            logger.debug("Cache hit for LLM verification")
            return cached

        # Подготовка контекста
        prev_context = compress_text(prev_segment.text, max_chars=200) if prev_segment else "(none)"
        next_context = compress_text(next_segment.text, max_chars=200) if next_segment else "(none)"
        entities_a = detect_named_entities(candidate.a_segment.tokens)
        entities_b = detect_named_entities(candidate.b_segment.tokens)

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            prev_context=prev_context,
            next_context=next_context,
            forced_guidance=FORCED_LIKE_GUIDANCE if candidate.is_forced_like else "",
            a_text=a_text,
            b_text=b_text,
            entities_a=", ".join(entities_a[:8]) if entities_a else "(none)",
            entities_b=", ".join(entities_b[:8]) if entities_b else "(none)",
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # OpenRouter headers
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url or "https://github.com/eXpressionist/SubDiffLLM",
                    "X-Title": self.site_name or "SubDiff",
                }

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 512,
                }
                _write_llm_debug_event(
                    self.debug_enabled,
                    self.debug_file,
                    {
                        "provider": "api",
                        "stage": "request",
                        "url": f"{self.base_url}/chat/completions",
                        "payload": payload,
                    },
                )

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                try:
                    response_payload = response.json()
                except Exception:
                    response_payload = {"raw_text": response.text}
                _write_llm_debug_event(
                    self.debug_enabled,
                    self.debug_file,
                    {
                        "provider": "api",
                        "stage": "response",
                        "status_code": response.status_code,
                        "payload": response_payload,
                    },
                )

                if response.status_code != 200:
                    logger.warning(f"API returned status {response.status_code}")
                    return None

                result = response.json()

                choices = result.get("choices", [])
                if not choices:
                    logger.warning("API response missing choices")
                    return None

                message = choices[0].get("message", {})
                response_text = message.get("content")

                # Fallback для reasoning моделей (например, Nvidia Nemotron)
                if response_text is None:
                    response_text = message.get("reasoning")

                if response_text is None:
                    logger.warning("API response missing content and reasoning")
                    return None

                verdict = self._parse_response(response_text)
                if verdict:
                    self.cache.set(a_text, b_text, verdict)

                return verdict

        except Exception as e:
            logger.warning(f"API verification failed: {e}")
            return None

    def _parse_response(self, response_text: str) -> LLMVerdict | None:
        """Распарсить JSON-ответ от API LLM."""
        if not response_text:
            logger.debug("Empty response text from API")
            return None

        data = _extract_json_value(response_text, dict)
        if data is None:
            logger.debug("No JSON object found in API response, assuming no error")
            return None

        try:
            return _verdict_from_mapping(data)
        except Exception as e:
            logger.warning(f"Failed to parse API response: {e}")
            return None

    async def verify_batch(
        self,
        pairs: list[tuple[Candidate, MergedSegment | None, MergedSegment | None]],
    ) -> list[LLMVerdict | None]:
        """
        Верифицировать несколько кандидатов в одном запросе (full-context режим).

        Args:
            pairs: Список кортежей (candidate, prev_segment, next_segment).

        Returns:
            Список LLMVerdict или None для каждой пары.
        """
        if not pairs:
            return []

        pairs_text_parts = []
        for i, (candidate, prev_segment, next_segment) in enumerate(pairs, 1):
            a_text = compress_text(candidate.a_segment.text, max_chars=300)
            b_text = compress_text(candidate.b_segment.text, max_chars=300)
            prev_context = compress_text(prev_segment.text, max_chars=100) if prev_segment else "-"
            next_context = compress_text(next_segment.text, max_chars=100) if next_segment else "-"
            forced_note = (
                "REFERENCE-ONLY: verify only if B is not covered by nearby STT.\n"
                if candidate.is_forced_like
                else ""
            )

            pairs_text_parts.append(
                f"ПАРА {i}:\n"
                f"Контекст до: {prev_context}\n"
                f"Контекст после: {next_context}\n"
                f"{forced_note}"
                f"A: {a_text}\n"
                f"B: {b_text}"
            )

        pairs_text = "\n\n".join(pairs_text_parts)
        prompt = FULL_CONTEXT_PROMPT_TEMPLATE.format(pairs_text=pairs_text)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url or "https://github.com/subdiff",
                    "X-Title": self.site_name or "SubDiff",
                }

                messages = [
                    {"role": "system", "content": FULL_CONTEXT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 512,
                }
                _write_llm_debug_event(
                    self.debug_enabled,
                    self.debug_file,
                    {
                        "provider": "api",
                        "stage": "request",
                        "url": f"{self.base_url}/chat/completions",
                        "payload": {
                            "model": payload["model"],
                            "messages_count": len(messages),
                        },
                    },
                )

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                try:
                    response_payload = response.json()
                except Exception:
                    response_payload = {"raw_text": response.text}

                _write_llm_debug_event(
                    self.debug_enabled,
                    self.debug_file,
                    {
                        "provider": "api",
                        "stage": "response",
                        "status_code": response.status_code,
                        "payload": response_payload,
                    },
                )

                if response.status_code != 200:
                    logger.warning(f"API returned status {response.status_code}")
                    return [None] * len(pairs)

                result = response.json()
                choices = result.get("choices", [])
                if not choices:
                    logger.warning("API response missing choices")
                    return [None] * len(pairs)

                message = choices[0].get("message", {})
                response_text = message.get("content") or message.get("reasoning")

                if response_text is None:
                    logger.warning("API response missing content and reasoning")
                    return [None] * len(pairs)

                return self._parse_batch_response(response_text, len(pairs))

        except Exception as e:
            logger.warning(f"API batch verification failed: {e}")
            return [None] * len(pairs)

    def _parse_batch_response(
        self, response_text: str, expected_count: int
    ) -> list[LLMVerdict | None]:
        """Распарсить JSON-массив ответов от API LLM."""
        results: list[LLMVerdict | None] = [None] * expected_count

        if not response_text:
            return results

        data = _extract_json_value(response_text, list)
        if data is None:
            return results

        try:
            for item in data:
                if not isinstance(item, dict):
                    continue
                idx = item.get("id", 0)
                if idx < 1 or idx > expected_count:
                    continue

                has_error = item.get("has_error", False)
                if not has_error:
                    results[idx - 1] = None
                    continue

                results[idx - 1] = _verdict_from_mapping(
                    item,
                    reason_key="reason",
                    default_confidence=0.7,
                )

            return results

        except Exception as e:
            logger.warning(f"Failed to parse batch API response: {e}")
            return results


class AutoLLMClient(LLMClient):
    """Клиент, который пытается использовать local, затем api, иначе off."""

    def __init__(
        self,
        local_url: str = "http://localhost:11434",
        local_model: str = "llama3.2",
        api_url: str = "https://openrouter.ai/api/v1",
        api_model: str = "meta-llama/llama-3.2-3b-instruct",
        api_key: str | None = None,
        cache: LLMCache | None = None,
        debug_enabled: bool = False,
        debug_file: str = "llm_debug.log",
    ):
        self.local_client = LocalLLMClient(
            base_url=local_url,
            model=local_model,
            cache=cache,
            debug_enabled=debug_enabled,
            debug_file=debug_file,
        )
        self.api_client = APILLLMClient(
            base_url=api_url,
            model=api_model,
            api_key=api_key,
            cache=cache,
            debug_enabled=debug_enabled,
            debug_file=debug_file,
        )
        self._selected_client: LLMClient | None = None

    async def is_available(self) -> bool:
        """Проверить доступность любого LLM."""
        if self._selected_client:
            return await self._selected_client.is_available()

        # Пытаемся local
        if await self.local_client.is_available():
            self._selected_client = self.local_client
            logger.info("Using local LLM")
            return True

        # Пытаемся api
        if await self.api_client.is_available():
            self._selected_client = self.api_client
            logger.info("Using API LLM")
            return True

        # Никакой не доступен
        self._selected_client = OffLLMClient()
        return False

    async def verify(
        self,
        candidate: Candidate,
        prev_segment: MergedSegment | None = None,
        next_segment: MergedSegment | None = None,
    ) -> LLMVerdict | None:
        """Верифицировать кандидата через доступный LLM."""
        if self._selected_client is None:
            await self.is_available()
        assert self._selected_client is not None

        return await self._selected_client.verify(
            candidate, prev_segment=prev_segment, next_segment=next_segment
        )

    async def verify_batch(
        self,
        pairs: list[tuple[Candidate, MergedSegment | None, MergedSegment | None]],
    ) -> list[LLMVerdict | None]:
        """Верифицировать несколько кандидатов через доступный LLM."""
        if self._selected_client is None:
            await self.is_available()
        assert self._selected_client is not None

        return await self._selected_client.verify_batch(pairs)


def create_llm_client(
    mode: LLMMode,
    local_url: str = "http://localhost:11434",
    local_model: str = "llama3.2",
    api_url: str = "https://openrouter.ai/api/v1",
    api_model: str = "meta-llama/llama-3.2-3b-instruct",
    api_key: str | None = None,
    site_url: str | None = None,
    site_name: str | None = None,
    debug_enabled: bool = False,
    debug_file: str = "llm_debug.log",
) -> LLMClient:
    """
    Создать LLM-клиент в зависимости от режима.

    Args:
        mode: Режим работы (off/auto/local/api).
        local_url: URL для локальной LLM.
        local_model: Модель для локальной LLM.
        api_url: URL для API LLM (OpenRouter).
        api_model: Модель для API LLM.
        api_key: API ключ для API LLM.
        site_url: URL сайта для OpenRouter headers.
        site_name: Название сайта для OpenRouter headers.

    Returns:
        LLMClient соответствующего типа.
    """
    cache = LLMCache()

    if mode == LLMMode.OFF:
        return OffLLMClient()
    elif mode == LLMMode.LOCAL:
        return LocalLLMClient(
            base_url=local_url,
            model=local_model,
            cache=cache,
            debug_enabled=debug_enabled,
            debug_file=debug_file,
        )
    elif mode == LLMMode.API:
        return APILLLMClient(
            base_url=api_url,
            model=api_model,
            api_key=api_key,
            cache=cache,
            site_url=site_url,
            site_name=site_name,
            debug_enabled=debug_enabled,
            debug_file=debug_file,
        )
    else:  # AUTO
        return AutoLLMClient(
            local_url=local_url,
            local_model=local_model,
            api_url=api_url,
            api_model=api_model,
            api_key=api_key,
            cache=cache,
            debug_enabled=debug_enabled,
            debug_file=debug_file,
        )


async def verify_candidates(
    candidates: list[Candidate],
    llm_client: LLMClient,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[tuple[Candidate, LLMVerdict | None]]:
    """
    Верифицировать список кандидатов через LLM.

    Args:
        candidates: Список кандидатов.
        llm_client: LLM-клиент.
        progress_callback: Опционально, callback для прогресса.

    Returns:
        Список пар (кандидат, вердикт).
    """
    results = []

    for i, candidate in enumerate(candidates):
        verdict = await llm_client.verify(candidate)
        results.append((candidate, verdict))

        if progress_callback:
            progress_callback(i + 1, len(candidates))

    return results
