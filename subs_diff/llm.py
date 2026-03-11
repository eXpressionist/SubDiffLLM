"""LLM адаптер для верификации кандидатов."""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from subs_diff.types import (
    Candidate,
    LLMVerdict,
    Severity,
    Category,
    LLMMode,
    MergedSegment,
)
from subs_diff.heuristics import compress_text
from subs_diff.heuristics import find_missing_content, detect_named_entities


logger = logging.getLogger(__name__)


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
- противоположный смысл;
- forced-сегмент (в B есть реплика без соответствия в A).

Отвечай ТОЛЬКО валидным JSON.
Если ошибки нет:
{"has_error": false}

Если ошибка есть:
{
  "has_error": true,
  "severity": "low" | "med" | "high",
  "category": "wrong_named_entity" | "terminology" | "missing_content" | "forced_mismatch" | "other",
  "short_reason": "Коротко, до 60 символов",
  "evidence": "A: '...' vs B: '...'",
  "confidence": 0.0-1.0
}

Будь консервативен: при сомнении ставь has_error=false."""

# Промпт для сравнения текстов с контекстом
COMPARISON_PROMPT_TEMPLATE = """Сравни эти сегменты субтитров без перевода.

КОНТЕКСТ (предыдущий сегмент A): {prev_context}
КОНТЕКСТ (следующий сегмент A): {next_context}

STT (A): {a_text}
Reference (B): {b_text}
ЭВРИСТИКА: ключевые токены B, которых нет в A: {missing_tokens}
ЭВРИСТИКА: сущности в A: {entities_a}
ЭВРИСТИКА: сущности в B: {entities_b}

Определи, есть ли серьёзная ошибка по смыслу."""


class LLMCache:
    """Кэш для LLM-ответов по хэшу текстов."""

    def __init__(self):
        self._cache: dict[str, LLMVerdict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, a_text: str, b_text: str) -> str:
        """Создать ключ кэша из текстов."""
        combined = f"{a_text}|||{b_text}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, a_text: str, b_text: str) -> Optional[LLMVerdict]:
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
        prev_segment: Optional[MergedSegment] = None,
        next_segment: Optional[MergedSegment] = None,
    ) -> Optional[LLMVerdict]:
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

    @abstractmethod
    async def is_available(self) -> bool:
        """Проверить доступность LLM."""
        pass


class OffLLMClient(LLMClient):
    """Клиент, который всегда возвращает None (LLM отключена)."""

    async def verify(
        self,
        candidate: Candidate,
        prev_segment: Optional[MergedSegment] = None,
        next_segment: Optional[MergedSegment] = None,
    ) -> Optional[LLMVerdict]:
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
        cache: Optional[LLMCache] = None,
        debug_enabled: bool = False,
        debug_file: str = "llm_debug.log",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache = cache or LLMCache()
        self.debug_enabled = debug_enabled
        self.debug_file = debug_file
        self._available: Optional[bool] = None

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
        prev_segment: Optional[MergedSegment] = None,
        next_segment: Optional[MergedSegment] = None,
    ) -> Optional[LLMVerdict]:
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
        missing_tokens = find_missing_content(
            candidate.a_segment.tokens, candidate.b_segment.tokens, min_length=5
        )
        entities_a = detect_named_entities(candidate.a_segment.tokens)
        entities_b = detect_named_entities(candidate.b_segment.tokens)

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            prev_context=prev_context,
            next_context=next_context,
            a_text=a_text,
            b_text=b_text,
            missing_tokens=", ".join(missing_tokens[:12]) if missing_tokens else "(none)",
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

    def _parse_response(self, response_text: str) -> Optional[LLMVerdict]:
        """Распарсить JSON-ответ от LLM."""
        try:
            # Пытаемся найти JSON в ответе
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON found in LLM response")
                return None

            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)

            # Новый формат: проверяем has_error
            has_error = data.get("has_error", False)
            if not has_error:
                return None  # Нет ошибки

            # Парсим severity
            severity_str = data.get("severity", "low").lower()
            severity_map = {
                "low": Severity.LOW,
                "med": Severity.MED,
                "high": Severity.HIGH,
            }
            severity = severity_map.get(severity_str, Severity.LOW)

            # Парсим категорию
            category_str = data.get("category", "other").lower()
            category_map = {
                "wrong_named_entity": Category.WRONG_NAMED_ENTITY,
                "terminology": Category.TERMINOLOGY,
                "missing_content": Category.MISSING_CONTENT,
                "forced_mismatch": Category.FORCED_MISMATCH,
                "other": Category.OTHER,
            }
            category = category_map.get(category_str, Category.OTHER)

            return LLMVerdict(
                severity=severity,
                category=category,
                short_reason=data.get("short_reason", "Unknown issue")[:60],
                evidence=data.get("evidence", "")[:100],
                confidence=float(data.get("confidence", 0.5)),
                suggested_fix=data.get("suggested_fix"),
                is_forced=bool(data.get("is_forced", False)),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None


class APILLLMClient(LLMClient):
    """Клиент для API LLM (OpenRouter-compatible)."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "meta-llama/llama-3.2-3b-instruct",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        cache: Optional[LLMCache] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
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
        self._available: Optional[bool] = None

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
        prev_segment: Optional[MergedSegment] = None,
        next_segment: Optional[MergedSegment] = None,
    ) -> Optional[LLMVerdict]:
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
        missing_tokens = find_missing_content(
            candidate.a_segment.tokens, candidate.b_segment.tokens, min_length=5
        )
        entities_a = detect_named_entities(candidate.a_segment.tokens)
        entities_b = detect_named_entities(candidate.b_segment.tokens)

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            prev_context=prev_context,
            next_context=next_context,
            a_text=a_text,
            b_text=b_text,
            missing_tokens=", ".join(missing_tokens[:12]) if missing_tokens else "(none)",
            entities_a=", ".join(entities_a[:8]) if entities_a else "(none)",
            entities_b=", ".join(entities_b[:8]) if entities_b else "(none)",
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # OpenRouter headers
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url or "https://github.com/subdiff",
                    "X-Title": self.site_name or "SubDiff",
                }

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 256,
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
                response_text = result["choices"][0]["message"]["content"]

                verdict = self._parse_response(response_text)
                if verdict:
                    self.cache.set(a_text, b_text, verdict)

                return verdict

        except Exception as e:
            logger.warning(f"API verification failed: {e}")
            return None

    def _parse_response(self, response_text: str) -> Optional[LLMVerdict]:
        """Распарсить JSON-ответ от API LLM."""
        try:
            data = json.loads(response_text)

            has_error = data.get("has_error")
            if has_error is False:
                return None

            severity_str = data.get("severity", "none").lower()
            severity_map = {
                "none": None,
                "low": Severity.LOW,
                "med": Severity.MED,
                "high": Severity.HIGH,
            }
            severity = severity_map.get(severity_str)

            if severity is None:
                return None

            category_str = data.get("category", "other").lower()
            category_map = {
                "none": Category.OTHER,
                "missing_content": Category.MISSING_CONTENT,
                "wrong_named_entity": Category.WRONG_NAMED_ENTITY,
                "terminology": Category.TERMINOLOGY,
                "forced_mismatch": Category.FORCED_MISMATCH,
                "other": Category.OTHER,
            }
            category = category_map.get(category_str, Category.OTHER)

            return LLMVerdict(
                severity=severity,
                category=category,
                short_reason=data.get("short_reason", "Unknown issue")[:100],
                evidence=data.get("evidence", "")[:100],
                confidence=float(data.get("confidence", 0.5)),
                suggested_fix=data.get("suggested_fix"),
                is_forced=bool(data.get("is_forced", False)),
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse API JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse API response: {e}")
            return None


class AutoLLMClient(LLMClient):
    """Клиент, который пытается использовать local, затем api, иначе off."""

    def __init__(
        self,
        local_url: str = "http://localhost:11434",
        local_model: str = "llama3.2",
        api_url: str = "https://openrouter.ai/api/v1",
        api_model: str = "meta-llama/llama-3.2-3b-instruct",
        api_key: Optional[str] = None,
        cache: Optional[LLMCache] = None,
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
        self._selected_client: Optional[LLMClient] = None

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
        prev_segment: Optional[MergedSegment] = None,
        next_segment: Optional[MergedSegment] = None,
    ) -> Optional[LLMVerdict]:
        """Верифицировать кандидата через доступный LLM."""
        if self._selected_client is None:
            await self.is_available()

        return await self._selected_client.verify(
            candidate, prev_segment=prev_segment, next_segment=next_segment
        )


def create_llm_client(
    mode: LLMMode,
    local_url: str = "http://localhost:11434",
    local_model: str = "llama3.2",
    api_url: str = "https://openrouter.ai/api/v1",
    api_model: str = "meta-llama/llama-3.2-3b-instruct",
    api_key: Optional[str] = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
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
            base_url=api_url, model=api_model, api_key=api_key, cache=cache,
            site_url=site_url, site_name=site_name,
            debug_enabled=debug_enabled, debug_file=debug_file
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
    progress_callback: Optional[callable] = None,
) -> list[tuple[Candidate, Optional[LLMVerdict]]]:
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
