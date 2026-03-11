# SubDiff

Клиентское приложение на Python для сравнения двух файлов субтитров в формате `.srt`. Предназначено для выявления ошибок в субтитрах, полученных через speech-to-text (Whisper и аналоги), путём сравнения с эталонными/переведёнными субтитрами.

## Возможности

- **Двухступенчатый алгоритм**: быстрое выравнивание на эвристиках + LLM-верификация
- **Экономия токенов**: по умолчанию LLM вызывается только для подозрительных сегментов (кандидатов)
- **Offline-first**: работает без LLM, используя эвристики (Jaccard, Levenshtein, 3-gram)
- **Детекция именованных сущностей**: приоритет на ошибки в именах, названиях, терминах
- **Устойчивость к разной сегментации**: автоматическая склейка соседних реплик
- **HTML-отчёт для ревью**: компактные карточки инцидентов с полными `A/B` текстами

## Установка

```bash
# Установка зависимостей
pip install -e .

# Или с dev-зависимостями для разработки
pip install -e ".[dev]"
```

### Зависимости

- `rapidfuzz` — быстрые метрики сходства (C++ бэкенд)
- `httpx` — HTTP-клиент для LLM API
- `jinja2` — генерация HTML-отчётов

## Использование

### Базовый вызов

```bash
python -m subs_diff compare --stt audio.srt --ref translated.srt --out report.json
```

Генерирует два файла:
- `report.json` — машинно-читаемый отчёт
- `report.html` — человекочитаемый интерактивный отчёт

### Конфигурация

Чтобы не вводить параметры LLM каждый раз, сохраните их в конфиг:

```bash
# Сохранить настройки
python -m subs_diff config set --api-key sk-... --api-model meta-llama/llama-3.2-3b-instruct

# Показать текущие настройки
python -m subs_diff config show

# Очистить конфиг
python -m subs_diff config clear
```

Конфиг хранится в:
- Windows: `%APPDATA%\SubDiff\config.json` или `~\.subs_diff_config.json`
- Linux/Mac: `~/.config/subs_diff/config.json` или `~/.subs_diff_config.json`

### CLI опции

| Опция | Описание | По умолчанию |
|-------|----------|--------------|
| `--stt` | Путь к SRT от STT (Whisper) | **обязательно** |
| `--ref` | Путь к reference SRT | **обязательно** |
| `--out` | Путь для выходного JSON | **обязательно** |
| `--time-tol` | Допуск по времени (сек) | `1.0` |
| `--max-merge` | Макс. сегментов для склейки | `5` |
| `--min-score` | Порог сходства для кандидата | `0.5` |
| `--llm` | Режим: `off`, `auto`, `local`, `api` | `auto` |
| `--llm-model` | Модель для api (переопределяет конфиг) | из конфига |
| `--llm-local-model` | Модель для local (переопределяет конфиг) | из конфига |
| `--llm-url` | URL для локальной LLM | из конфига |
| `--llm-key` | API ключ (переопределяет конфиг) | из конфига |
| `--llm-api-url` | URL для API LLM | из конфига |
| `--llm-site-url` | URL сайта для OpenRouter headers | из конфига |
| `--llm-site-name` | Название сайта для OpenRouter | из конфига |
| `--no-html` | Не генерировать HTML | `false` |
| `--checkpoint` | Включить автосохранение прогресса | `false` |
| `--checkpoint-interval` | Интервал чекпоинтов (кандидатов) | `5` |
| `--resume` | Продолжить с partial-checkpoint из `--out` | `false` |
| `--ref-only-missing` | Искать только элементы REF, которых нет в STT | `false` |
| `--llm-context-size` | Размер контекста (prev/next) для LLM | `1` |
| `--llm-debug` | Логировать запросы/ответы LLM в читаемый лог | `false` |
| `--llm-debug-file` | Путь к debug-логу LLM | `llm_debug.log` |
| `-v`, `--verbose` | Подробный вывод | `false` |

### Примеры

#### Только эвристики (без LLM)

```bash
python -m subs_diff compare --stt A.srt --ref B.srt --out report.json --llm off
```

#### С локальной LLM (Ollama)

```bash
# Сначала убедитесь, что Ollama запущена и модель загружена
ollama pull llama3.2
ollama serve

python -m subs_diff compare --stt A.srt --ref B.srt --out report.json --llm local
```

#### С API (OpenRouter)

```bash
# Сохраните ключ в конфиг (не нужно вводить каждый раз)
python -m subs_diff config set --api-key sk-or-... --api-model meta-llama/llama-3.2-3b-instruct

# Запустите сравнение (ключ берётся из конфига)
python -m subs_diff compare --stt A.srt --ref B.srt --out report.json --llm api
```

#### Переопределение конфига

```bash
# Использовать другую модель для этого запуска
python -m subs_diff compare --stt A.srt --ref B.srt --out report.json --llm api \
    --llm-model anthropic/claude-3-haiku
```

#### Прогресс и чекпоинты

```bash
# С прогресс-баром и автосохранением каждые 10 кандидатов
python -m subs_diff compare --stt A.srt --ref B.srt --out report.json --llm api \
    --checkpoint --checkpoint-interval 10

# При прерывании (Ctrl+C) результаты сохраняются в report.json
# Можно продолжить с последнего сохранённого места

# Возобновление с чекпоинта
python -m subs_diff compare --stt A.srt --ref B.srt --out report.json --llm api \
    --resume
```

#### Режим "только missing из REF"

```bash
python -m subs_diff compare --stt A.srt --ref B.srt --out report.json --llm api \
    --ref-only-missing
```

#### Пересборка HTML из JSON

```bash
python -m subs_diff.reporter report.json
```

## Формат отчёта

### JSON структура

```json
{
  "metadata": {
    "generated_at": "2026-02-25T10:30:00Z",
    "stt_file": "path/to/A.srt",
    "ref_file": "path/to/B.srt",
    "config": { "time_tol": 1.0, "max_merge": 5, "llm_mode": "auto" }
  },
  "issues": [
    {
      "issue_id": "ISS-001",
      "time_range": { "start_ms": 12500, "end_ms": 15000 },
      "a_segments": [3, 4],
      "b_segments": [5],
      "severity": "high",
      "category": "wrong_named_entity",
      "short_reason": "Имя собственное заменено на другое",
      "evidence": "A: 'Шерлок Холмс' | B: 'Шерлок Стаут'",
      "forced_detected": false,
      "llm_verdict": {
        "confidence": 0.85,
        "suggested_fix": "Шерлок Холмс"
      }
    }
  ],
  "summary": {
    "total_issues": 12,
    "by_severity": { "high": 3, "med": 5, "low": 4 },
    "by_category": { "wrong_named_entity": 5, "missing_content": 4, "other": 3 }
  }
}
```

### Категории проблем

| Категория | Описание |
|-----------|----------|
| `wrong_named_entity` | Ошибка в имени, названии, топониме |
| `missing_content` | Пропущен важный кусок смысла |
| `terminology` | Ошибка в термине/жаргонизме |
| `forced_mismatch` | Расхождение в forced-сегменте (надпись на экране) |
| `other` | Другое расхождение |

### Уровни важности

| Severity | Описание |
|----------|----------|
| `high` | Критичная ошибка (имя, термин, пропуск смысла) |
| `med` | Заметное расхождение |
| `low` | Minor difference, возможно не ошибка |

## Архитектура

```
subs_diff/
├── __init__.py          # Экспорт типов
├── __main__.py          # Entry point для python -m
├── cli.py               # CLI аргументы и main loop
├── types.py             # Dataclasses (Segment, Issue, Config...)
├── parser.py            # SRT парсер, токенизация, нормализация
├── align.py             # Выравнивание, merge сегментов
├── heuristics.py        # Метрики, редкие токены, детекция сущностей
├── llm.py               # LLM адаптер (off/local/api/auto)
├── report.py            # Генерация JSON и HTML отчётов
└── reporter.py          # Утилита пересборки HTML из JSON
```

### Алгоритм работы

1. **Парсинг**: Чтение SRT файлов, токенизация, нормализация
2. **Выравнивание**: Для каждого сегмента A поиск окна B по времени ±tol
3. **Merge**: Проба разных комбинаций склейки (1..max_merge сегментов)
4. **Метрики**: Вычисление Jaccard, Levenshtein, 3-gram, rare token overlap
5. **Кандидаты**: Отбор сегментов по эвристикам (включая ASR-подобные искажения)
6. **LLM верификация**: По умолчанию для кандидатов; в режиме `--ref-only-missing` — расширенная проверка aligned-пар
7. **Отчёт**: Генерация JSON и HTML

## Запуск тестов

```bash
pytest tests/ -v

# С покрытием
pytest tests/ -v --cov=subs_diff --cov-report=html
```

## Токен-экономия LLM

Применяемые техники:

1. **Кандидатный фильтр (по умолчанию)**: в LLM отправляются только кандидаты
2. **Сжатие контекста**: Удаление повторов, лимит 500 символов
3. **JSON-only ответы**: краткий формат без рассуждений
4. **Кэширование**: Ответы кэшируются по хэшу (A_text + B_text)
5. **Лимит токенов**: max_tokens=256, temperature=0.1

## Trade-offs

| Решение | Обоснование |
|---------|-------------|
| Двухступенчатый пайплайн | По умолчанию LLM работает только по кандидатам; можно расширить через `--ref-only-missing` |
| rapidfuzz для метрик | В 10-100x быстрее python-Levenshtein |
| Встроенный CSS в HTML | Один файл для пересылки |
| Merge до 5 сегментов | Компенсация разной сегментации |

## Лицензия

MIT
