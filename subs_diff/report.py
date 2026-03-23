"""Генерация отчётов: JSON и HTML."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

from subs_diff.types import (
    Issue,
    Report,
    ReportMetadata,
    ReportSummary,
    Severity,
    Category,
    Candidate,
    LLMVerdict,
    ms_to_srt,
)


# HTML-шаблон отчёта со встроенными стилями
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SubDiff Report — {{ metadata.stt_file }}</title>
    <style>
        :root {
            --bg: #f6f7fb;
            --card: #ffffff;
            --text: #1d2433;
            --muted: #5b6474;
            --line: #e5e8f0;
            --high: #c62828;
            --med: #ef6c00;
            --low: #1565c0;
        }

        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: "Segoe UI", Tahoma, sans-serif;
            line-height: 1.45;
        }

        .container {
            max-width: 980px;
            margin: 24px auto;
            padding: 0 12px 24px;
        }

        .header, .issue, .empty {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 10px;
        }

        .header {
            padding: 14px 16px;
            margin-bottom: 12px;
        }

        .title {
            margin: 0 0 8px 0;
            font-size: 20px;
        }

        .meta {
            font-size: 13px;
            color: var(--muted);
            display: grid;
            gap: 4px;
        }

        .issue {
            margin-bottom: 10px;
            padding: 12px;
        }

        .row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 12px;
            align-items: center;
            margin-bottom: 8px;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 12px;
            border: 1px solid var(--line);
            background: #fff;
        }

        .sev-high { color: var(--high); border-color: #ef9a9a; }
        .sev-med { color: var(--med); border-color: #ffcc80; }
        .sev-low { color: var(--low); border-color: #90caf9; }

        .cat-long-segment { background: #fff3e0; border-color: #ffcc80; }

        .time { font-family: Consolas, monospace; color: var(--muted); font-size: 13px; }
        .reason { font-weight: 600; margin: 6px 0; }

        .block-title {
            margin: 10px 0 4px;
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .02em;
        }

        .text-block {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid var(--line);
            background: #fbfcff;
            border-radius: 8px;
            padding: 10px;
            font-size: 14px;
        }

        details summary {
            cursor: pointer;
            color: var(--muted);
            font-size: 13px;
            margin-top: 8px;
        }

        .empty {
            padding: 24px;
            text-align: center;
            color: var(--muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">SubDiff Report</h1>
            <div class="meta">
                <div><strong>STT:</strong> {{ metadata.stt_file }}</div>
                <div><strong>Reference:</strong> {{ metadata.ref_file }}</div>
                <div><strong>Сгенерирован:</strong> {{ metadata.generated_at }}</div>
                <div><strong>Проблем:</strong> {{ summary.total_issues }}</div>
            </div>
        </div>

        {% if summary.total_issues > 0 %}
            {% for issue in issues %}
            <article class="issue">
                <div class="row">
                    <span class="time">{{ issue.time_range.0 | ms_to_time }} → {{ issue.time_range.1 | ms_to_time }}</span>
                    <span class="badge sev-{{ issue.severity }}">{{ issue.severity }}</span>
                    <span class="badge">{{ issue.category }}</span>
                    {% if issue.forced_detected %}<span class="badge">forced</span>{% endif %}
                </div>

                <div class="reason">{{ issue.short_reason }}</div>

                {% if issue.category == "long_segment" %}
                <div class="block-title" style="margin-top: 8px;">Предложения разделения</div>
                <pre class="text-block" style="background: #fff8e1;">{{ issue.evidence if issue.evidence else "(нет предложений)" }}</pre>
                {% endif %}

                <div class="block-title">STT (A)</div>
                <pre class="text-block">{{ issue.a_text if issue.a_text else "(пусто)" }}</pre>

                <div class="block-title">Reference (B)</div>
                <pre class="text-block">{{ issue.b_text if issue.b_text else "(пусто)" }}</pre>

                <details>
                    <summary>Показать детали</summary>
                    <div class="block-title">Evidence</div>
                    <pre class="text-block">{{ issue.evidence if issue.evidence else "(нет)" }}</pre>
                    {% if issue.llm_verdict %}
                    <div class="block-title">LLM</div>
                    <pre class="text-block">confidence={{ issue.llm_verdict.confidence }}, category={{ issue.llm_verdict.category }}, severity={{ issue.llm_verdict.severity }}</pre>
                    {% endif %}
                </details>
            </article>
            {% endfor %}
        {% else %}
            <div class="empty">Проблем не обнаружено.</div>
        {% endif %}
    </div>
</body>
</html>
"""


def ms_to_time_filter(ms: int) -> str:
    """Jinja фильтр для конвертации ms в HH:MM:SS,mmm."""
    return ms_to_srt(ms)


def generate_report(
    issues: list[Issue],
    stt_file: str,
    ref_file: str,
    config: dict,
) -> Report:
    """
    Сгенерировать полный отчёт.

    Args:
        issues: Список проблем.
        stt_file: Путь к STT файлу.
        ref_file: Путь к reference файлу.
        config: Конфигурация.

    Returns:
        Report объект.
    """
    # Считаем сводку
    by_severity: dict[str, int] = {"high": 0, "med": 0, "low": 0}
    by_category: dict[str, int] = {}

    for issue in issues:
        by_severity[issue.severity.value] = by_severity.get(issue.severity.value, 0) + 1
        cat_key = issue.category.value
        by_category[cat_key] = by_category.get(cat_key, 0) + 1

    metadata = ReportMetadata(
        generated_at=datetime.now(timezone.utc).isoformat(),
        stt_file=stt_file,
        ref_file=ref_file,
        config=config,
    )

    summary = ReportSummary(
        total_issues=len(issues),
        by_severity=by_severity,
        by_category=by_category,
    )

    return Report(
        metadata=metadata,
        issues=issues,
        summary=summary,
    )


def report_to_dict(report: Report) -> dict[str, Any]:
    """
    Конвертировать Report в словарь для JSON сериализации.

    Args:
        report: Report объект.

    Returns:
        Словарь для JSON.
    """
    return {
        "metadata": {
            "generated_at": report.metadata.generated_at,
            "stt_file": report.metadata.stt_file,
            "ref_file": report.metadata.ref_file,
            "config": report.metadata.config,
        },
        "issues": [
            {
                "issue_id": issue.issue_id,
                "time_range": {
                    "start_ms": issue.time_range[0],
                    "end_ms": issue.time_range[1],
                },
                "a_segments": issue.a_segments,
                "b_segments": issue.b_segments,
                "severity": issue.severity.value,
                "category": issue.category.value,
                "short_reason": issue.short_reason,
                "evidence": issue.evidence,
                "forced_detected": issue.forced_detected,
                "llm_verdict": (
                    {
                        "severity": issue.llm_verdict.severity.value,
                        "category": issue.llm_verdict.category.value,
                        "short_reason": issue.llm_verdict.short_reason,
                        "evidence": issue.llm_verdict.evidence,
                        "confidence": issue.llm_verdict.confidence,
                        "suggested_fix": issue.llm_verdict.suggested_fix,
                        "is_forced": issue.llm_verdict.is_forced,
                    }
                    if issue.llm_verdict
                    else None
                ),
                "a_text": issue.a_text,
                "b_text": issue.b_text,
            }
            for issue in report.issues
        ],
        "summary": {
            "total_issues": report.summary.total_issues,
            "by_severity": report.summary.by_severity,
            "by_category": report.summary.by_category,
        },
    }


def save_report_json(report: Report, filepath: str | Path) -> None:
    """
    Сохранить отчёт в JSON файл.

    Args:
        report: Report объект.
        filepath: Путь для сохранения.
    """
    path = Path(filepath)
    data = report_to_dict(report)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_report_html(report: Report, filepath: str | Path) -> None:
    """
    Сохранить отчёт в HTML файл.

    Args:
        report: Report объект.
        filepath: Путь для сохранения.
    """
    from jinja2 import Environment

    path = Path(filepath)
    env = Environment()
    env.filters["ms_to_time"] = ms_to_time_filter
    template = env.from_string(HTML_TEMPLATE)

    html_content = template.render(
        metadata=report.metadata,
        issues=report.issues,
        summary=report.summary,
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(html_content)


def generate_html_from_json(json_filepath: str | Path, output_html: str | Path | None = None) -> str:
    """
    Сгенерировать HTML из готового JSON файла.

    Args:
        json_filepath: Путь к JSON файлу.
        output_html: Опционально, путь для сохранения HTML.

    Returns:
        HTML содержимое.
    """
    json_path = Path(json_filepath)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Восстанавливаем объекты из словаря
    issues = []
    for issue_data in data["issues"]:
        llm_verdict = None
        if issue_data.get("llm_verdict"):
            v = issue_data["llm_verdict"]
            llm_verdict = LLMVerdict(
                severity=Severity(v["severity"]),
                category=Category(v["category"]),
                short_reason=v["short_reason"],
                evidence=v["evidence"],
                confidence=v["confidence"],
                suggested_fix=v.get("suggested_fix"),
                is_forced=v.get("is_forced", False),
            )

        issue = Issue(
            issue_id=issue_data["issue_id"],
            time_range=(
                issue_data["time_range"]["start_ms"],
                issue_data["time_range"]["end_ms"],
            ),
            a_segments=issue_data["a_segments"],
            b_segments=issue_data["b_segments"],
            severity=Severity(issue_data["severity"]),
            category=Category(issue_data["category"]),
            short_reason=issue_data["short_reason"],
            evidence=issue_data["evidence"],
            forced_detected=issue_data["forced_detected"],
            llm_verdict=llm_verdict,
            a_text=issue_data.get("a_text", ""),
            b_text=issue_data.get("b_text", ""),
        )
        issues.append(issue)

    metadata = ReportMetadata(
        generated_at=data["metadata"]["generated_at"],
        stt_file=data["metadata"]["stt_file"],
        ref_file=data["metadata"]["ref_file"],
        config=data["metadata"]["config"],
    )

    summary = ReportSummary(
        total_issues=data["summary"]["total_issues"],
        by_severity=data["summary"]["by_severity"],
        by_category=data["summary"]["by_category"],
    )

    report = Report(
        metadata=metadata,
        issues=issues,
        summary=summary,
    )

    # Генерируем HTML
    from jinja2 import Environment

    env = Environment()
    env.filters["ms_to_time"] = ms_to_time_filter
    template = env.from_string(HTML_TEMPLATE)

    html_content = template.render(
        metadata=report.metadata,
        issues=report.issues,
        summary=report.summary,
    )

    if output_html:
        out_path = Path(output_html)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


def create_issue_from_candidate(
    candidate: Candidate,
    issue_id: str,
    llm_verdict: Optional[LLMVerdict] = None,
) -> Issue:
    """
    Создать Issue из Candidate и LLM вердикта.

    Args:
        candidate: Кандидат на проблему.
        issue_id: Уникальный ID проблемы.
        llm_verdict: Опционально, вердикт от LLM.

    Returns:
        Issue объект.
    """
    # Проверка на forced: в reference есть текст, а в STT ничего нет
    is_forced = candidate.is_forced_like or (
        len(candidate.a_segment.indices) == 0 and len(candidate.b_segment.indices) > 0
    )

    # Если есть LLM вердикт — используем его
    if llm_verdict:
        return Issue(
            issue_id=issue_id,
            time_range=(candidate.a_segment.start_ms, candidate.a_segment.end_ms),
            a_segments=candidate.a_segment.indices,
            b_segments=candidate.b_segment.indices,
            severity=llm_verdict.severity,
            category=llm_verdict.category,
            short_reason=llm_verdict.short_reason,
            evidence=llm_verdict.evidence,
            forced_detected=is_forced or llm_verdict.is_forced,
            llm_verdict=llm_verdict,
            a_text=candidate.a_segment.text,
            b_text=candidate.b_segment.text,
        )

    # Для forced сегментов (без LLM)
    if is_forced:
        time_range = (candidate.b_segment.start_ms, candidate.b_segment.end_ms)
        return Issue(
            issue_id=issue_id,
            time_range=time_range,
            a_segments=[],
            b_segments=candidate.b_segment.indices,
            severity=Severity.MED,
            category=Category.FORCED_MISMATCH,
            short_reason="Текст в reference без соответствия в STT (forced)",
            evidence=f"B: '{candidate.b_segment.text[:50]}...'" if candidate.b_segment.text else "",
            forced_detected=True,
            llm_verdict=None,
            a_text="",
            b_text=candidate.b_segment.text,
        )

    # По умолчанию (не должно вызываться при новой логике)
    time_range = (candidate.a_segment.start_ms, candidate.a_segment.end_ms)
    return Issue(
        issue_id=issue_id,
        time_range=time_range,
        a_segments=candidate.a_segment.indices,
        b_segments=candidate.b_segment.indices,
        severity=Severity.LOW,
        category=Category.OTHER,
        short_reason="No issue detected",
        evidence="",
        forced_detected=False,
        llm_verdict=None,
        a_text=candidate.a_segment.text,
        b_text=candidate.b_segment.text,
    )
