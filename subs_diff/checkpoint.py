"""Checkpoint persistence for interrupted comparisons."""

import logging
from pathlib import Path

from subs_diff.report import generate_report, load_report_json, save_report_json
from subs_diff.types import Issue

logger = logging.getLogger(__name__)


def save_checkpoint(
    issues: list[Issue],
    out_file: str | Path | None,
    processed: int = 0,
    total: int = 0,
) -> None:
    """Save partial comparison results in the report JSON shape."""
    if out_file is None:
        return

    try:
        report = generate_report(
            issues=issues,
            stt_file="",
            ref_file="",
            config={"partial": True, "processed": processed, "total": total},
        )
        save_report_json(report, out_file)
        logger.info("Чекпоинт сохранён: %s/%s проблем обработано", processed, total)
    except Exception as exc:
        logger.error("Ошибка сохранения чекпоинта: %s", exc)


def load_resume_checkpoint(out_file: str | Path) -> tuple[list[Issue], int]:
    """
    Load a partial checkpoint from a report JSON file.

    Returns:
        A pair of restored issues and processed candidate count. If the file is
        absent or is not a partial checkpoint, returns an empty checkpoint.
    """
    path = Path(out_file)
    if not path.exists():
        return [], 0

    report = load_report_json(path)
    cfg = report.metadata.config
    processed = int(cfg.get("processed", 0)) if cfg.get("partial") else 0
    return report.issues, processed
