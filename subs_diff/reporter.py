"""Утилита для пересборки HTML из готового JSON отчёта."""

import argparse
import sys
from pathlib import Path

from subs_diff.report import generate_html_from_json


def main() -> int:
    """Точка входа для python -m subs_diff.reporter."""
    parser = argparse.ArgumentParser(
        prog="subs-diff-reporter",
        description="Пересобрать HTML отчёт из готового JSON файла",
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Путь к JSON файлу отчёта",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Путь для выходного HTML (по умолчанию: <json_file>.html)",
    )

    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Ошибка: файл не найден: {json_path}", file=sys.stderr)
        return 1

    # Определяем выходной путь
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.with_suffix(".html")

    # Генерируем HTML
    try:
        generate_html_from_json(json_path, output_path)
        print(f"HTML отчёт сохранён: {output_path}")
        return 0
    except Exception as e:
        print(f"Ошибка генерации HTML: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
