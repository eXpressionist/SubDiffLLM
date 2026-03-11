"""Модуль для работы с конфигурационным файлом SubDiff."""

import json
import os
from pathlib import Path
from typing import Optional


# Пути для поиска конфига
CONFIG_LOCATIONS = [
    # Текущая директория
    Path.cwd() / "subs_diff_config.json",
    # Домашняя директория
    Path.home() / ".subs_diff_config.json",
    # AppData (Windows)
    Path(os.environ.get("APPDATA", "")) / "SubDiff" / "config.json" if os.name == "nt" else None,
    # XDG (Linux/Mac)
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "subs_diff" / "config.json" if os.name != "nt" else None,
]


def get_config_path() -> Path:
    """
    Найти существующий конфиг или вернуть путь по умолчанию.

    Returns:
        Путь к конфигурационному файлу.
    """
    # Ищем существующий конфиг
    for loc in CONFIG_LOCATIONS:
        if loc and loc.exists():
            return loc

    # Возвращаем путь для создания нового (в домашней директории)
    return Path.home() / ".subs_diff_config.json"


def load_config() -> dict:
    """
    Загрузить конфигурацию из файла.

    Returns:
        Словарь с конфигурацией или пустой словарь.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load config file: {e}")
        return {}


def save_config(config: dict) -> Path:
    """
    Сохранить конфигурацию в файл.

    Args:
        config: Словарь с конфигурацией.

    Returns:
        Путь к сохранённому файлу.
    """
    config_path = get_config_path()

    # Создаём директорию если нужно
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config_path


def get_llm_config() -> dict:
    """
    Получить LLM-конфигурацию.

    Returns:
        Словарь с LLM настройками.
    """
    config = load_config()
    return config.get("llm", {})


def save_llm_config(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    api_model: Optional[str] = None,
    local_url: Optional[str] = None,
    local_model: Optional[str] = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
) -> Path:
    """
    Сохранить LLM-конфигурацию.

    Args:
        api_key: API ключ.
        api_url: URL API.
        api_model: Модель API.
        local_url: URL локальной LLM.
        local_model: Модель локальной LLM.
        site_url: URL сайта для OpenRouter.
        site_name: Название сайта.

    Returns:
        Путь к сохранённому файлу.
    """
    config = load_config()

    if "llm" not in config:
        config["llm"] = {}

    llm_config = config["llm"]

    if api_key is not None:
        llm_config["api_key"] = api_key
    if api_url is not None:
        llm_config["api_url"] = api_url
    if api_model is not None:
        llm_config["api_model"] = api_model
    if local_url is not None:
        llm_config["local_url"] = local_url
    if local_model is not None:
        llm_config["local_model"] = local_model
    if site_url is not None:
        llm_config["site_url"] = site_url
    if site_name is not None:
        llm_config["site_name"] = site_name

    return save_config(config)


def get_api_key() -> Optional[str]:
    """
    Получить API ключ из конфига или окружения.

    Returns:
        API ключ или None.
    """
    # Сначала пробуем из конфига
    llm_config = get_llm_config()
    if llm_config.get("api_key"):
        return llm_config["api_key"]

    # Затем из окружения
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
