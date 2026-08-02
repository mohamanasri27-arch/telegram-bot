"""Runtime settings the user can change from Telegram, persisted across restarts.

Kept in a small JSON file next to the code rather than in .env, so changing a
setting never risks rewriting the user's bot token.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path(__file__).with_name("settings.json")

MODE_PLAIN = "plain"
MODE_PROMPT = "prompt"

ACCURACY_FAST = "large-v3-turbo"
ACCURACY_HIGH = "large-v3"

DEFAULTS = {
    "mode": MODE_PLAIN,
    "model": ACCURACY_HIGH,
    "clean_fillers": True,
}


def load() -> dict:
    if not SETTINGS_FILE.exists():
        return dict(DEFAULTS)
    try:
        stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("settings.json is unreadable; falling back to defaults")
        return dict(DEFAULTS)
    return {**DEFAULTS, **stored}


def save(settings: dict) -> None:
    try:
        SETTINGS_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        logger.exception("Could not write settings.json; change will not survive restart")


def get(key: str):
    return load().get(key, DEFAULTS.get(key))


def set_value(key: str, value) -> dict:
    settings = load()
    settings[key] = value
    save(settings)
    return settings
