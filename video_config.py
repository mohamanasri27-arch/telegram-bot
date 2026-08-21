"""Editing preferences for the video pipeline, persisted in video_settings.json.

Kept separate from settings.py so the video editor and the Telegram bot can
never overwrite each other's file. Every key has a default here, so a missing
or half-written JSON file still produces a working editor.
"""
import copy
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).with_name("video_settings.json")

PROJECT_DIR = Path(__file__).parent
INBOX_DIR = PROJECT_DIR / "videos-in"
OUTBOX_DIR = PROJECT_DIR / "videos-out"
ASSETS_DIR = PROJECT_DIR / "assets"

DEFAULTS = {
    "language": "fa",

    "cleanup": {
        # Cut the dead air out of the middle of the video.
        "remove_silence": True,
        # Anything quieter than this counts as silence. Raise it (towards 0)
        # if quiet speech is being cut, lower it if pauses are being kept.
        "silence_threshold_db": -34.0,
        # Ignore pauses shorter than this: they are natural speech rhythm.
        "min_silence_seconds": 0.8,
        # Leave this much of each pause in place, so cuts do not sound abrupt.
        "keep_padding_seconds": 0.18,
        # Drop the run-up and the run-out where nothing is happening yet.
        "trim_start": True,
        "trim_end": True,
        # Even out the volume so quiet videos are not unwatchable.
        "normalize_audio": True,
        "target_loudness_lufs": -14.0,
        # A light lift, not a look: slight contrast and saturation only.
        "color_polish": True,
    },

    "subtitles": {
        "enabled": True,
        # Which language gets burned onto the picture:
        # "fa", "en", "both", or "none" (SRT files are still written).
        "burn": "fa",
        # Font must be installed on this computer. Tahoma ships with Windows
        # and covers Persian. Vazirmatn looks better if you install it.
        "font": "Tahoma",
        "english_font": "Arial",
        # Percentage of the video height, so it scales with the resolution.
        "font_size_pct": 5.2,
        "english_font_size_pct": 3.2,
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 2.6,
        "shadow": 0.8,
        # Distance from the bottom edge, as a percentage of video height.
        "margin_bottom_pct": 7.0,
        # Cue shaping: shorter cues are easier to read than whole sentences.
        "max_chars_per_line": 42,
        "max_lines": 2,
        "max_cue_seconds": 4.5,
        # Write .srt files next to the video as well as burning them in.
        "write_srt": True,
        # Translate every cue into English for the .en.srt file. This is a
        # network request per cue, so turn it off if you only need Persian.
        "translate_english": True,
    },

    "clips": {
        # Cut the long video into short vertical clips for Reels/Shorts/TikTok.
        "enabled": True,
        "count": 5,
        "min_seconds": 25.0,
        "max_seconds": 60.0,
        "aspect": "9:16",
        "width": 1080,
        "height": 1920,
        # "crop" keeps the centre of the frame and throws the sides away.
        # "blur" fits the whole frame and fills the gaps with a blurred copy.
        "framing": "crop",
        # Shift the crop window sideways, -1.0 (far left) to 1.0 (far right).
        "crop_offset": 0.0,
        "burn_subtitles": True,
        # Only cut clips when the video is at least this long.
        "min_source_seconds": 90.0,
    },

    "branding": {
        # All optional: leave the filename empty to skip that piece.
        # Paths are relative to the assets/ folder unless absolute.
        "intro": "",
        "outro": "",
        "music": "",
        "music_volume": 0.12,
        # Pull the music down while somebody is speaking.
        "duck_music": True,
        "logo": "",
        "logo_position": "top-right",
        "logo_width_pct": 9.0,
        "logo_opacity": 0.75,
        "fade_in_seconds": 0.4,
        "fade_out_seconds": 0.8,
    },

    "output": {
        # "" keeps the source resolution. Otherwise a height like 1080 or 720.
        "height": "",
        "crf": 20,
        "preset": "medium",
        "audio_bitrate": "192k",
        # Move the original into videos-in/done/ once it has been edited.
        "archive_originals": True,
        # Also save the plain transcript as a .txt file.
        "write_transcript": True,
    },

    "whisper": {
        # Leave empty to reuse the model the bot is already configured with.
        # "large-v3" is the most accurate, "large-v3-turbo" is ~3x faster.
        "model": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Overlay stored values on the defaults without losing unset keys."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load() -> dict:
    if not CONFIG_FILE.exists():
        return copy.deepcopy(DEFAULTS)
    try:
        stored = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("video_settings.json is unreadable; using the defaults")
        return copy.deepcopy(DEFAULTS)
    if not isinstance(stored, dict):
        logger.warning("video_settings.json is not an object; using the defaults")
        return copy.deepcopy(DEFAULTS)
    return _deep_merge(DEFAULTS, stored)


def save(config: dict) -> None:
    try:
        CONFIG_FILE.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        logger.exception("Could not write video_settings.json")


def write_default_file() -> Path:
    """Create video_settings.json on first run so the user has something to edit."""
    if not CONFIG_FILE.exists():
        save(copy.deepcopy(DEFAULTS))
    return CONFIG_FILE


def asset_path(name: str) -> Path | None:
    """Resolve a branding asset, tolerating both bare names and full paths."""
    if not name:
        return None
    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = ASSETS_DIR / candidate
    return candidate if candidate.exists() else None
