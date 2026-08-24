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
    # The language actually spoken in the video.
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

    # Everything here exists to make the transcript more accurate. None of it
    # changes the sound of the finished video.
    "accuracy": {
        # Clean the audio before the recogniser hears it. This copy is thrown
        # away afterwards, so it never affects what the viewer hears.
        "clean_audio_first": True,
        "denoise": True,
        "even_out_levels": True,
        # One sentence saying what the video is about, in the spoken language.
        # This is the single biggest win available for jargon and names:
        # told the subject, the recogniser stops turning unfamiliar words into
        # ordinary ones that sound similar.
        #   e.g. "این ویدیو درباره‌ی برنامه‌نویسی، وایب کدینگ و ساخت وب‌سایت است."
        "topic": "",
        # Let each chunk see the running transcript. Reads better on clean
        # single-speaker audio; on noisy audio it can spiral into repetition.
        "use_context": False,
    },

    "subtitles": {
        "enabled": True,
        # Languages burned onto the picture, in order. The first is the main
        # one and sits above, larger; a second sits under it, smaller.
        # []  or "none" burns nothing. Examples:
        #   ["fa"]        just Persian
        #   ["fa", "en"]  Persian over English
        #   ["de"]        German only
        "burn": ["fa"],
        # Languages to translate into, for the .srt files (and available to
        # burn). The spoken language is always included for free.
        "translate_to": ["en", "de"],
        # Fonts per language. Must be installed on this computer. Tahoma ships
        # with Windows and covers Persian; Vazirmatn looks better if installed.
        "fonts": {
            "fa": "Tahoma",
            "en": "Arial",
            "de": "Arial",
            "default": "Arial",
        },
        # Percentages of the video height, so they scale with the resolution.
        "font_size_pct": 5.2,
        "secondary_font_size_pct": 3.2,
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
        # The colour of a word not yet spoken, used only by the karaoke
        # animation. Grey keeps the highlight monochrome.
        "dim_color": "#8C8C8C",
        "outline_width": 3.0,
        "shadow": 0.6,
        "bold": True,
        # Distance from the bottom edge, as a percentage of video height.
        "margin_bottom_pct": 7.0,
        # Cue shaping: shorter cues are easier to read than whole sentences.
        "max_chars_per_line": 42,
        "max_lines": 2,
        "max_cue_seconds": 4.5,
        # How the text arrives on screen. All of these are monochrome and
        # short on purpose — the point is to stop text appearing with a jolt.
        #   none      appears instantly
        #   fade      soft fade in and out          (default)
        #   rise      fades in while sliding up a little
        #   pop       fades in while scaling up slightly
        #   karaoke   each word brightens as it is spoken
        "animation": "fade",
        "animation_ms": 180,
        # Write .srt files next to the video as well as burning them in.
        "write_srt": True,
    },

    # Extras that suit a promo or teaser. All optional, all off by default.
    "promo": {
        # A line held over the opening seconds — the hook.
        "hook_text": "",
        "hook_seconds": 3.0,
        "hook_position": "top",
        "hook_size_pct": 5.5,
        # A closing card: your handle, a call to action, a website.
        "end_card_text": "",
        "end_card_seconds": 2.5,
        "end_card_background": "#101820",
        "end_card_size_pct": 6.0,
        # A slow zoom on the vertical clips, to keep a locked-off shot alive.
        # 0 is off; 0.05 (five percent across the clip) is felt, not seen.
        "punch_in": 0.0,
    },

    "clips": {
        # Cut the long video into short vertical clips for Reels/Shorts/TikTok.
        "enabled": True,
        "count": 5,
        # Instagram Reels hold attention best well under a minute, so the
        # window is tighter than the YouTube Shorts limit would allow.
        "min_seconds": 15.0,
        "max_seconds": 30.0,
        "aspect": "9:16",
        "width": 1080,
        "height": 1920,
        # "crop" keeps the centre of the frame and throws the sides away.
        # "blur" fits the whole frame and fills the gaps with a blurred copy.
        "framing": "crop",
        # Shift the crop window sideways, -1.0 (far left) to 1.0 (far right).
        "crop_offset": 0.0,
        "burn_subtitles": True,
        # Only cut clips when the video is at least this long. Follows the clip
        # length above: a 60s source can still yield two 30s Reels.
        "min_source_seconds": 60.0,
    },

    # Lay your own recorded voice over the video.
    "voiceover": {
        # The recording. Relative to assets/ unless it is a full path.
        "file": "",
        # "replace" throws the original sound away and uses only your voice.
        # "mix" keeps the original underneath, at original_volume.
        "mode": "replace",
        "volume": 1.0,
        "original_volume": 0.12,
        # Pull the original down whenever your voice is speaking.
        "duck_original": True,
        # Seconds into the video where the recording should start.
        "start_at": 0.0,
        # Take the subtitles from your recording rather than the video's own
        # audio. This is almost always what you want: the recording is what
        # the viewer will actually hear.
        "transcribe": True,
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
        # Shape of the finished video:
        #   source     leave it as filmed          (default)
        #   vertical   1080x1920, for Reels and Stories
        #   square     1080x1080, for the feed
        #   landscape  1920x1080, for YouTube
        "format": "source",
        # How a reframe fills the new shape: "crop" or "blur". Same meaning as
        # in clips above.
        "framing": "crop",
        # "" keeps the source resolution. Otherwise a height like 1080 or 720.
        # Ignored unless format is "source".
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


# Named output shapes, as (width, height).
FORMATS = {
    "vertical": (1080, 1920),
    "square": (1080, 1080),
    "landscape": (1920, 1080),
}


def frame_size(name: str) -> tuple[int, int] | None:
    """The pixel size for a named output format, or None to keep the source."""
    return FORMATS.get(str(name or "source").strip().lower())


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
