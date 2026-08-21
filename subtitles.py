"""Turn timed speech into subtitle files, in both Persian and English.

ASS rather than SRT is used for the burned-in subtitles because SRT carries no
styling: font, size, outline and position would all fall back to whatever the
renderer felt like. ASS pins them.

Persian is right-to-left, which libass handles on its own through fribidi as
long as the text is stored in logical order — the order it is spoken. Nothing
here should reverse it. What does need doing by hand is line breaking, since
WrapStyle 2 turns off automatic wrapping so that a line never breaks in a place
that reads badly in Persian.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Persian and English are written into the same file as two named styles.
STYLE_PERSIAN = "FA"
STYLE_ENGLISH = "EN"

BURN_PERSIAN = "fa"
BURN_ENGLISH = "en"
BURN_BOTH = "both"
BURN_NONE = "none"

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

# Font sizes are given as a percentage of the frame height, which reads well
# until the frame is vertical: 5% of a 1920-tall 9:16 clip is enormous next to
# its 1080-wide picture. Capping the reference against the width keeps a clip
# in proportion with the 16:9 edit it was cut from, and leaves 16:9 untouched.
VERTICAL_REFERENCE_RATIO = 1.4

# Average glyph advance as a fraction of the font size. Arabic script is the
# narrower of the two because its letters join.
LATIN_CHAR_RATIO = 0.55
ARABIC_CHAR_RATIO = 0.48

# Multiplier from font size to the height one line of it occupies.
LINE_HEIGHT_RATIO = 1.45


@dataclass
class SubtitleCue:
    """One on-screen line, with either or both languages filled in."""

    start: float
    end: float
    persian: str = ""
    english: str = ""

    def text_for(self, language: str) -> str:
        return self.persian if language == BURN_PERSIAN else self.english


def _ass_colour(value: str, alpha: int = 0) -> str:
    """#RRGGBB to ASS's &HAABBGGRR, which is byte-reversed and alpha-first."""
    match = _HEX_RE.match((value or "").strip())
    if not match:
        return "&H00FFFFFF"
    red, green, blue = (int(match.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:  # rounding can carry into the next second
        centis, secs = 0, secs + 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis, secs = 0, secs + 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def wrap(text: str, max_chars: int, max_lines: int, hard_limit: int | None = None) -> list[str]:
    """Break text into lines of at most `max_chars`, never mid-word.

    When the text needs more lines than allowed, the budget per line is grown
    instead of the tail being thrown away — a clipped subtitle is worse than a
    slightly tall one. `hard_limit` is where that growth stops: it is the
    number of characters that physically fit across the frame, and going past
    it would push the text off both edges of the picture.
    """
    words = text.split()
    if not words:
        return []

    ceiling = hard_limit if hard_limit is not None else max(8, max_chars)
    budget = min(max(8, max_chars), ceiling)
    for _ in range(4):
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > budget:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        if len(lines) <= max_lines or budget >= ceiling:
            return lines
        budget = min(int(budget * 1.35), ceiling)

    return lines


def _reference_height(width: int, height: int) -> float:
    """The dimension font sizes are measured against. See the ratio above."""
    return min(float(height), width * VERTICAL_REFERENCE_RATIO)


def _chars_that_fit(usable_width: int, font_size: int, ratio: float) -> int:
    """How many characters of this size fit across the frame before clipping."""
    return max(8, int(usable_width / max(1.0, font_size * ratio)))


def _escape_ass(text: str) -> str:
    """Curly braces open an override block in ASS, so they cannot be literal."""
    return text.replace("{", "(").replace("}", ")")


def _style_line(
    name: str, font: str, size: int, config: dict, margin_v: int, side_margin: int
) -> str:
    primary = _ass_colour(config.get("primary_color", "#FFFFFF"))
    outline = _ass_colour(config.get("outline_color", "#000000"))
    return (
        f"Style: {name},{font},{size},{primary},&H000000FF,{outline},&H80000000,"
        f"0,0,0,0,100,100,0,0,1,"
        f"{float(config.get('outline_width', 2.6)):.1f},"
        f"{float(config.get('shadow', 0.8)):.1f},"
        f"2,{side_margin},{side_margin},{margin_v},1"
    )


def build_ass(
    cues: list[SubtitleCue], config: dict, width: int, height: int, mode: str
) -> str:
    """Render the cues as an ASS subtitle document."""
    reference = _reference_height(width, height)
    persian_size = max(12, int(reference * float(config.get("font_size_pct", 5.2)) / 100))
    english_size = max(10, int(reference * float(config.get("english_font_size_pct", 3.2)) / 100))
    base_margin = max(10, int(height * float(config.get("margin_bottom_pct", 7.0)) / 100))

    show_persian = mode in (BURN_PERSIAN, BURN_BOTH)
    show_english = mode in (BURN_ENGLISH, BURN_BOTH)

    side_margin = max(20, int(width * 0.06))
    usable_width = max(80, width - 2 * side_margin)

    max_chars = int(config.get("max_chars_per_line", 42))
    max_lines = int(config.get("max_lines", 2))

    # Never wrap wider than the picture, whatever the configured line length
    # says: a line that runs off both edges is unreadable in a way that an
    # extra line is not.
    persian_fit = _chars_that_fit(usable_width, persian_size, ARABIC_CHAR_RATIO)
    english_fit = _chars_that_fit(usable_width, english_size, LATIN_CHAR_RATIO)
    persian_budget = min(max_chars, persian_fit)
    english_budget = min(int(max_chars * 1.15), english_fit)

    english_line_height = int(english_size * LINE_HEIGHT_RATIO)
    # The style value is the one-line case; each cue overrides it with the room
    # its own English block actually needs.
    persian_margin = base_margin
    if show_english and show_persian:
        persian_margin = base_margin + english_line_height

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        # 2 means "only break where there is an explicit \\N", which is what
        # keeps the hand-made wrapping above from being second-guessed.
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        _style_line(
            STYLE_PERSIAN, config.get("font", "Tahoma"), persian_size,
            config, persian_margin, side_margin,
        ),
        _style_line(
            STYLE_ENGLISH, config.get("english_font", "Arial"), english_size,
            config, base_margin, side_margin,
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cue in cues:
        if cue.end <= cue.start:
            continue
        start, end = _ass_time(cue.start), _ass_time(cue.end)

        english_lines: list[str] = []
        if show_english and cue.english.strip():
            english_lines = wrap(cue.english, english_budget, max_lines, english_fit)

        if show_persian and cue.persian.strip():
            body = "\\N".join(
                _escape_ass(line)
                for line in wrap(cue.persian, persian_budget, max_lines, persian_fit)
            )
            if body:
                # A three-line English block needs the Persian lifted three
                # lines, not the one the style assumes.
                margin = base_margin + len(english_lines) * english_line_height
                lines.append(f"Dialogue: 0,{start},{end},{STYLE_PERSIAN},,0,0,{margin},,{body}")

        if english_lines:
            body = "\\N".join(_escape_ass(line) for line in english_lines)
            lines.append(f"Dialogue: 0,{start},{end},{STYLE_ENGLISH},,0,0,0,,{body}")

    return "\n".join(lines) + "\n"


def build_srt(cues: list[SubtitleCue], language: str) -> str:
    """Render the cues as a plain SRT file for one language."""
    blocks: list[str] = []
    index = 1
    for cue in cues:
        text = cue.text_for(language).strip()
        if not text or cue.end <= cue.start:
            continue
        blocks.append(
            f"{index}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{text}\n"
        )
        index += 1
    return "\n".join(blocks)


def write_ass(
    cues: list[SubtitleCue], path: Path, config: dict, width: int, height: int, mode: str
) -> Path:
    path.write_text(build_ass(cues, config, width, height, mode), encoding="utf-8")
    return path


def write_srt(cues: list[SubtitleCue], path: Path, language: str) -> Path | None:
    content = build_srt(cues, language)
    if not content.strip():
        return None
    # A BOM is what makes players and editors read the Persian as UTF-8 rather
    # than guessing at a legacy code page.
    path.write_text(content, encoding="utf-8-sig")
    return path


def slice_cues(cues: list[SubtitleCue], start: float, end: float) -> list[SubtitleCue]:
    """Take the cues inside a time window and rebase them to start at zero.

    Used when a clip is cut out of the middle of a video: the clip's subtitles
    are the same words, but their timings have to start counting again from the
    clip's own first frame.
    """
    window: list[SubtitleCue] = []
    for cue in cues:
        if cue.end <= start or cue.start >= end:
            continue
        window.append(
            SubtitleCue(
                start=max(0.0, cue.start - start),
                end=min(end, cue.end) - start,
                persian=cue.persian,
                english=cue.english,
            )
        )
    return window


def transcript_text(cues: list[SubtitleCue], language: str) -> str:
    """Join the cues back into flowing text, for the .txt transcript."""
    return " ".join(cue.text_for(language).strip() for cue in cues if cue.text_for(language).strip())
