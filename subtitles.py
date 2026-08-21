"""Turn timed speech into subtitle files, in any language, plainly styled.

ASS rather than SRT is used for the burned-in subtitles because SRT carries no
styling: font, size, outline and position would all fall back to whatever the
renderer felt like. ASS pins them, and it is also what makes animation and
word-by-word highlighting possible at all.

Persian is right-to-left, which libass handles on its own through fribidi as
long as the text is stored in logical order — the order it is spoken. Nothing
here should reverse it. What does need doing by hand is line breaking, since
WrapStyle 2 turns off automatic wrapping so that a line never breaks in a place
that reads badly.

Nothing here knows that Persian is special. A cue holds a dict of language code
to text, so burning Persian over German is the same code path as Persian over
English — only the font and the character-width estimate change.
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

BURN_NONE = "none"

# Scripts that read right to left. libass reorders these itself; the list is
# here only to pick sensible defaults for width estimates and fonts.
RTL_LANGUAGES = {"fa", "ar", "ur", "he", "ps", "ku"}

# Languages written in Arabic script, whose letters join and so average
# narrower per character than Latin ones.
ARABIC_SCRIPT_LANGUAGES = {"fa", "ar", "ur", "ps", "ku"}

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

# Font sizes are given as a percentage of the frame height, which reads well
# until the frame is vertical: 5% of a 1920-tall 9:16 clip is enormous next to
# its 1080-wide picture. Capping the reference against the width keeps a clip
# in proportion with the 16:9 edit it was cut from, and leaves 16:9 untouched.
VERTICAL_REFERENCE_RATIO = 1.4

# Average glyph advance as a fraction of the font size.
LATIN_CHAR_RATIO = 0.55
ARABIC_CHAR_RATIO = 0.48

# Multiplier from font size to the height one line of it occupies.
LINE_HEIGHT_RATIO = 1.45

# Animation styles. All of them are monochrome and short on purpose: the point
# is to stop text appearing with a jolt, not to decorate it.
ANIMATION_NONE = "none"
ANIMATION_FADE = "fade"
ANIMATION_RISE = "rise"
ANIMATION_POP = "pop"
ANIMATION_KARAOKE = "karaoke"
ANIMATIONS = (ANIMATION_NONE, ANIMATION_FADE, ANIMATION_RISE, ANIMATION_POP, ANIMATION_KARAOKE)

# How far "rise" travels, as a fraction of the font size.
RISE_DISTANCE_RATIO = 0.55


@dataclass(frozen=True)
class TimedWord:
    """One spoken word and the seconds it occupies."""

    start: float
    end: float
    text: str


@dataclass
class SubtitleCue:
    """One on-screen line, in however many languages have been filled in."""

    start: float
    end: float
    source: str = "fa"
    texts: dict[str, str] = field(default_factory=dict)
    # Word timings for the spoken language, kept only for karaoke highlighting.
    words: list[TimedWord] = field(default_factory=list)

    @property
    def source_text(self) -> str:
        return self.texts.get(self.source, "")

    def text_for(self, language: str) -> str:
        return self.texts.get(language, "")


@dataclass(frozen=True)
class TextOverlay:
    """A caption that is not speech: a hook line, a label, a call to action."""

    start: float
    end: float
    text: str
    language: str = "fa"
    # 8 = top centre, 5 = middle centre, 2 = bottom centre (ASS numpad layout).
    alignment: int = 8
    size_pct: float = 5.0
    margin_pct: float = 8.0


def is_rtl(language: str) -> bool:
    return language in RTL_LANGUAGES


def char_ratio(language: str) -> float:
    return ARABIC_CHAR_RATIO if language in ARABIC_SCRIPT_LANGUAGES else LATIN_CHAR_RATIO


def style_name(language: str) -> str:
    return f"L{language.upper()}"


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
    return [" ".join(line) for line in wrap_tokens(text.split(), max_chars, max_lines, hard_limit)]


def wrap_tokens(
    tokens: list[str], max_chars: int, max_lines: int, hard_limit: int | None = None
) -> list[list[str]]:
    """The same wrapping, but keeping the words separate.

    Karaoke needs to know which word landed on which line, which is lost the
    moment the line is joined into a string.
    """
    if not tokens:
        return []

    ceiling = hard_limit if hard_limit is not None else max(8, max_chars)
    budget = min(max(8, max_chars), ceiling)
    lines: list[list[str]] = []
    for _ in range(4):
        lines = []
        current: list[str] = []
        length = 0
        for token in tokens:
            addition = len(token) + (1 if current else 0)
            if current and length + addition > budget:
                lines.append(current)
                current, length = [token], len(token)
            else:
                current.append(token)
                length += addition
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


def _font_for(config: dict, language: str) -> str:
    fonts = config.get("fonts") or {}
    if language in fonts and fonts[language]:
        return fonts[language]
    if fonts.get("default"):
        return fonts["default"]
    return "Tahoma" if language in ARABIC_SCRIPT_LANGUAGES else "Arial"


def _style_line(
    name: str, font: str, size: int, config: dict, margin_v: int, side_margin: int
) -> str:
    primary = _ass_colour(config.get("primary_color", "#FFFFFF"))
    outline = _ass_colour(config.get("outline_color", "#000000"))
    # Secondary is the "not spoken yet" colour, which only karaoke uses. A dim
    # grey keeps the highlight monochrome; ASS's red default would not.
    secondary = _ass_colour(config.get("dim_color", "#8C8C8C"))
    bold = 1 if config.get("bold", True) else 0
    return (
        f"Style: {name},{font},{size},{primary},{secondary},{outline},&H80000000,"
        f"{bold},0,0,0,100,100,0,0,1,"
        f"{float(config.get('outline_width', 3.0)):.1f},"
        f"{float(config.get('shadow', 0.6)):.1f},"
        f"2,{side_margin},{side_margin},{margin_v},1"
    )


def _fade_tag(milliseconds: int) -> str:
    return f"\\fad({milliseconds},{min(milliseconds, 160)})"


def _animation_prefix(
    animation: str, milliseconds: int, centre_x: int, baseline_y: int, rise: int
) -> str:
    """The ASS override block that opens an animated line."""
    if animation == ANIMATION_NONE:
        return ""
    if animation == ANIMATION_POP:
        return (
            "{" + _fade_tag(milliseconds)
            + f"\\fscx92\\fscy92\\t(0,{milliseconds},\\fscx100\\fscy100)" + "}"
        )
    if animation == ANIMATION_RISE:
        # \move needs absolute coordinates, so the line also has to carry its
        # own anchor: \an2 puts the anchor at the bottom centre of the block,
        # which is where the style's margins would have put it anyway.
        return (
            "{\\an2"
            + f"\\move({centre_x},{baseline_y + rise},{centre_x},{baseline_y},0,{milliseconds})"
            + _fade_tag(milliseconds) + "}"
        )
    # Both fade and karaoke open with a plain fade; karaoke adds its own
    # per-word tags to the body.
    return "{" + _fade_tag(milliseconds) + "}"


def _karaoke_body(
    cue: SubtitleCue, lines_of_words: list[list[str]], words: list[TimedWord]
) -> str | None:
    """Lay \\k timings over the wrapped text so words light up as spoken.

    Returns None when the word timings do not line up with the wrapped text,
    in which case the caller should fall back to a plain line rather than
    render something out of sync.
    """
    flat = [token for line in lines_of_words for token in line]
    if not words or len(flat) != len(words):
        return None

    pieces: list[str] = []
    lead = int(round((words[0].start - cue.start) * 100))
    if lead > 0:
        pieces.append(f"{{\\k{lead}}}")

    index = 0
    for line_number, line in enumerate(lines_of_words):
        if line_number:
            pieces.append("\\N")
        for position, token in enumerate(line):
            word = words[index]
            # A word holds its highlight until the next one starts, so the gap
            # between words belongs to the word before it. Without this the
            # highlight would blink off during every pause.
            following = words[index + 1].start if index + 1 < len(words) else cue.end
            duration = max(1, int(round((following - word.start) * 100)))
            spacer = " " if position else ""
            pieces.append(f"{{\\k{duration}}}{spacer}{_escape_ass(token)}")
            index += 1

    return "".join(pieces)


def normalise_burn(value, source_language: str) -> list[str]:
    """Accept the old "fa"/"both"/"none" spellings as well as a list of codes."""
    if value is None:
        return [source_language]
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("", BURN_NONE):
            return []
        if lowered == "both":
            return [source_language, "en"]
        return [lowered]
    return [str(item).strip().lower() for item in value if str(item).strip()]


def build_ass(
    cues: list[SubtitleCue],
    config: dict,
    width: int,
    height: int,
    burn: list[str],
    overlays: list[TextOverlay] | None = None,
) -> str:
    """Render the cues, and any non-speech overlays, as an ASS document.

    `burn` is ordered: the first language is the primary one and sits above,
    larger; a second sits under it, smaller. Further languages are ignored,
    because a third line of text leaves no picture to read it over.
    """
    burn = [language for language in burn][:2]
    overlays = overlays or []

    reference = _reference_height(width, height)
    primary_size = max(12, int(reference * float(config.get("font_size_pct", 5.2)) / 100))
    secondary_size = max(10, int(reference * float(config.get("secondary_font_size_pct", 3.2)) / 100))
    base_margin = max(10, int(height * float(config.get("margin_bottom_pct", 7.0)) / 100))

    side_margin = max(20, int(width * 0.06))
    usable_width = max(80, width - 2 * side_margin)

    max_chars = int(config.get("max_chars_per_line", 42))
    max_lines = int(config.get("max_lines", 2))

    animation = str(config.get("animation", ANIMATION_FADE)).lower()
    if animation not in ANIMATIONS:
        logger.warning("Unknown subtitle animation %r; falling back to fade", animation)
        animation = ANIMATION_FADE
    animation_ms = max(0, int(config.get("animation_ms", 180)))
    if animation_ms == 0:
        animation = ANIMATION_NONE

    sizes = {language: (primary_size if index == 0 else secondary_size)
             for index, language in enumerate(burn)}
    secondary_line_height = int(secondary_size * LINE_HEIGHT_RATIO)

    styles = []
    for index, language in enumerate(burn):
        margin = base_margin + (secondary_line_height if index == 0 and len(burn) > 1 else 0)
        styles.append(
            _style_line(
                style_name(language), _font_for(config, language), sizes[language],
                config, margin, side_margin,
            )
        )
    for language in {overlay.language for overlay in overlays}:
        if language in burn:
            continue
        styles.append(
            _style_line(
                style_name(language), _font_for(config, language), primary_size,
                config, base_margin, side_margin,
            )
        )
    if not styles:
        styles.append(
            _style_line("LDEFAULT", _font_for(config, "en"), primary_size,
                        config, base_margin, side_margin)
        )

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
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    centre_x = width // 2

    for cue in cues:
        if cue.end <= cue.start:
            continue
        start, end = _ass_time(cue.start), _ass_time(cue.end)

        # Wrap the lower language first: how tall it turns out decides how far
        # the upper one has to be lifted.
        wrapped: dict[str, list[list[str]]] = {}
        for index, language in enumerate(burn):
            text = cue.text_for(language).strip()
            if not text:
                continue
            size = sizes[language]
            fit = _chars_that_fit(usable_width, size, char_ratio(language))
            budget = min(max_chars if index == 0 else int(max_chars * 1.15), fit)
            wrapped[language] = wrap_tokens(text.split(), budget, max_lines, fit)

        lower_lines = len(wrapped.get(burn[1], [])) if len(burn) > 1 else 0

        for index, language in enumerate(burn):
            token_lines = wrapped.get(language)
            if not token_lines:
                continue
            size = sizes[language]
            margin = base_margin + (lower_lines * secondary_line_height if index == 0 else 0)
            baseline_y = height - margin
            rise = max(4, int(size * RISE_DISTANCE_RATIO))

            body = None
            # Karaoke only makes sense for the language actually being spoken;
            # a translation has no per-word timing to follow.
            if animation == ANIMATION_KARAOKE and language == cue.source:
                body = _karaoke_body(cue, token_lines, cue.words)

            prefix = _animation_prefix(animation, animation_ms, centre_x, baseline_y, rise)
            if body is None:
                body = "\\N".join(
                    " ".join(_escape_ass(token) for token in line) for line in token_lines
                )
            if not body:
                continue

            lines.append(
                f"Dialogue: 0,{start},{end},{style_name(language)},,0,0,{margin},,{prefix}{body}"
            )

    for overlay in overlays:
        if overlay.end <= overlay.start or not overlay.text.strip():
            continue
        size = max(12, int(reference * overlay.size_pct / 100))
        fit = _chars_that_fit(usable_width, size, char_ratio(overlay.language))
        token_lines = wrap_tokens(overlay.text.split(), min(max_chars, fit), 3, fit)
        body = "\\N".join(" ".join(_escape_ass(t) for t in line) for line in token_lines)
        margin = max(10, int(height * overlay.margin_pct / 100))
        # Overlays carry their size and alignment inline so they can differ
        # from the subtitle style without needing a style of their own.
        tags = f"\\an{overlay.alignment}\\fs{size}" + _fade_tag(max(160, animation_ms))
        lines.append(
            f"Dialogue: 1,{_ass_time(overlay.start)},{_ass_time(overlay.end)},"
            f"{style_name(overlay.language)},,0,0,{margin},,{{{tags}}}{body}"
        )

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
    cues: list[SubtitleCue],
    path: Path,
    config: dict,
    width: int,
    height: int,
    burn: list[str],
    overlays: list[TextOverlay] | None = None,
) -> Path:
    path.write_text(
        build_ass(cues, config, width, height, burn, overlays), encoding="utf-8"
    )
    return path


def write_srt(cues: list[SubtitleCue], path: Path, language: str) -> Path | None:
    content = build_srt(cues, language)
    if not content.strip():
        return None
    # A BOM is what makes players and editors read the text as UTF-8 rather
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
                source=cue.source,
                texts=dict(cue.texts),
                words=[
                    TimedWord(word.start - start, word.end - start, word.text)
                    for word in cue.words
                ],
            )
        )
    return window


def transcript_text(cues: list[SubtitleCue], language: str) -> str:
    """Join the cues back into flowing text, for the .txt transcript."""
    return " ".join(
        cue.text_for(language).strip() for cue in cues if cue.text_for(language).strip()
    )
