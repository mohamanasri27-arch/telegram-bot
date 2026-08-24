"""Pick the strongest stretches of a long video and cut them into short clips.

Choosing highlights is a judgement call, and there is no model running here to
make it. What there is, is the transcript — so the choice is made from what the
transcript can actually show: where speech is dense rather than halting, and
where a thought starts and finishes instead of being cut in half.

That is a heuristic, not taste. It reliably avoids clips that open mid-word or
consist mostly of silence, which is most of what makes an automatic cut
unwatchable. Tune it, or override it, from video_settings.json.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import ffmpeg_tools
import subtitles
import video_config
from subtitles import SubtitleCue

logger = logging.getLogger(__name__)

# A gap this long before a cue means the speaker had finished the last thought.
FRESH_START_GAP = 0.45

CLIP_FADE_SECONDS = 0.3


@dataclass
class Highlight:
    start: float
    end: float
    score: float
    opening_line: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _word_count(text: str) -> int:
    return len(text.split())


def _starts_cleanly(cues: list[SubtitleCue], index: int) -> bool:
    """True when this cue opens a thought rather than continuing one."""
    if index == 0:
        return True
    previous = cues[index - 1]
    if cues[index].start - previous.end >= FRESH_START_GAP:
        return True
    return previous.source_text.rstrip().endswith((".", "؟", "!", "…", "؛"))


def _ends_cleanly(cue: SubtitleCue) -> bool:
    return cue.source_text.rstrip().endswith((".", "؟", "!", "…"))


def pick_highlights(
    cues: list[SubtitleCue], duration: float, config: dict
) -> list[Highlight]:
    """Choose non-overlapping windows worth cutting into clips."""
    wanted = max(0, int(config.get("count", 5)))
    minimum = float(config.get("min_seconds", 25.0))
    maximum = float(config.get("max_seconds", 60.0))
    if wanted == 0 or duration < minimum:
        return []

    if not cues:
        return _evenly_spaced(duration, wanted, minimum, maximum)

    candidates: list[Highlight] = []
    for index, cue in enumerate(cues):
        start = cue.start
        words = 0
        end = cue.start
        last_index = index

        for forward in range(index, len(cues)):
            if cues[forward].end - start > maximum:
                break
            words += _word_count(cues[forward].source_text)
            end = cues[forward].end
            last_index = forward
            # Once the window is long enough, stop at the first clean ending
            # rather than running on to the maximum for no reason.
            if end - start >= minimum and _ends_cleanly(cues[forward]):
                break

        length = end - start
        if length < minimum:
            continue

        # Words per second is the core signal: a window that is mostly pause
        # scores badly however long it is.
        density = words / length
        score = density
        if _starts_cleanly(cues, index):
            score += 0.55
        if _ends_cleanly(cues[last_index]):
            score += 0.35
        # Nudge towards the middle of the allowed length, where clips sit best.
        score -= abs(length - (minimum + maximum) / 2) / (maximum * 4)

        candidates.append(
            Highlight(start=start, end=end, score=score, opening_line=cue.source_text)
        )

    chosen: list[Highlight] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(chosen) >= wanted:
            break
        # Clips that overlap are the same clip twice as far as a viewer is
        # concerned, so keep a real gap between them.
        if any(
            candidate.start < picked.end + 1.0 and picked.start < candidate.end + 1.0
            for picked in chosen
        ):
            continue
        chosen.append(candidate)

    chosen.sort(key=lambda item: item.start)
    return chosen


def _evenly_spaced(
    duration: float, wanted: int, minimum: float, maximum: float
) -> list[Highlight]:
    """Fallback for a video with no usable transcript: cut at regular intervals."""
    length = min(maximum, max(minimum, duration / max(1, wanted)))
    if duration < length:
        return []
    stride = (duration - length) / max(1, wanted - 1) if wanted > 1 else 0.0
    highlights = []
    for index in range(wanted):
        start = index * stride
        if start + length > duration:
            break
        highlights.append(Highlight(start=start, end=start + length, score=0.0))
    return highlights


def render_clip(
    source: Path,
    destination: Path,
    highlight: Highlight,
    cues: list[SubtitleCue],
    config: dict,
    source_info: ffmpeg_tools.MediaInfo,
    output_config: dict,
    subtitle_config: dict,
    workdir: Path,
    source_language: str = "fa",
    punch_in: float = 0.0,
) -> None:
    """Cut one vertical clip, with its own re-timed subtitles burned in."""
    width = ffmpeg_tools.even(int(config.get("width", 1080)))
    height = ffmpeg_tools.even(int(config.get("height", 1920)))
    length = highlight.duration

    reframe = ffmpeg_tools.vertical_filter(
        source_info,
        width,
        height,
        str(config.get("framing", "crop")),
        float(config.get("crop_offset", 0.0)),
    )

    video_chain = [reframe]

    # The zoom goes on before the subtitles, so the picture drifts in and the
    # text stays nailed to the bottom of the frame where it belongs.
    if punch_in > 0:
        video_chain.append(
            ffmpeg_tools.punch_in_filter(punch_in, length, source_info.fps, width, height)
        )

    burn = subtitles.normalise_burn(subtitle_config.get("burn"), source_language)
    if config.get("burn_subtitles", True) and burn:
        window = subtitles.slice_cues(cues, highlight.start, highlight.end)
        if window:
            # A bare relative name keeps the filter free of the ':' and '\'
            # that a Windows path would otherwise smuggle into the syntax.
            ass_name = f"{destination.stem}.ass"
            subtitles.write_ass(
                window, workdir / ass_name, subtitle_config, width, height, burn
            )
            video_chain.append(
                ffmpeg_tools.subtitle_filter(ass_name, workdir, video_config.FONTS_DIR)
            )

    fade_out_start = max(0.0, length - CLIP_FADE_SECONDS)
    video_chain.append(f"fade=t=in:st=0:d={CLIP_FADE_SECONDS}")
    video_chain.append(f"fade=t=out:st={fade_out_start:.3f}:d={CLIP_FADE_SECONDS}")

    graph = f"[0:v]{','.join(video_chain)}[v]"
    audio_label = None
    if source_info.has_audio:
        graph += (
            f";[0:a]afade=t=in:st=0:d={CLIP_FADE_SECONDS},"
            f"afade=t=out:st={fade_out_start:.3f}:d={CLIP_FADE_SECONDS}[a]"
        )
        audio_label = "[a]"

    ffmpeg_tools.encode(
        # Both before -i: after it they are output options, which happens to
        # work with a single input and quietly stops working with more.
        inputs=[[
            "-ss", f"{highlight.start:.3f}",
            "-t", f"{length:.3f}",
            "-i", str(source),
        ]],
        graph=graph,
        video_label="[v]",
        audio_label=audio_label,
        destination=destination,
        crf=int(output_config.get("crf", 20)) + 1,
        preset=str(output_config.get("preset", "medium")),
        audio_bitrate=str(output_config.get("audio_bitrate", "192k")),
        workdir=workdir,
    )


def index_text(highlights: list[Highlight]) -> str:
    """A short human-readable list of what was cut and where it came from."""
    lines = ["# clips cut from the edited video", ""]
    for number, highlight in enumerate(highlights, start=1):
        minutes, seconds = divmod(int(highlight.start), 60)
        lines.append(
            f"clip-{number:02d}.mp4  "
            f"from {minutes:02d}:{seconds:02d}  "
            f"({highlight.duration:.0f}s)  {highlight.opening_line[:60]}"
        )
    return "\n".join(lines) + "\n"
