"""Thin, typed wrappers around the ffmpeg and ffprobe command line tools.

Everything that actually touches pixels lives here. The rules that decide what
should happen to a video live in video_pipeline.py, so this module stays a
vocabulary of small operations rather than a policy.

Two details are load-bearing on Windows and easy to lose:

  - Filter graphs are written to a file and passed with -filter_complex_script.
    A silence-cut graph can run to tens of kilobytes, and Windows refuses any
    command line over 8191 characters.
  - Filters that take a filename (the subtitle burner) get a bare relative name
    and are run from the working directory. ffmpeg's filter syntax treats ':'
    and '\' as punctuation, so a Windows absolute path inside a filter needs
    three levels of escaping to survive. Not passing one avoids the problem.
"""
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    pass


class FFmpegMissing(FFmpegError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool


def _binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise FFmpegMissing(name)
    return found


def ensure_available() -> None:
    """Raise FFmpegMissing unless both tools are on PATH."""
    _binary("ffmpeg")
    _binary("ffprobe")


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    logger.debug("running: %s", " ".join(args))
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        # ffmpeg puts the real reason in the last few lines of stderr; the rest
        # is a banner nobody needs in a log file.
        tail = "\n".join((result.stderr or "").strip().splitlines()[-12:])
        raise FFmpegError(f"{Path(args[0]).name} failed:\n{tail}")
    return result


_MAJOR_VERSION_RE = re.compile(r"version\s+n?(\d+)")


def _major_version() -> int:
    try:
        banner = subprocess.run(
            [_binary("ffmpeg"), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ).stdout
    except OSError:
        return 0
    match = _MAJOR_VERSION_RE.search(banner or "")
    return int(match.group(1)) if match else 0


def _filter_script_flag() -> str:
    """ffmpeg 7 renamed -filter_complex_script and 8 may drop the old spelling."""
    return "-/filter_complex" if _major_version() >= 7 else "-filter_complex_script"


def probe(path: Path) -> MediaInfo:
    """Read duration, dimensions and stream layout out of a media file."""
    result = _run([
        _binary("ffprobe"), "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ])
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for source in (data.get("format", {}), video or {}, audio or {}):
        try:
            duration = float(source.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            break

    fps = 30.0
    if video:
        raw_rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1"
        try:
            numerator, _, denominator = raw_rate.partition("/")
            denominator_value = float(denominator or 1)
            if denominator_value:
                candidate = float(numerator) / denominator_value
                if candidate > 0:
                    fps = candidate
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return MediaInfo(
        duration=duration,
        width=int(video.get("width") or 0) if video else 0,
        height=int(video.get("height") or 0) if video else 0,
        fps=fps,
        has_audio=audio is not None,
        has_video=video is not None,
    )


# --------------------------------------------------------------------------
# Silence analysis
# --------------------------------------------------------------------------

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(
    path: Path, threshold_db: float, min_silence: float, duration: float
) -> list[tuple[float, float]]:
    """Return the (start, end) spans ffmpeg considers silent."""
    result = subprocess.run(
        [
            _binary("ffmpeg"), "-hide_banner", "-nostdin",
            "-i", str(path),
            "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # A non-zero exit here means the analysis failed, not that the video is bad.
    # Returning no silences simply leaves the video uncut.
    if result.returncode != 0:
        logger.warning("Silence detection failed; keeping the video uncut")
        return []

    silences: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in (result.stderr or "").splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = max(0.0, float(start_match.group(1)))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            silences.append((pending_start, float(end_match.group(1))))
            pending_start = None

    # A silence that runs to the end of the file never gets an end marker.
    if pending_start is not None and duration > pending_start:
        silences.append((pending_start, duration))

    return silences


def keep_ranges(
    silences: list[tuple[float, float]],
    duration: float,
    *,
    padding: float,
    trim_start: bool,
    trim_end: bool,
    minimum_keep: float = 0.25,
) -> list[tuple[float, float]]:
    """Turn silent spans into the spans worth keeping.

    Each kept span is grown by `padding` at both ends so speech does not start
    the instant a cut lands, which is what makes automated cuts sound clipped.
    """
    if duration <= 0:
        return []
    if not silences:
        return [(0.0, duration)]

    # Leading and trailing silence is only removed when asked for; a cut in the
    # middle is always fair game.
    trimmed = list(silences)
    if not trim_start and trimmed and trimmed[0][0] <= 0.05:
        trimmed = trimmed[1:]
    if not trim_end and trimmed and trimmed[-1][1] >= duration - 0.05:
        trimmed = trimmed[:-1]

    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in trimmed:
        start = max(0.0, min(cursor, duration))
        end = max(start, min(silence_start + padding, duration))
        if end - start >= minimum_keep:
            spans.append((start, end))
        cursor = max(cursor, silence_end - padding)

    if duration - cursor >= minimum_keep:
        spans.append((max(0.0, cursor), duration))

    if not spans:
        return [(0.0, duration)]

    # Padding can push neighbouring spans into each other; merging them keeps
    # the filter graph small and stops frames being duplicated.
    merged = [spans[0]]
    for start, end in spans[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 0.02:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def cut_to_ranges(
    source: Path, destination: Path, ranges: list[tuple[float, float]], info: MediaInfo,
    *, crf: int, preset: str, audio_bitrate: str,
    video_filter: str | None = None, audio_filter: str | None = None,
    silent_audio: bool = False,
) -> None:
    """Re-encode `source` keeping only `ranges`, joined end to end.

    `video_filter` and `audio_filter` are applied once after the join rather
    than to every kept span, so colour and loudness work costs nothing extra.
    `silent_audio` bolts a silent track onto a video that has none, which lets
    every later stage assume audio exists.
    """
    if not ranges:
        raise FFmpegError("no ranges to keep")

    args = [
        _binary("ffmpeg"), "-hide_banner", "-nostdin", "-y",
        "-i", str(source),
    ]
    audio_source = "0:a"
    if not info.has_audio and silent_audio:
        args += [
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        ]
        audio_source = "1:a"

    keep_audio = info.has_audio or silent_audio

    video_parts: list[str] = []
    audio_parts: list[str] = []
    concat_inputs: list[str] = []

    for index, (start, end) in enumerate(ranges):
        video_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        if keep_audio:
            audio_parts.append(
                f"[{audio_source}]atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        concat_inputs.append(f"[v{index}]" + (f"[a{index}]" if keep_audio else ""))

    audio_flag = 1 if keep_audio else 0
    graph = ";".join(video_parts + audio_parts) + ";" + (
        "".join(concat_inputs)
        + f"concat=n={len(ranges)}:v=1:a={audio_flag}"
        + ("[cv][ca]" if keep_audio else "[cv]")
    )
    graph += f";[cv]{video_filter}[outv]" if video_filter else ";[cv]null[outv]"
    if keep_audio:
        graph += f";[ca]{audio_filter}[outa]" if audio_filter else ";[ca]anull[outa]"

    args += _graph_args(graph, destination.parent)
    args += ["-map", "[outv]"]
    if keep_audio:
        args += ["-map", "[outa]", "-c:a", "aac", "-b:a", audio_bitrate]
    args += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(destination),
    ]
    _run(args)


# --------------------------------------------------------------------------
# Filter graph plumbing
# --------------------------------------------------------------------------

def _graph_args(graph: str, near: Path) -> list[str]:
    """Write a filter graph to a temp file and return the flags that load it.

    The file is left on disk for ffmpeg to read and is cleared away with the
    rest of the run's working directory afterwards.
    """
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".filter", delete=False, encoding="utf-8",
        dir=str(near) if near.exists() else None,
    )
    with handle:
        handle.write(graph)
    return [_filter_script_flag(), handle.name]


def even(value: float) -> int:
    """Round to an even integer, which is what yuv420p encoding requires."""
    return max(2, int(round(value / 2)) * 2)


def scale_filter(target_height: int, source: MediaInfo) -> str | None:
    """A scale filter that keeps the aspect ratio, or None if already correct."""
    if not target_height or not source.height or target_height == source.height:
        return None
    width = even(source.width * target_height / source.height)
    return f"scale={width}:{even(target_height)}:flags=lanczos"


def vertical_filter(
    source: MediaInfo, width: int, height: int, framing: str, offset: float
) -> str:
    """Reframe a landscape video into a vertical one.

    "crop" keeps the middle of the frame and is right for a talking head.
    "blur" keeps the whole frame and fills the bars with a blurred copy, which
    loses nothing but reads as smaller on a phone.
    """
    width, height = even(width), even(height)

    if framing == "blur":
        return (
            f"split[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=28[bgb];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )

    target_ratio = width / height
    source_ratio = (source.width / source.height) if source.height else target_ratio

    if source_ratio <= target_ratio:
        # Already at least as narrow as the target: scaling and padding the top
        # and bottom keeps the whole frame instead of cropping the head off.
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    crop_width = even(source.height * target_ratio)
    # offset runs -1..1 across the unused width on either side of centre.
    slack = max(0, source.width - crop_width) / 2
    left = even(slack + max(-1.0, min(1.0, offset)) * slack)
    left = max(0, min(left, source.width - crop_width))
    return f"crop={crop_width}:{source.height}:{left}:0,scale={width}:{height}:flags=lanczos"


def logo_overlay_position(position: str, margin: int = 36) -> str:
    """Translate a corner name into overlay x:y coordinates."""
    positions = {
        "top-left": f"{margin}:{margin}",
        "top-right": f"W-w-{margin}:{margin}",
        "bottom-left": f"{margin}:H-h-{margin}",
        "bottom-right": f"W-w-{margin}:H-h-{margin}",
    }
    return positions.get(position, positions["top-right"])


def encode(
    inputs: list[list[str]],
    graph: str,
    video_label: str,
    audio_label: str | None,
    destination: Path,
    *,
    crf: int,
    preset: str,
    audio_bitrate: str,
    workdir: Path,
    fps: float | None = None,
) -> None:
    """Run one encode from an assembled filter graph.

    `inputs` is a list of per-input argument groups, so an input can carry its
    own flags (-stream_loop for music, -f lavfi for generated silence).
    """
    args = [_binary("ffmpeg"), "-hide_banner", "-nostdin", "-y"]
    for group in inputs:
        args += group
    args += _graph_args(graph, workdir)
    args += ["-map", video_label]
    if audio_label:
        args += ["-map", audio_label, "-c:a", "aac", "-b:a", audio_bitrate]
    else:
        args += ["-an"]
    if fps:
        args += ["-r", f"{fps:.4f}"]
    args += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(destination),
    ]
    _run(args, cwd=workdir)


def speech_filter(config: dict) -> str:
    """The filter chain that cleans speech up before Whisper hears it.

    This never touches the audio anyone will listen to — only the throwaway wav
    fed to the recogniser — so it can be more aggressive than a mix would
    tolerate. Room rumble, handling noise and hiss are all things the decoder
    has to work around, and removing them is the cheapest accuracy the pipeline
    can buy.
    """
    parts = ["highpass=f=85"]
    if config.get("denoise", True):
        parts.append("afftdn=nr=12:nf=-28")
    if config.get("even_out_levels", True):
        # Whisper handles a steady voice better than one that drifts between
        # loud and almost inaudible within the same sentence.
        parts.append("dynaudnorm=f=250:g=7:p=0.9")
    return ",".join(parts)


def extract_audio(source: Path, destination: Path, audio_filter: str | None = None) -> None:
    """Pull a 16 kHz mono wav out of a video, which is what Whisper wants."""
    args = [
        _binary("ffmpeg"), "-hide_banner", "-nostdin", "-y",
        "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000",
    ]
    if audio_filter:
        args += ["-af", audio_filter]
    args += ["-c:a", "pcm_s16le", str(destination)]
    _run(args)


def punch_in_filter(
    amount: float, seconds: float, fps: float, width: int, height: int
) -> str:
    """A slow zoom towards the centre, to keep a locked-off shot alive.

    Deliberately tiny: a few percent across the whole clip is felt rather than
    seen, which is the difference between production value and a gimmick.
    """
    frames = max(1, int(seconds * fps))
    step = max(1e-6, amount / frames)
    return (
        f"zoompan=z='min(1+{step:.8f}*on,{1 + amount:.4f})'"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d=1:s={even(width)}x{even(height)}:fps={fps:.4f}"
    )


def colour_input(colour: str, width: int, height: int, fps: float, seconds: float) -> list[str]:
    """Input arguments for a generated solid-colour segment."""
    return [
        "-f", "lavfi", "-t", f"{max(0.1, seconds):.3f}",
        "-i", f"color=c={colour}:s={even(width)}x{even(height)}:r={fps:.4f}",
    ]


def silence_input(seconds: float) -> list[str]:
    """Input arguments for a silent stereo track of a given length."""
    return [
        "-f", "lavfi", "-t", f"{max(0.1, seconds):.3f}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
    ]


def thumbnail(source: Path, destination: Path, at_second: float) -> None:
    """Grab a single frame, for a cover image."""
    _run([
        _binary("ffmpeg"), "-hide_banner", "-nostdin", "-y",
        "-ss", f"{max(0.0, at_second):.3f}",
        "-i", str(source),
        "-frames:v", "1", "-q:v", "2",
        str(destination),
    ])
