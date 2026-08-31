"""Cut several short clips to music — the product-teaser style of edit.

This is a different job from the rest of the pipeline. Everything else starts
from one video of somebody talking and is paced by their speech. A montage has
no speech to follow: it starts from a handful of short shots and is paced by
the music, which is why it needs its own planner rather than another flag on
the main edit.

Beat detection here is deliberately modest. It finds where the low end of the
track punches — which is the kick drum in almost any music somebody would put
under a teaser — and offers those as cut points. It is not a tempo tracker, so
the shot length it produces is always clamped into a sane range rather than
trusted outright, and fixed-length shots stay the default.
"""
import logging
import math
import random
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import ffmpeg_tools

logger = logging.getLogger(__name__)

# The analysis rate for onset detection. Kick drums live well below this, and
# a low rate keeps the pure-Python pass fast on a long track.
ANALYSIS_RATE = 8000
FRAME_SAMPLES = 256

# An onset has to stand this far above the local average to count as a hit
# rather than as the track simply being loud.
ONSET_THRESHOLD = 1.45

# How far either side a frame must be the loudest to count as the hit itself
# rather than its decay. Must stay well under the gap between beats: at 180 BPM
# that gap is a third of a second.
PEAK_WINDOW_SECONDS = 0.12

# Two hits closer together than this are the same hit.
MIN_ONSET_GAP_SECONDS = 0.18

GRADES = {
    "none": "",
    # Slightly crushed, slightly desaturated, cool in the shadows.
    "cinematic": "eq=contrast=1.12:saturation=0.90:gamma=0.98,colorbalance=rs=-0.02:bs=0.06",
    "warm": "eq=contrast=1.06:saturation=1.08,colorbalance=rs=0.06:bs=-0.04",
    "cool": "eq=contrast=1.06:saturation=1.00,colorbalance=rs=-0.06:bs=0.06",
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# iPhones save these by default and ffmpeg cannot open them, so it is worth
# saying so plainly rather than letting the probe fail with nothing useful.
UNREADABLE_IMAGE_SUFFIXES = {".heic", ".heif"}

# How far a still photo drifts across its slot. Enough that it reads as a shot
# rather than a slide; little enough that nobody notices it happening.
STILL_ZOOM = 0.12
# Stills are scaled past the frame before the zoom so that pushing in crops
# real pixels instead of enlarging the ones already on screen.
STILL_OVERSCAN = 1.35


@dataclass
class Shot:
    source: Path
    start: float
    duration: float
    is_image: bool = False


def gather_sources(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Collect the videos and photos to build from, in a stable order.

    Returns the usable files and, separately, any that cannot be opened, so
    the caller can tell the difference between "no input" and "input in a
    format ffmpeg will not read".
    """
    usable: list[Path] = []
    unreadable: list[Path] = []

    def consider(item: Path) -> None:
        suffix = item.suffix.lower()
        if suffix in VIDEO_SUFFIXES or suffix in IMAGE_SUFFIXES:
            usable.append(item)
        elif suffix in UNREADABLE_IMAGE_SUFFIXES:
            unreadable.append(item)

    for path in paths:
        if path.is_dir():
            for item in sorted(path.iterdir()):
                if item.is_file():
                    consider(item)
        elif path.is_file():
            consider(path)
    return usable, unreadable


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


# --------------------------------------------------------------------------
# Where the music punches
# --------------------------------------------------------------------------

def detect_onsets(music: Path, workdir: Path) -> list[float]:
    """Return the seconds at which the low end of the track hits."""
    analysis = workdir / "beat.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(music),
                "-ac", "1", "-ar", str(ANALYSIS_RATE),
                "-af", "lowpass=f=250",
                "-c:a", "pcm_s16le", str(analysis),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, OSError):
        logger.warning("Could not analyse the music; falling back to fixed-length shots")
        return []

    try:
        with wave.open(str(analysis), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, OSError):
        return []

    samples = struct.unpack(f"<{len(frames) // 2}h", frames[: len(frames) // 2 * 2])
    if not samples:
        return []

    # Root-mean-square energy per short frame.
    energies: list[float] = []
    for index in range(0, len(samples) - FRAME_SAMPLES, FRAME_SAMPLES):
        window = samples[index:index + FRAME_SAMPLES]
        energies.append(math.sqrt(sum(value * value for value in window) / FRAME_SAMPLES))
    if len(energies) < 8:
        return []

    seconds_per_frame = FRAME_SAMPLES / ANALYSIS_RATE
    # Two different windows, because they answer two different questions.
    # "Is this loud for this part of the track?" needs a wide view, so a quiet
    # intro and a loud drop are each judged on their own terms. "Is this the
    # hit itself rather than its tail?" needs a narrow one — wider than the gap
    # between beats and only the loudest beat per window survives, which at 120
    # BPM throws away five beats in six.
    average_span = max(4, int(1.0 / seconds_per_frame))
    peak_span = max(1, int(PEAK_WINDOW_SECONDS / seconds_per_frame))

    onsets: list[float] = []
    last = -99.0
    for index, energy in enumerate(energies):
        low = max(0, index - average_span)
        high = min(len(energies), index + average_span + 1)
        neighbourhood = energies[low:high]
        average = sum(neighbourhood) / len(neighbourhood)
        if average <= 1e-6 or energy < average * ONSET_THRESHOLD:
            continue
        peak_low = max(0, index - peak_span)
        peak_high = min(len(energies), index + peak_span + 1)
        if energy < max(energies[peak_low:peak_high]):
            continue
        moment = index * seconds_per_frame
        if moment - last < MIN_ONSET_GAP_SECONDS:
            continue
        onsets.append(round(moment, 3))
        last = moment

    logger.info("Found %d beats in the music", len(onsets))
    return onsets


def _cut_points(
    onsets: list[float], total: float, minimum: float, maximum: float, fallback: float
) -> list[float]:
    """Turn raw onsets into shot boundaries of a sensible length."""
    if not onsets:
        step = max(0.2, fallback)
        points, moment = [], step
        while moment < total:
            points.append(moment)
            moment += step
        return points

    points: list[float] = []
    previous = 0.0
    for onset in onsets:
        if onset >= total:
            break
        gap = onset - previous
        if gap < minimum:
            continue
        # A long stretch with no usable beat still needs cutting, or one shot
        # sits there for ten seconds while the music carries on.
        while onset - previous > maximum:
            previous += maximum
            points.append(round(previous, 3))
        if onset - previous >= minimum:
            points.append(onset)
            previous = onset
    while total - previous > maximum:
        previous += maximum
        points.append(round(previous, 3))
    return points


def plan_shots(clips: list[Path], config: dict, onsets: list[float], total: float) -> list[Shot]:
    """Decide which piece of which clip fills each slot in the timeline."""
    minimum = float(config.get("min_shot_seconds", 0.8))
    maximum = float(config.get("max_shot_seconds", 4.0))
    fallback = float(config.get("shot_seconds", 2.0))

    boundaries = _cut_points(
        onsets if config.get("beat_sync", False) else [],
        total, minimum, maximum, fallback,
    )
    edges = [0.0] + boundaries + [total]

    order = list(range(len(clips)))
    if str(config.get("order", "sequence")).lower() == "shuffle":
        random.shuffle(order)

    stills = [is_image(clip) for clip in clips]
    # A photo has no length of its own, so it fills whatever slot it is given.
    durations = [
        0.0 if still else ffmpeg_tools.probe(clip).duration
        for clip, still in zip(clips, stills)
    ]
    # Where in each video the next slot should start, so a clip used twice
    # shows two different moments rather than the same opening frames again.
    cursors = [0.0] * len(clips)

    shots: list[Shot] = []
    for index in range(len(edges) - 1):
        length = edges[index + 1] - edges[index]
        if length < 0.15:
            continue
        which = order[index % len(order)]

        if stills[which]:
            shots.append(Shot(clips[which], 0.0, round(length, 3), is_image=True))
            continue

        available = durations[which]
        if available <= 0:
            continue
        start = cursors[which]
        if start + length > available:
            start = 0.0 if length >= available else max(0.0, available - length)
        cursors[which] = start + length
        shots.append(Shot(clips[which], round(start, 3), round(min(length, available), 3)))
    return shots


def render(
    shots: list[Shot],
    destination: Path,
    config: dict,
    output_config: dict,
    workdir: Path,
    music: Path | None = None,
    music_volume: float = 0.9,
) -> None:
    """Render the planned shots into one vertical video cut to the music."""
    if not shots:
        raise ffmpeg_tools.FFmpegError("no shots to render")

    width = ffmpeg_tools.even(int(config.get("width", 1080)))
    height = ffmpeg_tools.even(int(config.get("height", 1920)))
    fps = float(config.get("fps", 30))
    grade = GRADES.get(str(config.get("grade", "none")).lower(), "")
    fade = max(0.0, float(config.get("transition_seconds", 0.0)))
    crossfade = str(config.get("transition", "cut")).lower() == "fade" and fade > 0

    inputs: list[list[str]] = []
    chains: list[str] = []
    labels: list[str] = []

    for index, shot in enumerate(shots):
        if shot.is_image:
            # A still needs a frame rate and a length given to it, since it has
            # neither. Without -framerate the image demuxer assumes 25 and the
            # shot ends up the wrong length.
            inputs.append([
                "-loop", "1", "-framerate", f"{fps:.4f}",
                "-t", f"{shot.duration:.3f}", "-i", str(shot.source),
            ])
            # Zooming alternately in and out stops a row of photos feeling
            # like one repeated move.
            parts = [_still_motion(index, shot.duration, fps, width, height)]
        else:
            # -ss and -t both belong BEFORE -i. After it they are output
            # options, which with several inputs means no shot gets trimmed at
            # all and the montage runs long.
            inputs.append([
                "-ss", f"{shot.start:.3f}", "-t", f"{shot.duration:.3f}", "-i", str(shot.source),
            ])
            info = ffmpeg_tools.probe(shot.source)
            parts = [ffmpeg_tools.vertical_filter(
                info, width, height, str(config.get("framing", "crop")), 0.0
            )]
        if grade:
            parts.append(grade)
        parts += [f"fps={fps:.4f}", "setsar=1", "format=yuv420p"]
        if crossfade:
            out = max(0.0, shot.duration - fade)
            parts.append(f"fade=t=in:st=0:d={fade}")
            parts.append(f"fade=t=out:st={out:.3f}:d={fade}")
        chains.append(f"[{index}:v]{','.join(parts)}[v{index}]")
        labels.append(f"[v{index}]")

    chains.append(f"{''.join(labels)}concat=n={len(shots)}:v=1:a=0[outv]")

    total = sum(shot.duration for shot in shots)
    if music:
        music_index = len(inputs)
        inputs.append(["-stream_loop", "-1", "-i", str(music)])
        chains.append(
            f"[{music_index}:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS,"
            f"volume={max(0.0, music_volume):.3f},"
            f"afade=t=out:st={max(0.0, total - 1.2):.3f}:d=1.2,"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[outa]"
        )
    else:
        silence = len(inputs)
        inputs.append(ffmpeg_tools.silence_input(total))
        chains.append(
            f"[{silence}:a]aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo[outa]"
        )

    ffmpeg_tools.encode(
        inputs=inputs,
        graph=";".join(chains),
        video_label="[outv]",
        audio_label="[outa]",
        destination=destination,
        crf=int(output_config.get("crf", 20)),
        preset=str(output_config.get("preset", "medium")),
        audio_bitrate=str(output_config.get("audio_bitrate", "192k")),
        workdir=workdir,
    )


def _still_motion(index: int, seconds: float, fps: float, width: int, height: int) -> str:
    """Scale a photo past the frame and drift across it for the shot's length.

    A photograph held still for a second and a half reads as a slideshow. The
    same photograph moving slowly reads as a shot, and it costs nothing.
    """
    frames = max(1, int(seconds * fps))
    step = STILL_ZOOM / frames
    top = 1.0 + STILL_ZOOM
    over_w, over_h = ffmpeg_tools.even(width * STILL_OVERSCAN), ffmpeg_tools.even(height * STILL_OVERSCAN)
    zoom = (
        f"min(1+{step:.8f}*on,{top:.4f})" if index % 2 == 0
        else f"max({top:.4f}-{step:.8f}*on,1.0)"
    )
    return (
        f"scale={over_w}:{over_h}:force_original_aspect_ratio=increase,"
        f"crop={over_w}:{over_h},"
        f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d=1:s={ffmpeg_tools.even(width)}x{ffmpeg_tools.even(height)}:fps={fps:.4f}"
    )


def index_text(shots: list[Shot]) -> str:
    """A readable list of what ended up where, for checking the result."""
    lines = ["# montage shot list", ""]
    moment = 0.0
    for number, shot in enumerate(shots, start=1):
        lines.append(
            f"{number:02d}  at {moment:6.2f}s  {shot.duration:4.2f}s  "
            f"{shot.source.name}" + ("  [photo]" if shot.is_image else f" (from {shot.start:.2f}s)")
        )
        moment += shot.duration
    return "\n".join(lines) + "\n"
