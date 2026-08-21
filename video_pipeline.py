"""The edit itself: what happens to a video, in what order, and why.

The work is deliberately split into two encodes rather than one per feature.

  1. The body pass cuts the silence out and fixes colour and loudness in the
     same encode, because the cut has to re-encode anyway.
  2. The finish pass burns the subtitles, lays the music under the speech,
     stamps the logo, joins the intro and outro on and fades the ends.

Everything downstream reads its timings from the body, so the transcript is
made after the cutting and before the finishing. Transcribing the original
would produce subtitles that drift further out of sync with every cut, and
subtitling before the intro is joined on is what keeps the cues from having to
be shifted afterwards.
"""
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import clipper
import ffmpeg_tools
import subtitles
import video_config
from subtitles import SubtitleCue

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv", ".flv"}


@dataclass
class EditResult:
    source: Path
    output_dir: Path
    edited: Path | None = None
    clips: list[Path] = field(default_factory=list)
    subtitle_files: list[Path] = field(default_factory=list)
    transcript: Path | None = None
    thumbnail: Path | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"edited: {self.edited.name}" if self.edited else "edited: none"]
        if self.clips:
            parts.append(f"{len(self.clips)} clips")
        if self.subtitle_files:
            parts.append(", ".join(path.name for path in self.subtitle_files))
        return " | ".join(parts)


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


# --------------------------------------------------------------------------
# Stage 1: the body
# --------------------------------------------------------------------------

def _body_filters(config: dict, info: ffmpeg_tools.MediaInfo) -> tuple[str, str]:
    cleanup = config["cleanup"]

    video_parts: list[str] = []
    target_height = config["output"].get("height")
    if target_height:
        scale = ffmpeg_tools.scale_filter(int(target_height), info)
        if scale:
            video_parts.append(scale)
    if cleanup.get("color_polish", True):
        # A lift, not a look. Anything stronger starts making skin tones lie.
        video_parts.append("eq=contrast=1.05:saturation=1.07:gamma=1.02")
    video_parts.append("setsar=1")

    audio_parts: list[str] = []
    if cleanup.get("normalize_audio", True):
        target = float(cleanup.get("target_loudness_lufs", -14.0))
        # Single pass: a two-pass measurement is more exact, but it doubles the
        # decode time of the longest stage for a difference nobody can hear.
        audio_parts.append(f"loudnorm=I={target}:TP=-1.5:LRA=11")

    return ",".join(video_parts), ",".join(audio_parts)


def _build_body(source: Path, destination: Path, config: dict, info: ffmpeg_tools.MediaInfo) -> str:
    """Cut, colour-correct and level the source into the working body file."""
    cleanup = config["cleanup"]
    output = config["output"]

    ranges = [(0.0, info.duration)]
    note = "kept full length"

    if cleanup.get("remove_silence", True) and info.has_audio and info.duration > 0:
        silences = ffmpeg_tools.detect_silences(
            source,
            float(cleanup.get("silence_threshold_db", -34.0)),
            float(cleanup.get("min_silence_seconds", 0.8)),
            info.duration,
        )
        if silences:
            ranges = ffmpeg_tools.keep_ranges(
                silences,
                info.duration,
                padding=float(cleanup.get("keep_padding_seconds", 0.18)),
                trim_start=bool(cleanup.get("trim_start", True)),
                trim_end=bool(cleanup.get("trim_end", True)),
            )
            kept = sum(end - start for start, end in ranges)
            note = (
                f"cut {info.duration - kept:.0f}s of silence "
                f"in {len(ranges)} pieces ({info.duration:.0f}s to {kept:.0f}s)"
            )

    video_filter, audio_filter = _body_filters(config, info)
    ffmpeg_tools.cut_to_ranges(
        source,
        destination,
        ranges,
        info,
        crf=int(output.get("crf", 20)),
        preset=str(output.get("preset", "medium")),
        audio_bitrate=str(output.get("audio_bitrate", "192k")),
        video_filter=video_filter or None,
        audio_filter=audio_filter or None,
        # From here on every stage can assume there is an audio track to work
        # with, which removes a branch from all of them.
        silent_audio=True,
    )
    return note


# --------------------------------------------------------------------------
# Stage 2: the transcript
# --------------------------------------------------------------------------

async def _transcribe(
    body: Path, workdir: Path, config: dict, transcriber, translator
) -> list[SubtitleCue]:
    """Transcribe the cut body, and translate it if English is wanted.

    The `persian` field holds whatever language was spoken and `english` holds
    its translation, which is what the two subtitle styles map onto.
    """
    sub_config = config["subtitles"]
    speech = workdir / "speech.wav"
    ffmpeg_tools.extract_audio(body, speech)

    timed = await transcriber.transcribe_cues(
        str(speech),
        language=str(config.get("language", "fa")),
        max_chars=int(sub_config.get("max_chars_per_line", 42))
        * max(1, int(sub_config.get("max_lines", 2))),
        max_seconds=float(sub_config.get("max_cue_seconds", 4.5)),
    )
    cues = [SubtitleCue(item.start, item.end, persian=item.text) for item in timed]

    if cues and sub_config.get("translate_english", True):
        logger.info("Translating %d subtitle cues into English", len(cues))
        english = await translator.translate_many([cue.persian for cue in cues])
        for cue, translation in zip(cues, english):
            cue.english = translation

    return cues


# --------------------------------------------------------------------------
# Stage 3: the finish
# --------------------------------------------------------------------------

class _GraphBuilder:
    """Accumulates ffmpeg inputs and filter chains while indices stay in step.

    Input numbering is positional, so every optional piece that adds an input
    shifts the numbering of everything after it. Handing out the index at the
    moment the input is added is what stops that from going wrong.
    """

    def __init__(self, first_input: Path) -> None:
        self.inputs: list[list[str]] = [["-i", str(first_input)]]
        self.chains: list[str] = []

    def add_input(self, args: list[str]) -> int:
        self.inputs.append(args)
        return len(self.inputs) - 1

    def add(self, chain: str) -> None:
        self.chains.append(chain)

    @property
    def graph(self) -> str:
        return ";".join(self.chains)


_AUDIO_FORMAT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


def _normalise_segment(
    builder: _GraphBuilder, path: Path, width: int, height: int, fps: float, tag: str
) -> tuple[str, str, float]:
    """Bring an intro or outro up to the body's format so concat will accept it."""
    info = ffmpeg_tools.probe(path)
    index = builder.add_input(["-i", str(path)])

    video_label = f"[{tag}v]"
    builder.add(
        f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={fps:.4f},format=yuv420p{video_label}"
    )

    if info.has_audio:
        audio_source = f"[{index}:a]"
    else:
        # concat refuses a stream layout that varies between segments, so a
        # silent intro still needs a silent *track*.
        silent = builder.add_input([
            "-f", "lavfi", "-t", f"{max(0.1, info.duration):.3f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        ])
        audio_source = f"[{silent}:a]"

    audio_label = f"[{tag}a]"
    builder.add(f"{audio_source}{_AUDIO_FORMAT}{audio_label}")
    return video_label, audio_label, info.duration


def _finish(
    body: Path,
    destination: Path,
    cues: list[SubtitleCue],
    config: dict,
    body_info: ffmpeg_tools.MediaInfo,
    workdir: Path,
) -> list[str]:
    """Burn subtitles, add music, logo, intro/outro and fades in one encode."""
    branding = config["branding"]
    sub_config = config["subtitles"]
    output = config["output"]
    notes: list[str] = []

    width, height, fps = body_info.width, body_info.height, body_info.fps
    builder = _GraphBuilder(body)

    # --- picture -----------------------------------------------------------
    video_chain = [f"fps={fps:.4f}", "setsar=1"]
    burn_mode = str(sub_config.get("burn", subtitles.BURN_PERSIAN))
    if sub_config.get("enabled", True) and burn_mode != subtitles.BURN_NONE and cues:
        ass_name = "main.ass"
        subtitles.write_ass(cues, workdir / ass_name, sub_config, width, height, burn_mode)
        # Relative name plus cwd=workdir: a Windows absolute path inside a
        # filter would need its colon and backslashes escaped three times over.
        video_chain.append(f"ass={ass_name}")
        notes.append(f"burned {len(cues)} {burn_mode} subtitle cues")
    video_chain.append("format=yuv420p")
    builder.add(f"[0:v]{','.join(video_chain)}[bodyv]")
    body_video = "[bodyv]"

    logo = video_config.asset_path(str(branding.get("logo", "")))
    if logo:
        index = builder.add_input(["-loop", "1", "-i", str(logo)])
        logo_width = ffmpeg_tools.even(width * float(branding.get("logo_width_pct", 9.0)) / 100)
        opacity = max(0.0, min(1.0, float(branding.get("logo_opacity", 0.75))))
        builder.add(
            f"[{index}:v]scale={logo_width}:-2,format=rgba,"
            f"colorchannelmixer=aa={opacity:.3f}[logo]"
        )
        position = ffmpeg_tools.logo_overlay_position(str(branding.get("logo_position", "top-right")))
        # shortest=1 ends the overlay with the body, since -loop 1 never ends.
        builder.add(f"{body_video}[logo]overlay={position}:shortest=1[bodyvl]")
        body_video = "[bodyvl]"
        notes.append("stamped the logo")

    # --- sound -------------------------------------------------------------
    builder.add(f"[0:a]{_AUDIO_FORMAT}[speech]")
    body_audio = "[speech]"

    music = video_config.asset_path(str(branding.get("music", "")))
    if music:
        duck = bool(branding.get("duck_music", True))
        if duck:
            # The speech is needed twice: once to be heard, once to tell the
            # compressor when to pull the music down.
            builder.add("[speech]asplit=2[speechout][speechkey]")
            speech_label, key_label = "[speechout]", "[speechkey]"
        else:
            speech_label, key_label = "[speech]", None

        index = builder.add_input(["-stream_loop", "-1", "-i", str(music)])
        volume = max(0.0, float(branding.get("music_volume", 0.12)))
        builder.add(f"[{index}:a]{_AUDIO_FORMAT},volume={volume:.3f}[music]")

        music_label = "[music]"
        if key_label:
            builder.add(
                f"[music]{key_label}sidechaincompress="
                f"threshold=0.03:ratio=12:attack=15:release=350[musicduck]"
            )
            music_label = "[musicduck]"

        # duration=first stops the looped music from running forever;
        # normalize=0 keeps amix from halving the speech to make room.
        builder.add(
            f"{speech_label}{music_label}amix=inputs=2:duration=first:normalize=0[bodya]"
        )
        body_audio = "[bodya]"
        notes.append("laid music under the speech" + (" with ducking" if duck else ""))

    # --- intro and outro ---------------------------------------------------
    segments: list[tuple[str, str]] = []
    total_duration = body_info.duration

    intro = video_config.asset_path(str(branding.get("intro", "")))
    if intro:
        video_label, audio_label, duration = _normalise_segment(
            builder, intro, width, height, fps, "intro"
        )
        segments.append((video_label, audio_label))
        total_duration += duration
        notes.append("added the intro")

    segments.append((body_video, body_audio))

    outro = video_config.asset_path(str(branding.get("outro", "")))
    if outro:
        video_label, audio_label, duration = _normalise_segment(
            builder, outro, width, height, fps, "outro"
        )
        segments.append((video_label, audio_label))
        total_duration += duration
        notes.append("added the outro")

    if len(segments) > 1:
        joined = "".join(f"{video}{audio}" for video, audio in segments)
        builder.add(f"{joined}concat=n={len(segments)}:v=1:a=1[joinedv][joineda]")
        final_video, final_audio = "[joinedv]", "[joineda]"
    else:
        final_video, final_audio = segments[0]

    # --- fades -------------------------------------------------------------
    fade_in = max(0.0, float(branding.get("fade_in_seconds", 0.4)))
    fade_out = max(0.0, float(branding.get("fade_out_seconds", 0.8)))
    video_fades, audio_fades = [], []
    if fade_in > 0:
        video_fades.append(f"fade=t=in:st=0:d={fade_in}")
        audio_fades.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0 and total_duration > fade_out:
        start = total_duration - fade_out
        video_fades.append(f"fade=t=out:st={start:.3f}:d={fade_out}")
        audio_fades.append(f"afade=t=out:st={start:.3f}:d={fade_out}")

    builder.add(f"{final_video}{','.join(video_fades) if video_fades else 'null'}[outv]")
    builder.add(f"{final_audio}{','.join(audio_fades) if audio_fades else 'anull'}[outa]")

    ffmpeg_tools.encode(
        inputs=builder.inputs,
        graph=builder.graph,
        video_label="[outv]",
        audio_label="[outa]",
        destination=destination,
        crf=int(output.get("crf", 20)),
        preset=str(output.get("preset", "medium")),
        audio_bitrate=str(output.get("audio_bitrate", "192k")),
        workdir=workdir,
    )
    return notes


# --------------------------------------------------------------------------
# The whole edit
# --------------------------------------------------------------------------

async def edit(
    source: Path,
    output_dir: Path,
    config: dict,
    transcriber,
    translator,
) -> EditResult:
    """Run every enabled stage against one video and return what was produced."""
    ffmpeg_tools.ensure_available()

    info = ffmpeg_tools.probe(source)
    if not info.has_video:
        raise ffmpeg_tools.FFmpegError(f"{source.name} has no video track")
    if info.duration <= 0:
        raise ffmpeg_tools.FFmpegError(f"{source.name} has no readable duration")

    result = EditResult(source=source, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = output_dir / ".work"
    workdir.mkdir(exist_ok=True)
    stem = source.stem

    logger.info(
        "%s: %.0fs, %dx%d, %.2f fps", source.name, info.duration, info.width, info.height, info.fps
    )

    body = workdir / "body.mp4"
    logger.info("Cleaning up the picture and sound...")
    result.notes.append(_build_body(source, body, config, info))
    body_info = ffmpeg_tools.probe(body)

    cues: list[SubtitleCue] = []
    if config["subtitles"].get("enabled", True):
        logger.info("Transcribing the speech (this is the slow part)...")
        try:
            cues = await _transcribe(body, workdir, config, transcriber, translator)
            result.notes.append(f"transcribed {len(cues)} cues")
        except Exception:
            # Losing the subtitles should not lose the edit.
            logger.exception("Transcription failed; continuing without subtitles")
            result.notes.append("transcription failed, no subtitles")

    if cues and config["subtitles"].get("write_srt", True):
        for language, suffix in (("fa", f"{stem}.fa.srt"), ("en", f"{stem}.en.srt")):
            written = subtitles.write_srt(cues, output_dir / suffix, language)
            if written:
                result.subtitle_files.append(written)

    if cues and config["output"].get("write_transcript", True):
        transcript = output_dir / f"{stem}.transcript.txt"
        body_text = subtitles.transcript_text(cues, "fa")
        english_text = subtitles.transcript_text(cues, "en")
        if english_text:
            body_text = f"{body_text}\n\n---\n\n{english_text}"
        transcript.write_text(body_text + "\n", encoding="utf-8-sig")
        result.transcript = transcript

    logger.info("Finishing: subtitles, music, branding...")
    edited = output_dir / f"{stem}-edited.mp4"
    result.notes.extend(_finish(body, edited, cues, config, body_info, workdir))
    result.edited = edited

    clip_config = config["clips"]
    if clip_config.get("enabled", True) and body_info.duration >= float(
        clip_config.get("min_source_seconds", 90.0)
    ):
        highlights = clipper.pick_highlights(cues, body_info.duration, clip_config)
        if highlights:
            logger.info("Cutting %d vertical clips...", len(highlights))
            clip_dir = output_dir / f"{stem}-clips"
            clip_dir.mkdir(exist_ok=True)
            for number, highlight in enumerate(highlights, start=1):
                destination = clip_dir / f"clip-{number:02d}.mp4"
                try:
                    clipper.render_clip(
                        body, destination, highlight, cues, clip_config,
                        body_info, config["output"], config["subtitles"], workdir,
                    )
                    result.clips.append(destination)
                except ffmpeg_tools.FFmpegError:
                    logger.exception("Could not cut clip %d", number)
            if result.clips:
                (clip_dir / "clips.txt").write_text(
                    clipper.index_text(highlights), encoding="utf-8-sig"
                )
                result.notes.append(f"cut {len(result.clips)} vertical clips")

    try:
        cover = output_dir / f"{stem}-cover.jpg"
        ffmpeg_tools.thumbnail(edited, cover, min(1.5, body_info.duration / 2))
        result.thumbnail = cover
    except ffmpeg_tools.FFmpegError:
        logger.debug("No cover frame could be taken", exc_info=True)

    shutil.rmtree(workdir, ignore_errors=True)
    return result
