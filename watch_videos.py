"""Watch a folder for videos and edit whatever lands in it.

Drop a file into videos-in/, and the edited video, the subtitles and the short
clips appear in videos-out/. Nothing to click, nothing to configure per video.

Run it with filenames instead to edit those files once and exit:

    python watch_videos.py holiday.mp4 interview.mov
"""
import argparse
import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path

import ffmpeg_tools
import video_config
import video_pipeline
from transcriber import Transcriber
from translator import Translator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("video-editor")

POLL_SECONDS = 3.0

# A file is only picked up once its size has stopped changing, so a video still
# being copied in over a network or off a phone is never edited half-written.
STABLE_CHECKS = 2

# Partial downloads and the stub files editors leave behind.
IGNORED_PREFIXES = (".", "~$")
IGNORED_SUFFIXES = (".part", ".crdownload", ".tmp", ".download")

FFMPEG_HELP = """
ffmpeg was not found on this computer, and the editor cannot do anything without it.

  Windows : open PowerShell and run    winget install Gyan.FFmpeg
            then close and reopen this window so PATH is picked up.
  macOS   : brew install ffmpeg
  Linux   : sudo apt install ffmpeg

Then start this again.
"""


def _ignored(path: Path) -> bool:
    name = path.name
    return (
        name.startswith(IGNORED_PREFIXES)
        or name.lower().endswith(IGNORED_SUFFIXES)
        or not video_pipeline.is_video(path)
    )


class StabilityTracker:
    """Remembers file sizes between polls so growing files are left alone."""

    def __init__(self) -> None:
        self._seen: dict[Path, tuple[int, int]] = {}

    def is_settled(self, path: Path) -> bool:
        try:
            size = path.stat().st_size
        except OSError:
            self._seen.pop(path, None)
            return False
        if size <= 0:
            return False

        last_size, count = self._seen.get(path, (-1, 0))
        count = count + 1 if size == last_size else 0
        self._seen[path] = (size, count)
        return count >= STABLE_CHECKS

    def forget(self, path: Path) -> None:
        self._seen.pop(path, None)


def _archive(source: Path, config: dict) -> None:
    """Move a finished original out of the inbox so it is not edited twice."""
    if not config["output"].get("archive_originals", True):
        return
    done = source.parent / "done"
    done.mkdir(exist_ok=True)
    destination = done / source.name
    if destination.exists():
        destination = done / f"{source.stem}-{int(time.time())}{source.suffix}"
    try:
        shutil.move(str(source), str(destination))
    except OSError:
        logger.warning("Could not move %s into done/; leaving it where it is", source.name)


async def _edit_one(source: Path, output_root: Path, config: dict, transcriber, translator) -> bool:
    destination = output_root / source.stem
    logger.info("=" * 60)
    logger.info("Editing %s", source.name)
    started = time.monotonic()

    try:
        result = await video_pipeline.edit(source, destination, config, transcriber, translator)
    except ffmpeg_tools.FFmpegMissing:
        raise
    except Exception:
        logger.exception("Could not edit %s", source.name)
        return False

    logger.info("Done in %.0f seconds -> %s", time.monotonic() - started, destination)
    for note in result.notes:
        logger.info("  - %s", note)
    logger.info("  %s", result.summary())
    return True


async def watch(config: dict) -> None:
    inbox = video_config.INBOX_DIR
    outbox = video_config.OUTBOX_DIR
    inbox.mkdir(exist_ok=True)
    outbox.mkdir(exist_ok=True)
    video_config.ASSETS_DIR.mkdir(exist_ok=True)

    transcriber = Transcriber(config["whisper"].get("model") or None)
    translator = Translator()

    if config["subtitles"].get("enabled", True):
        logger.info("Loading the speech model (first run downloads it, be patient)...")
        await transcriber.load()

    logger.info("=" * 60)
    logger.info("  READY. Put videos in:  %s", inbox)
    logger.info("  Edited videos appear in: %s", outbox)
    logger.info("  Settings: %s", video_config.CONFIG_FILE.name)
    logger.info("  Press Ctrl+C to stop.")
    logger.info("=" * 60)

    tracker = StabilityTracker()
    while True:
        try:
            candidates = sorted(
                path for path in inbox.iterdir() if path.is_file() and not _ignored(path)
            )
        except OSError:
            logger.warning("Could not read %s", inbox)
            candidates = []

        for path in candidates:
            if not tracker.is_settled(path):
                continue
            tracker.forget(path)
            if await _edit_one(path, outbox, config, transcriber, translator):
                _archive(path, config)
            logger.info("Watching for the next video...")

        await asyncio.sleep(POLL_SECONDS)


async def edit_files(paths: list[Path], config: dict) -> int:
    outbox = video_config.OUTBOX_DIR
    outbox.mkdir(exist_ok=True)

    transcriber = Transcriber(config["whisper"].get("model") or None)
    translator = Translator()

    failures = 0
    for path in paths:
        if not path.exists():
            logger.error("No such file: %s", path)
            failures += 1
            continue
        if not await _edit_one(path, outbox, config, transcriber, translator):
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Edit videos automatically: clean up, subtitle, brand and clip."
    )
    parser.add_argument(
        "videos", nargs="*", type=Path,
        help="Videos to edit once. With none given, watch the videos-in folder.",
    )
    arguments = parser.parse_args()

    config = video_config.load()
    created = video_config.write_default_file()
    if created.exists():
        logger.info("Settings file: %s", created)

    try:
        ffmpeg_tools.ensure_available()
    except ffmpeg_tools.FFmpegMissing:
        print(FFMPEG_HELP)
        return 2

    try:
        if arguments.videos:
            return 1 if asyncio.run(edit_files(arguments.videos, config)) else 0
        asyncio.run(watch(config))
    except KeyboardInterrupt:
        logger.info("Stopped.")
    except ffmpeg_tools.FFmpegMissing:
        print(FFMPEG_HELP)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
