"""Download and verify everything the editor needs, once, up front.

The point of this script is that no editing session ever stops to wait for a
download again. It fetches the speech models and the Persian font, checks that
ffmpeg has every filter the pipeline uses, and then edits a video it makes
itself to prove the whole chain works end to end.

Run it after setup.bat, or any time something seems broken:

    python prepare.py               everything
    python prepare.py --check       verify only, download nothing
"""
import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("prepare")

PROJECT_DIR = Path(__file__).parent
FONTS_DIR = PROJECT_DIR / "fonts"

# Every filter the pipeline can reach for. A build missing one of these fails
# only later, in the middle of a render, with a message nobody can act on.
REQUIRED_FILTERS = [
    "ass", "silencedetect", "loudnorm", "sidechaincompress", "amix", "asplit",
    "adelay", "concat", "overlay", "zoompan", "gblur", "eq", "fade", "afade",
    "highpass", "afftdn", "dynaudnorm", "atrim", "trim", "scale", "crop", "pad",
]

# Both are worth having on disk: the accurate one for finished work, the fast
# one for a rough pass or a long recording.
DEFAULT_MODELS = ["large-v3", "large-v3-turbo"]

# Vazirmatn renders Persian far better than Tahoma. Bundling the file rather
# than installing it system-wide avoids needing admin rights, and libass is
# pointed at the folder directly.
FONT_FILES = {
    "Vazirmatn-Regular.ttf": [
        "https://raw.githubusercontent.com/rastikerdar/vazirmatn/v33.003/fonts/ttf/Vazirmatn-Regular.ttf",
        "https://github.com/rastikerdar/vazirmatn/raw/v33.003/fonts/ttf/Vazirmatn-Regular.ttf",
    ],
    "Vazirmatn-Bold.ttf": [
        "https://raw.githubusercontent.com/rastikerdar/vazirmatn/v33.003/fonts/ttf/Vazirmatn-Bold.ttf",
        "https://github.com/rastikerdar/vazirmatn/raw/v33.003/fonts/ttf/Vazirmatn-Bold.ttf",
    ],
}

DOWNLOAD_RETRIES = 5


class Report:
    """Collects the run's results so it ends with one readable summary.

    A warning and a failure are deliberately different things: a missing font
    means Persian looks slightly worse, a missing ffmpeg means nothing works at
    all. Only the second should stop the run or call the setup broken.
    """

    OK, WARN, FAIL = "OK", "WARN", "FAIL"

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def ok(self, message: str) -> None:
        self.lines.append((self.OK, message))
        logger.info("  OK    %s", message)

    def warn(self, message: str) -> None:
        self.lines.append((self.WARN, message))
        logger.warning("  WARN  %s", message)

    def fail(self, message: str) -> None:
        self.lines.append((self.FAIL, message))
        logger.error("  FAIL  %s", message)

    def _of(self, level: str) -> list[str]:
        return [message for kind, message in self.lines if kind == level]

    @property
    def failed(self) -> list[str]:
        return self._of(self.FAIL)

    @property
    def warnings(self) -> list[str]:
        return self._of(self.WARN)


def _section(title: str) -> None:
    logger.info("\n%s\n%s", title, "-" * len(title))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_ffmpeg(report: Report) -> None:
    _section("ffmpeg")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            report.ok(f"{tool} found")
        else:
            report.fail(f"{tool} is NOT installed - run: winget install Gyan.FFmpeg")
            return

    try:
        listing = subprocess.run(
            [shutil.which("ffmpeg"), "-hide_banner", "-filters"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace",
        ).stdout or ""
    except OSError as error:
        report.fail(f"could not ask ffmpeg for its filters: {error}")
        return

    names = {line.split()[1] for line in listing.splitlines() if len(line.split()) > 2}
    missing = [name for name in REQUIRED_FILTERS if name not in names]
    if missing:
        report.fail(f"ffmpeg is missing filters: {', '.join(missing)}")
    else:
        report.ok(f"all {len(REQUIRED_FILTERS)} required filters present")


def check_packages(report: Report) -> None:
    _section("Python packages")
    for module, label in (
        ("faster_whisper", "faster-whisper (speech to text)"),
        ("deep_translator", "deep-translator (translation)"),
    ):
        try:
            __import__(module)
            report.ok(label)
        except ImportError:
            report.fail(f"{label} is missing - run: pip install -r requirements.txt")


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------

def fetch_models(report: Report, models: list[str]) -> None:
    _section("Speech models")
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        report.fail("faster-whisper is not installed, so no model can be fetched")
        return

    for name in models:
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                logger.info("  fetching %s (attempt %d/%d)...", name, attempt, DOWNLOAD_RETRIES)
                path = download_model(name)
                report.ok(f"{name} ready at {path}")
                break
            except Exception as error:
                # These downloads are several gigabytes and routinely drop on a
                # filtered connection. They resume, so retrying is cheap and is
                # very often all that is needed.
                if attempt == DOWNLOAD_RETRIES:
                    report.fail(f"{name} did not finish downloading: {error}")
                else:
                    wait = min(30, 3 * attempt)
                    logger.warning("    dropped (%s); resuming in %ds", type(error).__name__, wait)
                    time.sleep(wait)


def use_bundled_font(report: Report) -> None:
    """Point the settings at Vazirmatn, but only once it is really there."""
    if not (FONTS_DIR / "Vazirmatn-Regular.ttf").exists():
        return
    import video_config
    config = video_config.load()
    fonts = config["subtitles"].setdefault("fonts", {})
    if fonts.get("fa") == "Vazirmatn":
        report.ok("Persian subtitles already set to Vazirmatn")
        return
    fonts["fa"] = "Vazirmatn"
    video_config.save(config)
    report.ok("Persian subtitles switched to Vazirmatn")


def fetch_fonts(report: Report) -> None:
    _section("Fonts")
    FONTS_DIR.mkdir(exist_ok=True)
    for name, urls in FONT_FILES.items():
        destination = FONTS_DIR / name
        if destination.exists() and destination.stat().st_size > 10_000:
            report.ok(f"{name} already present")
            continue

        last_error: Exception | None = None
        for url in urls:
            for attempt in range(1, 3):
                try:
                    logger.info("  fetching %s...", name)
                    with urllib.request.urlopen(url, timeout=60) as response:
                        data = response.read()
                    if len(data) < 10_000:
                        raise ValueError("file looks truncated")
                    destination.write_bytes(data)
                    report.ok(f"{name} downloaded")
                    break
                except (urllib.error.URLError, ValueError, OSError) as error:
                    last_error = error
                    time.sleep(attempt)
            if destination.exists():
                break
        else:
            # Tahoma still works, so this is a downgrade and not a stop.
            report.warn(f"{name} unavailable ({last_error}) - Tahoma will be used instead")


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def self_test(report: Report) -> None:
    _section("Self-test")
    import asyncio

    import ffmpeg_tools
    import video_config
    import video_pipeline

    with tempfile.TemporaryDirectory(dir=str(PROJECT_DIR)) as raw:
        work = Path(raw)
        source = work / "selftest.mp4"
        try:
            subprocess.run(
                [
                    shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            report.ok("test footage generated")
        except (subprocess.CalledProcessError, OSError) as error:
            report.fail(f"could not generate test footage: {error}")
            return

        # Stand-ins for the model and the translator: this test is about the
        # ffmpeg chain, and running the real ones would make it slow and would
        # fail whenever the network is down rather than when something is
        # actually broken.
        class Cues:
            async def load(self):
                pass

            async def transcribe_cues(self, path, **kwargs):
                from transcriber import Cue, TimedText
                return [
                    Cue(1.0, 4.0, "خط اول آزمایشی",
                        (TimedText(1.0, 2.3, "خط"), TimedText(2.4, 3.2, "اول"),
                         TimedText(3.3, 4.0, "آزمایشی"))),
                    Cue(5.0, 8.0, "خط دوم آزمایشی.",
                        (TimedText(5.0, 6.3, "خط"), TimedText(6.4, 7.2, "دوم"),
                         TimedText(7.3, 8.0, "آزمایشی.")))
                ]

        class Words:
            async def translate_many(self, texts, target="en", batch_size=20):
                return [f"test line {index + 1}" for index in range(len(texts))]

        config = video_config.load()
        config["output"].update({"preset": "ultrafast", "crf": 30})
        config["subtitles"].update({"burn": ["fa", "en"], "animation": "karaoke"})
        config["promo"].update({"hook_text": "آزمایش", "end_card_text": "@test",
                                "end_card_seconds": 1.0, "punch_in": 0.04})
        config["clips"].update({"enabled": True, "count": 1, "min_source_seconds": 5,
                                "min_seconds": 4, "max_seconds": 8})

        try:
            result = asyncio.run(
                video_pipeline.edit(source, work / "out", config, Cues(), Words())
            )
        except Exception as error:
            report.fail(f"the pipeline raised: {error}")
            return

        if result.edited and result.edited.exists():
            info = ffmpeg_tools.probe(result.edited)
            report.ok(f"edited video rendered ({info.width}x{info.height}, {info.duration:.0f}s)")
        else:
            report.fail("no edited video was produced")
        if result.clips:
            report.ok(f"{len(result.clips)} vertical clip(s) cut")
        else:
            report.fail("no clips were cut")
        if result.subtitle_files:
            report.ok(f"subtitles written: {', '.join(p.suffixes[0].lstrip('.') for p in result.subtitle_files)}")
        else:
            report.fail("no subtitle files were written")


def check_model_loads(report: Report, model: str) -> None:
    _section("Loading the speech model")
    try:
        from faster_whisper import WhisperModel
        started = time.monotonic()
        WhisperModel(model, device="cpu", compute_type="int8")
        report.ok(f"{model} loaded in {time.monotonic() - started:.0f}s")
    except Exception as error:
        report.fail(f"{model} would not load: {error}")


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify everything the video editor needs."
    )
    parser.add_argument("--check", action="store_true",
                        help="verify what is installed, download nothing")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help=f"speech models to fetch (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--skip-selftest", action="store_true",
                        help="skip the end-to-end render test")
    arguments = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Instagram Reels Editor - preparing")
    logger.info("=" * 60)

    report = Report()
    check_ffmpeg(report)
    check_packages(report)

    if not arguments.check:
        fetch_fonts(report)
        use_bundled_font(report)
        fetch_models(report, arguments.models)
        if arguments.models:
            check_model_loads(report, arguments.models[0])

    if not arguments.skip_selftest and not report.failed:
        self_test(report)
    elif report.failed:
        logger.warning("\nSkipping the self-test: fix the failures above first.")

    logger.info("\n%s", "=" * 60)
    if report.failed:
        logger.error("  NOT READY - %d problem(s) that stop it working:", len(report.failed))
        for line in report.failed:
            logger.error("    - %s", line)
        logger.info("=" * 60)
        return 1

    logger.info("  READY. Everything is downloaded and working.")
    logger.info("  Nothing will need downloading again.")
    if report.warnings:
        logger.info("")
        logger.info("  %d thing(s) worth knowing, none of them blocking:", len(report.warnings))
        for line in report.warnings:
            logger.info("    - %s", line)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
