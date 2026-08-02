"""Speech-to-text for Persian voice messages, running locally via faster-whisper.

The model runs entirely on the user's machine: no API key, no regional
restrictions. Model weights are downloaded once and cached for every run after.
"""
import asyncio
import logging
import os

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# large-v3-turbo has near-large accuracy on Persian but a much smaller decoder,
# so it runs several times faster than medium/large on CPU. Override with
# WHISPER_MODEL in .env (small = fastest/least accurate, large-v3 = slowest/best).
DEFAULT_MODEL_SIZE = "large-v3-turbo"

# A natural Persian sentence primes the decoder toward correct Persian spelling
# and punctuation instead of drifting into phonetic nonsense.
PERSIAN_PRIMER = (
    "خب، امروز درباره‌ی کار و پروژه‌هایی که در محیط کارمان انجام می‌دهیم صحبت می‌کنم."
)


class TranscriptionError(Exception):
    pass


class Transcriber:
    def __init__(self, model_size: str | None = None) -> None:
        self._model_size = model_size or os.environ.get("WHISPER_MODEL", DEFAULT_MODEL_SIZE)
        self._model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()

    async def load(self) -> None:
        """Load the model, downloading it on first ever run."""
        async with self._load_lock:
            if self._model is not None:
                return
            logger.info("Loading Whisper model '%s' (first run downloads it)", self._model_size)
            self._model = await asyncio.to_thread(
                WhisperModel,
                self._model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=os.cpu_count() or 4,
            )
            logger.info("Whisper model ready")

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe Persian speech from an audio file into Persian text."""
        await self.load()

        try:
            segments, _ = await asyncio.to_thread(
                self._model.transcribe,
                audio_path,
                language="fa",
                beam_size=5,
                vad_filter=True,
                initial_prompt=PERSIAN_PRIMER,
                # Each chunk decodes independently, so one bad chunk can no longer
                # drag the rest into a repetition loop.
                condition_on_previous_text=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            logger.exception("Transcription failed for %s", audio_path)
            raise TranscriptionError("transcription_failed") from exc
