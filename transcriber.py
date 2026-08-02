"""Speech-to-text for Persian voice messages, running locally via faster-whisper.

The model runs entirely on the user's machine: no API key, no regional
restrictions. Model weights are downloaded once and cached for every run after.

Domain vocabulary from vocabulary.txt is fed to the decoder as hotwords, which
is what keeps work-specific jargon and company names from being mangled.
"""
import asyncio
import logging
import os

from faster_whisper import WhisperModel

import persian_text
import settings
import vocabulary

logger = logging.getLogger(__name__)

# Accuracy first: large-v3 keeps all 32 decoder layers, where turbo keeps only 4.
# That decoder is what gets Persian verb conjugation right, so the extra time is
# worth it. Switch to the faster model any time with /accuracy.
DEFAULT_MODEL_SIZE = "large-v3"

# A natural Persian sentence primes the decoder toward correct Persian spelling
# and punctuation instead of drifting into phonetic nonsense.
PERSIAN_PRIMER = (
    "خب، امروز درباره‌ی کار و پروژه‌هایی که در محیط کارمان انجام می‌دهیم صحبت می‌کنم."
)


class TranscriptionError(Exception):
    pass


class Transcriber:
    def __init__(self, model_size: str | None = None) -> None:
        self._model_size = (
            model_size
            or os.environ.get("WHISPER_MODEL")
            or settings.get("model")
            or DEFAULT_MODEL_SIZE
        )
        self._model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()
        self._hotwords: str | None = None

    @property
    def model_size(self) -> str:
        return self._model_size

    async def switch_model(self, model_size: str) -> None:
        """Drop the current model and load a different one."""
        if model_size == self._model_size and self._model is not None:
            return
        async with self._load_lock:
            self._model = None
            self._model_size = model_size
        await self.load()

    def reload_hotwords(self) -> None:
        """Pick up terms added since startup without restarting the bot."""
        self._hotwords = vocabulary.load_hotwords()

    async def load(self) -> None:
        """Load the model, downloading it on first ever run."""
        async with self._load_lock:
            if self._model is not None:
                return

            self._hotwords = vocabulary.load_hotwords()

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
                # A wider beam explores more spellings before committing, which
                # is where most of the remaining word-level errors come from.
                beam_size=10,
                # Let the beam keep searching past the first finished candidate.
                patience=2.0,
                # Whisper's own recovery ladder: a chunk that decodes badly at 0
                # is retried at higher temperatures instead of being kept as-is.
                # Pinning this to a single 0.0 silently disabled that retry.
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                vad_filter=True,
                # Keep the default 2s silence gap so sentences stay whole, but pad
                # each speech chunk a bit more so quiet word edges aren't clipped.
                vad_parameters={"speech_pad_ms": 600},
                initial_prompt=PERSIAN_PRIMER,
                hotwords=self._hotwords,
                # Each chunk decodes independently, so one bad chunk can no longer
                # drag the rest into a repetition loop.
                condition_on_previous_text=False,
            )
            raw = " ".join(segment.text.strip() for segment in segments)
            cleaned = persian_text.normalize(raw)
            if settings.get("clean_fillers"):
                cleaned = persian_text.remove_fillers(cleaned)
            return cleaned
        except Exception as exc:
            logger.exception("Transcription failed for %s", audio_path)
            raise TranscriptionError("transcription_failed") from exc
