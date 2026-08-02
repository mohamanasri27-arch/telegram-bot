"""Speech-to-text for Persian voice messages, running locally via faster-whisper.

The model runs entirely on the user's machine: no API key, no network calls,
no regional restrictions. The first run downloads the model weights once and
caches them for every run afterwards.
"""
import asyncio
import logging
import os

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# "small" balances Persian accuracy against CPU speed. Set WHISPER_MODEL=medium
# in .env for noticeably better accuracy at roughly 3x the transcription time.
DEFAULT_MODEL_SIZE = "small"


class TranscriptionError(Exception):
    pass


class Transcriber:
    def __init__(self, model_size: str | None = None) -> None:
        self._model_size = model_size or os.environ.get("WHISPER_MODEL", DEFAULT_MODEL_SIZE)
        self._model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()

    async def _get_model(self) -> WhisperModel:
        """Load the model on first use so bot startup stays fast."""
        async with self._load_lock:
            if self._model is None:
                logger.info("Loading Whisper model '%s' (first run downloads it)", self._model_size)
                self._model = await asyncio.to_thread(
                    WhisperModel, self._model_size, device="cpu", compute_type="int8"
                )
                logger.info("Whisper model loaded")
        return self._model

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe Persian speech from an audio file into Persian text."""
        model = await self._get_model()

        try:
            segments, _ = await asyncio.to_thread(
                model.transcribe,
                audio_path,
                language="fa",
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            logger.exception("Transcription failed for %s", audio_path)
            raise TranscriptionError("transcription_failed") from exc
