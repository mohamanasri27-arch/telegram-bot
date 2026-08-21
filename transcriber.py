"""Speech-to-text for Persian voice messages, running locally via faster-whisper.

The model runs entirely on the user's machine: no API key, no regional
restrictions. Model weights are downloaded once and cached for every run after.

Domain vocabulary from vocabulary.txt is fed to the decoder as hotwords, which
is what keeps work-specific jargon and company names from being mangled.

Two shapes of output are offered. transcribe() returns one block of text, which
is what the Telegram bot sends back. transcribe_cues() returns short timed
chunks, which is what the video editor turns into subtitles.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass

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

# A pause this long inside a sentence is a natural place to start a new cue.
CUE_BREAK_GAP_SECONDS = 0.55

# Ending a cue here reads far better than breaking mid-clause.
SENTENCE_ENDINGS = (".", "؟", "!", "؛", "…", "?")


@dataclass(frozen=True)
class TimedText:
    """One subtitle-sized chunk of speech with the seconds it covers."""

    start: float
    end: float
    text: str


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

            logger.info(
                "Loading Whisper model '%s'. On first use this downloads the "
                "weights, and the bot will not answer messages until it finishes.",
                self._model_size,
            )
            started = time.monotonic()
            self._model = await asyncio.to_thread(
                WhisperModel,
                self._model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=os.cpu_count() or 4,
            )
            logger.info("Whisper model ready (took %.0f seconds)", time.monotonic() - started)

    async def _decode(self, audio_path: str, language: str, word_timestamps: bool) -> list:
        """Run the decoder and return faster-whisper's own segment objects."""
        await self.load()

        persian = language == "fa"
        try:
            segments, _ = await asyncio.to_thread(
                self._model.transcribe,
                audio_path,
                language=language,
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
                # Both of these are tuned for Persian, so they are only helpful
                # when Persian is what is being decoded.
                initial_prompt=PERSIAN_PRIMER if persian else None,
                hotwords=self._hotwords if persian else None,
                # Each chunk decodes independently, so one bad chunk can no longer
                # drag the rest into a repetition loop.
                condition_on_previous_text=False,
                word_timestamps=word_timestamps,
            )
            # The generator does the real work, so drain it inside the thread's
            # error handling rather than leaving it to explode further up.
            return list(segments)
        except Exception as exc:
            logger.exception("Transcription failed for %s", audio_path)
            raise TranscriptionError("transcription_failed") from exc

    def _clean(self, text: str, language: str) -> str:
        """Apply the Persian tidy-ups, which do not apply to other languages."""
        if language != "fa":
            return " ".join(text.split())
        cleaned = persian_text.normalize(text)
        if settings.get("clean_fillers"):
            cleaned = persian_text.remove_fillers(cleaned)
        return cleaned

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe Persian speech from an audio file into Persian text."""
        segments = await self._decode(audio_path, language="fa", word_timestamps=False)
        raw = " ".join(segment.text.strip() for segment in segments)
        return self._clean(raw, "fa")

    async def transcribe_cues(
        self,
        audio_path: str,
        *,
        language: str = "fa",
        max_chars: int = 84,
        max_seconds: float = 4.5,
    ) -> list[TimedText]:
        """Transcribe into short timed chunks suitable for subtitles.

        Whisper's own segments are whole sentences and routinely run past ten
        seconds, which is far too much text to put on screen at once. Word-level
        timestamps let the speech be re-grouped into cues that break at pauses
        and sentence endings instead of at arbitrary points.
        """
        segments = await self._decode(audio_path, language=language, word_timestamps=True)

        words = []
        for segment in segments:
            segment_words = getattr(segment, "words", None)
            if segment_words:
                words.extend(segment_words)
            elif (segment.text or "").strip():
                # No word timings available: keep the segment whole rather than
                # inventing timings that would drift out of sync.
                words.append(
                    TimedText(segment.start, segment.end, segment.text.strip())
                )

        cues: list[TimedText] = []
        buffer: list[str] = []
        start: float | None = None
        end: float = 0.0

        def flush() -> None:
            nonlocal buffer, start
            if buffer and start is not None:
                text = self._clean("".join(buffer), language)
                if text:
                    cues.append(TimedText(start, max(end, start + 0.4), text))
            buffer = []
            start = None

        for word in words:
            token = getattr(word, "word", None)
            if token is None:
                token = word.text
            if not token.strip():
                continue

            gap = word.start - end if start is not None else 0.0
            too_long = start is not None and (word.end - start) > max_seconds
            too_wide = sum(len(part) for part in buffer) + len(token) > max_chars
            if start is not None and (gap > CUE_BREAK_GAP_SECONDS or too_long or too_wide):
                flush()

            if start is None:
                start = word.start
            buffer.append(token)
            end = word.end

            if token.rstrip().endswith(SENTENCE_ENDINGS):
                flush()

        flush()
        return cues
