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
SENTENCE_ENDINGS = (".", "؟", "!", "…", "?")
CLAUSE_ENDINGS = ("،", ",", "؛", ";", ":")

# A cue shorter than this is a flicker, not a subtitle.
MIN_CUE_SECONDS = 0.7

# How far back the splitter may look for a better place to break than the one
# the length limit lands on.
BREAK_LOOKBACK_WORDS = 5

# A break scoring at least this much is good enough to take early, before any
# limit forces one.
NATURAL_BREAK_SCORE = 3.0


@dataclass(frozen=True)
class TimedText:
    """One subtitle-sized chunk of speech with the seconds it covers."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Cue:
    """A subtitle-sized chunk, with the timing of each word inside it.

    The word timings are what make it possible to highlight words as they are
    spoken; nothing else needs them.
    """

    start: float
    end: float
    text: str
    words: tuple[TimedText, ...] = ()


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

    def _prompt_for(self, language: str, topic: str | None) -> str | None:
        """The sentence that primes the decoder before it hears anything.

        A primer in the right language pushes Whisper towards correct spelling
        instead of phonetic guesses. Naming the subject on top of that is the
        single cheapest accuracy win available: told the video is about
        programming, the decoder stops turning jargon into ordinary words that
        sound similar.
        """
        parts = []
        if language == "fa":
            parts.append(PERSIAN_PRIMER)
        if topic and topic.strip():
            parts.append(topic.strip())
        return " ".join(parts) or None

    async def _decode(
        self,
        audio_path: str,
        language: str,
        word_timestamps: bool,
        topic: str | None = None,
        use_context: bool = False,
    ) -> list:
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
                initial_prompt=self._prompt_for(language, topic),
                hotwords=self._hotwords if persian else None,
                # Off by default: each chunk decodes independently, so one bad
                # chunk cannot drag the rest into a repetition loop. Turning it
                # on gives the decoder the running context, which reads better
                # on clean single-speaker audio and worse on anything noisy.
                condition_on_previous_text=use_context,
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
        topic: str | None = None,
        use_context: bool = False,
    ) -> list[Cue]:
        """Transcribe into short timed chunks suitable for subtitles.

        Whisper's own segments are whole sentences and routinely run past ten
        seconds, which is far too much text to put on screen at once. Word-level
        timestamps let the speech be re-grouped into cues that break where a
        reader would pause.
        """
        segments = await self._decode(
            audio_path, language=language, word_timestamps=True,
            topic=topic, use_context=use_context,
        )

        words: list[TimedText] = []
        for segment in segments:
            segment_words = getattr(segment, "words", None)
            if segment_words:
                for word in segment_words:
                    token = (getattr(word, "word", None) or "").strip()
                    if token:
                        words.append(TimedText(word.start, word.end, token))
            elif (segment.text or "").strip():
                # No word timings available: keep the segment whole rather than
                # inventing timings that would drift out of sync.
                words.append(TimedText(segment.start, segment.end, segment.text.strip()))

        return [
            self._build_cue(group, language)
            for group in _group_words(words, max_chars, max_seconds)
            if group
        ]

    def _build_cue(self, group: list[TimedText], language: str) -> Cue:
        raw = " ".join(word.text for word in group)
        text = self._clean(raw, language)
        start = group[0].start
        end = max(group[-1].end, start + MIN_CUE_SECONDS)
        # Cleaning can merge or drop tokens, which would leave the word timings
        # pointing at the wrong words. Only keep them when the count still
        # matches, since karaoke silently goes out of sync otherwise.
        words = tuple(group) if len(text.split()) == len(group) else ()
        return Cue(start=start, end=end, text=text, words=words)


def _break_score(word: TimedText, following: TimedText | None) -> float:
    """How good a place the end of this word is to end a cue.

    Punctuation and pauses are the two things a reader uses to decide where a
    thought ended, so they are what the score is built from.
    """
    text = word.text.rstrip()
    score = 0.0
    if text.endswith(SENTENCE_ENDINGS):
        score += 3.0
    elif text.endswith(CLAUSE_ENDINGS):
        score += 1.5
    gap = (following.start - word.end) if following else 99.0
    score += min(2.0, max(0.0, gap) * 4.0)
    return score


def _group_words(
    words: list[TimedText], max_chars: int, max_seconds: float
) -> list[list[TimedText]]:
    """Split a stream of timed words into cue-sized groups.

    The greedy version of this — fill until full, then cut — reliably cuts in
    the middle of a phrase, which is what makes automatic subtitles read badly.
    Instead, when a limit is reached the last few words are searched for the
    best place a reader would have broken, and the cut is made there.
    """
    groups: list[list[TimedText]] = []
    buffer: list[TimedText] = []
    length = 0

    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else None
        buffer.append(word)
        length += len(word.text) + (1 if len(buffer) > 1 else 0)

        span = word.end - buffer[0].start
        over_limit = span > max_seconds or length > max_chars

        if over_limit and len(buffer) > 1:
            cut = _choose_break(buffer, words, index)
            groups.append(buffer[:cut])
            buffer = buffer[cut:]
            length = sum(len(item.text) for item in buffer) + max(0, len(buffer) - 1)
            continue

        # A full stop or a real pause ends the cue on its own, as long as there
        # is enough on screen to be worth reading.
        if span >= MIN_CUE_SECONDS and _break_score(word, following) >= NATURAL_BREAK_SCORE:
            groups.append(buffer)
            buffer, length = [], 0

    if buffer:
        groups.append(buffer)
    return groups


def _choose_break(
    buffer: list[TimedText], words: list[TimedText], last_index: int
) -> int:
    """Pick how many words of `buffer` to keep, breaking where a reader would.

    Returns a count, so buffer[:count] becomes the cue and the rest carries
    over. Never returns fewer than two words, since a one-word cue flashing on
    its own is worse than a slightly long one.
    """
    highest = len(buffer)
    lowest = max(2, highest - BREAK_LOOKBACK_WORDS)
    if lowest >= highest:
        return highest

    best_count, best_score = highest, -1.0
    for count in range(lowest, highest + 1):
        word = buffer[count - 1]
        # The word after the break is the next one in the buffer, or the next
        # one in the stream when the break is at the buffer's end.
        if count < len(buffer):
            following = buffer[count]
        else:
            following = words[last_index + 1] if last_index + 1 < len(words) else None
        # Breaking later is mildly preferred, so a marginally better score
        # further back does not throw away half the cue.
        score = _break_score(word, following) + (count / highest) * 0.4
        if score > best_score:
            best_count, best_score = count, score
    return best_count
