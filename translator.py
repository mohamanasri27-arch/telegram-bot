"""Translation logic using free Google Translate, kept independent of the Telegram layer.

Code blocks (triple backtick fences) and inline code (single backticks) are
extracted before translation and restored verbatim afterwards, since the
underlying translation engine has no concept of source code and would
otherwise mangle it.
"""
import asyncio
import logging
import re

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

PERSIAN_RE = re.compile(r"[؀-ۿ]")
CODE_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)
PLACEHOLDER_PREFIX = "XCODEBLOCKX"
PLACEHOLDER_SUFFIX = "X"


def is_persian(text: str) -> bool:
    return bool(PERSIAN_RE.search(text))


class TranslationError(Exception):
    pass


class Translator:
    def __init__(self) -> None:
        pass

    async def translate(self, text: str) -> str:
        target = "en" if is_persian(text) else "fa"

        code_blocks = []

        def _stash(match: re.Match) -> str:
            code_blocks.append(match.group(0))
            return f"{PLACEHOLDER_PREFIX}{len(code_blocks) - 1}{PLACEHOLDER_SUFFIX}"

        stashed_text = CODE_RE.sub(_stash, text)

        translated = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                translated = await asyncio.to_thread(
                    GoogleTranslator(source="auto", target=target).translate,
                    stashed_text,
                )
                break
            except Exception:
                logger.exception("Translation failed (attempt %d/%d)", attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    raise TranslationError("api_error") from None
                await asyncio.sleep(RETRY_BASE_DELAY * attempt)

        if not translated:
            raise TranslationError("empty_response")

        for index, block in enumerate(code_blocks):
            translated = translated.replace(f"{PLACEHOLDER_PREFIX}{index}{PLACEHOLDER_SUFFIX}", block)

        return translated.strip()
