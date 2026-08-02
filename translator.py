"""Translation logic using free Google Translate, kept independent of Telegram.

Two kinds of text are shielded from the translation engine and restored
afterwards:

  - code blocks and inline code, which must survive byte-for-byte
  - glossary terms from vocabulary.txt, so the user's jargon and company names
    come out as the exact English they chose rather than a phonetic guess
"""
import asyncio
import logging
import re

from deep_translator import GoogleTranslator

import vocabulary

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

PERSIAN_RE = re.compile(r"[؀-ۿ]")
CODE_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)

# Deliberately alphanumeric and unpunctuated: Google Translate leaves tokens
# like this untouched, whereas bracketed or symbolic markers get mangled.
PLACEHOLDER_PREFIX = "XKEEP"
PLACEHOLDER_SUFFIX = "X"


def is_persian(text: str) -> bool:
    return bool(PERSIAN_RE.search(text))


class TranslationError(Exception):
    pass


class Translator:
    def __init__(self) -> None:
        _, self._glossary = vocabulary.load_entries()
        logger.info("Loaded %d glossary terms for translation", len(self._glossary))

    def _shield(self, text: str) -> tuple[str, list[str]]:
        """Swap code and glossary terms for placeholders before translating."""
        replacements: list[str] = []

        def stash(value: str) -> str:
            replacements.append(value)
            return f"{PLACEHOLDER_PREFIX}{len(replacements) - 1}{PLACEHOLDER_SUFFIX}"

        text = CODE_RE.sub(lambda m: stash(m.group(0)), text)

        # Only pin glossary terms when translating Persian into English; going
        # the other way the English term is already what the user would type.
        if is_persian(text):
            for persian, english in self._glossary.items():
                if persian in text:
                    text = text.replace(persian, stash(english))

        return text, replacements

    @staticmethod
    def _restore(text: str, replacements: list[str]) -> str:
        for index, value in enumerate(replacements):
            # Translation engines sometimes lowercase a token or slip spaces into
            # it, so match loosely rather than trusting an exact string compare.
            pattern = re.compile(
                r"\s*".join(PLACEHOLDER_PREFIX) + r"\s*" + str(index) + r"\s*" + PLACEHOLDER_SUFFIX,
                re.IGNORECASE,
            )
            text = pattern.sub(lambda _: value, text)
        return text

    async def translate(self, text: str) -> str:
        target = "en" if is_persian(text) else "fa"
        shielded, replacements = self._shield(text)

        translated = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                translated = await asyncio.to_thread(
                    GoogleTranslator(source="auto", target=target).translate,
                    shielded,
                )
                break
            except Exception:
                logger.exception("Translation failed (attempt %d/%d)", attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    raise TranslationError("api_error") from None
                await asyncio.sleep(RETRY_BASE_DELAY * attempt)

        if not translated:
            raise TranslationError("empty_response")

        return self._restore(translated, replacements).strip()
