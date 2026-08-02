"""Shared loader for the term files that drive accuracy.

Two files, because the two mechanisms have very different size limits:

  vocabulary.txt — your own jargon. Feeds Whisper's `hotwords`, which the
      decoder truncates at 223 tokens (~30 Persian terms). Anything past that
      is silently discarded, so this file must stay short.

  glossary.txt — the big Persian→English dictionary. Used only for string
      substitution during translation, so it has no size limit at all.

Entries in either file may be written as "فارسی = English" to pin the
translation; a bare Persian line only affects speech recognition.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VOCABULARY_FILE = Path(__file__).with_name("vocabulary.txt")
GLOSSARY_FILE = Path(__file__).with_name("glossary.txt")

# Terms added from Telegram with /add. Kept out of version control so updating
# the bot never overwrites what the user has taught it.
MY_TERMS_FILE = Path(__file__).with_name("my_terms.txt")

# Whisper truncates hotwords at 448 // 2 - 1 tokens. Persian averages roughly
# 4 tokens per word, so this character budget keeps us clear of the cliff.
HOTWORDS_CHAR_BUDGET = 450


def _parse(path: Path, warn_if_missing: bool = True) -> tuple[list[str], dict[str, str]]:
    if not path.exists():
        if warn_if_missing:
            logger.warning("%s not found; continuing without it", path.name)
        return [], {}

    terms: list[str] = []
    glossary: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            persian, english = line.split("=", 1)
            persian, english = persian.strip(), english.strip()
            if persian and english:
                terms.append(persian)
                glossary[persian] = english
        else:
            terms.append(line)

    return terms, glossary


def load_glossary() -> dict[str, str]:
    """Every pinned translation, from all three files.

    Later sources win, so a term the user taught with /add overrides both the
    shipped dictionary and the project file. Longer phrases are matched before
    shorter ones nested inside them.
    """
    _, general = _parse(GLOSSARY_FILE)
    _, personal = _parse(VOCABULARY_FILE)
    _, mine = _parse(MY_TERMS_FILE, warn_if_missing=False)

    combined = {**general, **personal, **mine}
    ordered = dict(sorted(combined.items(), key=lambda kv: len(kv[0]), reverse=True))
    logger.info(
        "Glossary: %d general + %d personal + %d added = %d total",
        len(general), len(personal), len(mine), len(ordered),
    )
    return ordered


def load_my_terms() -> dict[str, str]:
    """Just the terms the user taught the bot with /add."""
    _, mine = _parse(MY_TERMS_FILE, warn_if_missing=False)
    return mine


def add_term(persian: str, english: str) -> None:
    """Append a term learned from Telegram, creating the file if needed."""
    persian, english = persian.strip(), english.strip()
    if not MY_TERMS_FILE.exists():
        MY_TERMS_FILE.write_text(
            "# واژه‌هایی که با دستور /add اضافه کرده‌اید.\n"
            "# این فایل موقع به‌روزرسانی بات پاک نمی‌شود.\n\n",
            encoding="utf-8",
        )
    with MY_TERMS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"{persian} = {english}\n")
    logger.info("Added term: %s = %s", persian, english)


def load_hotwords() -> str | None:
    """User terms only, trimmed to what Whisper will actually read.

    Terms added with /add come first: they are the ones the user just corrected,
    so they are the most important to fit inside the truncation limit.
    """
    mine, _ = _parse(MY_TERMS_FILE, warn_if_missing=False)
    project, _ = _parse(VOCABULARY_FILE)

    seen: set[str] = set()
    terms = [t for t in mine + project if not (t in seen or seen.add(t))]
    if not terms:
        return None

    kept: list[str] = []
    used = 0
    for term in terms:
        cost = len(term) + 2  # term plus the separator
        if used + cost > HOTWORDS_CHAR_BUDGET:
            break
        kept.append(term)
        used += cost

    if len(kept) < len(terms):
        logger.warning(
            "You have %d speech-recognition terms but only the first %d fit in "
            "Whisper's hotwords limit. Move the less critical ones from "
            "vocabulary.txt into glossary.txt (that file has no limit).",
            len(terms), len(kept),
        )
    else:
        logger.info("Hotwords: %d terms", len(kept))

    return "، ".join(kept)
