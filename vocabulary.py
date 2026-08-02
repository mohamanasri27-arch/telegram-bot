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

# Whisper truncates hotwords at 448 // 2 - 1 tokens. Persian averages roughly
# 4 tokens per word, so this character budget keeps us clear of the cliff.
HOTWORDS_CHAR_BUDGET = 450


def _parse(path: Path) -> tuple[list[str], dict[str, str]]:
    if not path.exists():
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
    """Every pinned translation, from both files.

    Personal terms are applied last so they win over the general dictionary,
    and longer phrases are matched before shorter ones nested inside them.
    """
    _, general = _parse(GLOSSARY_FILE)
    _, personal = _parse(VOCABULARY_FILE)

    combined = {**general, **personal}
    ordered = dict(sorted(combined.items(), key=lambda kv: len(kv[0]), reverse=True))
    logger.info(
        "Glossary: %d general + %d personal terms (%d total)",
        len(general), len(personal), len(ordered),
    )
    return ordered


def load_hotwords() -> str | None:
    """Personal terms only, trimmed to what Whisper will actually read."""
    terms, _ = _parse(VOCABULARY_FILE)
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
            "vocabulary.txt has %d terms but only the first %d fit in Whisper's "
            "hotwords limit. Move the less critical ones to glossary.txt.",
            len(terms), len(kept),
        )
    else:
        logger.info("Hotwords: %d personal terms", len(kept))

    return "، ".join(kept)
