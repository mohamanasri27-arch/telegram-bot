"""Shared loader for the user's domain vocabulary.

vocabulary.txt serves two purposes, so both the transcriber and the translator
read it through here:

  - every entry biases the speech model toward the user's jargon
  - entries written as "فارسی = English" also pin the translation of that term
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VOCABULARY_FILE = Path(__file__).with_name("vocabulary.txt")


def load_entries() -> tuple[list[str], dict[str, str]]:
    """Return (all Persian terms, {Persian term: English term}).

    Longer terms come first in the mapping so that multi-word entries are
    matched before any shorter entry nested inside them.
    """
    if not VOCABULARY_FILE.exists():
        logger.warning("vocabulary.txt not found; continuing without domain terms")
        return [], {}

    terms: list[str] = []
    glossary: dict[str, str] = {}

    for raw_line in VOCABULARY_FILE.read_text(encoding="utf-8").splitlines():
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

    ordered = dict(sorted(glossary.items(), key=lambda kv: len(kv[0]), reverse=True))
    return terms, ordered
