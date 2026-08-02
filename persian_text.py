"""Clean up raw Whisper output before it is shown or translated.

Whisper emits Arabic-script characters that look identical to their Persian
counterparts but are different code points. Left alone they make the Persian
text look wrong to a native reader, and they make the translator treat familiar
words as unknown ones.
"""
import re

# Arabic code points that must become their Persian equivalents.
CHARACTER_FIXES = {
    "ي": "ی",  # ARABIC YEH        -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA      -> FARSI YEH
    "ك": "ک",  # ARABIC KAF        -> KEHEH
    "ة": "ه",  # TEH MARBUTA       -> HEH
    "ؤ": "و",  # WAW WITH HAMZA    -> WAW
    "إ": "ا",  # ALEF WITH HAMZA   -> ALEF
    "أ": "ا",  # ALEF WITH HAMZA   -> ALEF
}

# Arabic-Indic and extended digits -> ASCII, so numbers survive translation.
for _offset, _arabic_zero in ((0, 0x0660), (0, 0x06F0)):
    for _digit in range(10):
        CHARACTER_FIXES[chr(_arabic_zero + _digit)] = str(_digit)

_TRANSLATION_TABLE = str.maketrans(CHARACTER_FIXES)

# Short vowel marks and the decorative kashida carry no meaning in modern text.
DIACRITICS_RE = re.compile(r"[ً-ْـ]")

ZWNJ = "‌"

# Whisper often stutters a word when the speaker hesitates.
REPEATED_WORD_RE = re.compile(r"\b(\S+)(\s+\1\b)+", re.UNICODE)

MULTISPACE_RE = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.،؛:؟!])")
MISSING_SPACE_AFTER_PUNCT_RE = re.compile(r"([.،؛:؟!])(?=[^\s\d])")
STRAY_ZWNJ_RE = re.compile(rf"{ZWNJ}{{2,}}|(?<=\s){ZWNJ}|{ZWNJ}(?=\s)")


def normalize(text: str) -> str:
    """Return the same sentence with Persian orthography and tidy spacing."""
    if not text:
        return text

    text = text.translate(_TRANSLATION_TABLE)
    text = DIACRITICS_RE.sub("", text)
    text = STRAY_ZWNJ_RE.sub("", text)
    text = REPEATED_WORD_RE.sub(r"\1", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = MISSING_SPACE_AFTER_PUNCT_RE.sub(r"\1 ", text)
    text = MULTISPACE_RE.sub(" ", text)

    return text.strip()
