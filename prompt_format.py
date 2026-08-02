"""Reshape a translated paragraph into a prompt an AI model reads well.

Spoken explanations arrive as one long run of sentences that mixes background
with the actual ask. Models follow the ask better when the two are separated,
so sentences are sorted into Context and Task.

This is a deliberately simple rule-based split, not an AI rewrite: it never
invents or drops wording, it only groups the sentences that are already there.
"""
import re

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", re.UNICODE)

# Phrases that mark a sentence as the thing being asked for rather than background.
TASK_MARKERS = (
    "i want", "i need", "i would like", "i'd like",
    "please", "help me", "help with",
    "can you", "could you", "would you",
    "how do", "how can", "how should", "what is", "what are", "why", "which",
    "give me", "show me", "tell me", "explain", "describe",
    "write", "create", "make", "build", "generate", "design",
    "suggest", "recommend", "review", "check", "fix", "improve", "optimize",
    "should i", "do i need",
)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


def _is_task(sentence: str) -> bool:
    lowered = sentence.lower().lstrip("\"'([ ")
    if sentence.rstrip().endswith("?"):
        return True
    return any(lowered.startswith(m) or f" {m}" in lowered for m in TASK_MARKERS)


def build(translated: str) -> str:
    """Return the translation regrouped under Context and Task headings."""
    sentences = _split_sentences(translated)
    if not sentences:
        return translated

    context = [s for s in sentences if not _is_task(s)]
    task = [s for s in sentences if _is_task(s)]

    if not task:
        # Nothing reads as a request, so the whole thing is background. Saying so
        # is more useful than inventing a task the speaker never stated.
        return (
            "## Context\n"
            f"{' '.join(context)}\n\n"
            "## Task\n"
            "(Add what you want the model to do with the context above.)"
        )

    if not context:
        return "## Task\n" + " ".join(task)

    return f"## Context\n{' '.join(context)}\n\n## Task\n{' '.join(task)}"
