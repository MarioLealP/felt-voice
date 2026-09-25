"""Chatterbox paralinguistic tags ([sigh], [gasp], ...) on the way to TTS.

Mario, 2026-09-25, after listening to the non-verbal A/B takes: "the laugh
is horrendous, remove all of those. The other non verbals are a mixed bag."
So the live /v1/tts path strips *every* bracketed tag before synthesis,
inline or standalone, and live speech never carries an uncurated non-verbal.

`FELT_VOICE_ALLOW_TAGS=true` exists for WP-19's offline batch, where Mario
curates sighs/gasps/groans by ear. Even then:
- [laugh] and [chuckle] are always stripped (banned outright);
- anything that isn't a known Chatterbox-Turbo tag is stripped (it would be
  read out as words);
- a request that is *only* tags gets them all stripped: RVC turns a lone
  non-verbal into fake words, so a tag must sit inside a real sentence.
"""

import re

_TAG = re.compile(r"\[([^\[\]]*)\]")

# Chatterbox-Turbo's paralinguistic tags, minus the banned ones.
_CURATABLE = frozenset({"sigh", "gasp", "groan", "cough", "sniff", "shush", "clear throat"})
_BANNED = frozenset({"laugh", "chuckle"})


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def strip_tags(text: str, allow: bool = False) -> str:
    """Removes bracketed tags from `text`. With `allow`, curatable tags
    inside a sentence that has real words are kept (see module docstring)."""
    stripped = _tidy(_TAG.sub(" ", text))
    if not allow or not re.search(r"\w", stripped):
        return stripped

    def keep(match: re.Match[str]) -> str:
        name = " ".join(match.group(1).lower().split())
        if name in _CURATABLE and name not in _BANNED:
            return f"[{name}]"
        return " "

    return _tidy(_TAG.sub(keep, text))
