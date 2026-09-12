"""Deterministic text cleanup before Piper synthesis.

Mario, 2026-09-12: "the windspeed comes as 8.3 km/h and it gets read
verbatim... maybe we can have a layer between text output -> voice to sort
those small details" — brain.py already has one instance of this class of
problem (a prompt instruction telling the model to leave a bare temperature
number as-is, no unit letter appended), but that only covers text the LLM
itself generates, and relies on it actually following the instruction every
time. This is a small, deterministic regex pass instead: it also covers
skill output that's never touched by the LLM at all (e.g. printer_status's
hardcoded "72.0°C" strings), and can't be skipped the way a prompt
instruction occasionally is.

Lives in felt-voice rather than felt-core because every speech path — the
dashboard's voice-chat replies and felt-desktop-agent's notification
read-aloud (player.py) — already converges on this service's one /v1/tts
endpoint. One place, not duplicated per caller.
"""

import re

# (pattern, replacement) — each pattern anchors on the digit immediately
# before the unit so the replacement can keep that digit and only expand
# the unit itself, leaving the rest of the number untouched either side.
_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(\d)\s?km/h\b", re.IGNORECASE), r"\1 kilometers per hour"),
    (re.compile(r"(\d)\s?mph\b", re.IGNORECASE), r"\1 miles per hour"),
    (re.compile(r"(\d)\s?m/s\b", re.IGNORECASE), r"\1 meters per second"),
    (re.compile(r"(\d)\s?kg\b", re.IGNORECASE), r"\1 kilograms"),
    (re.compile(r"(\d)\s?°C\b"), r"\1 degrees Celsius"),
    (re.compile(r"(\d)\s?°F\b"), r"\1 degrees Fahrenheit"),
    # A bare degree sign with no letter (get_weather's current-conditions
    # format, per brain.py's instruction not to append a unit) — after the
    # unit-specific patterns above so it only catches what's left over.
    (re.compile(r"(\d)\s?°(?!\w)"), r"\1 degrees"),
]


def normalize_for_speech(text: str) -> str:
    for pattern, replacement in _REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text
