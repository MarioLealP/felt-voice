"""Which language a line is in - Felt speaks English or European Portuguese
(2026-10-21). Pure, no dependencies: used to tell Sopro the language."""

import re

_WORD = re.compile(r"[^\W\d_]+")
# Common short words of each language; a line's language is whichever it uses
# more of. Accented letters only Portuguese has count for Portuguese too.
_PT = frozenset(
    "não é que de do da dos das um uma uns umas o os as no na nos nas em com para por mas já também "
    "isso isto aquilo eu tu ele ela nós vocês eles elas meu minha teu tua está estás estou são és "
    "foi vai vais vou tem tens tenho fazer faz fazes aqui ali agora então muito pouco mais menos "
    "bem sim pois bora olha ainda depois antes porque quando onde como quem qual se lá cá".split()
)
_EN = frozenset(
    "the and is are was were you your i me my it its to of in on at for with that this what "
    "not no yes do does did have has had be been will would can could just so but or if "
    "then there here now how why when where who which about from okay".split()
)
_PT_LETTERS = re.compile(r"[ãõçâêôáéíóúà]", re.IGNORECASE)


def guess_lang(text: str) -> str | None:
    """'pt', 'en', or None when there's nothing to tell them apart."""
    words = [w.lower() for w in _WORD.findall(text)]
    pt = sum(w in _PT for w in words) + len(_PT_LETTERS.findall(text))
    en = sum(w in _EN for w in words)
    if pt == en:
        return None
    return "pt" if pt > en else "en"
