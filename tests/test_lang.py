import pytest

from felt_voice.lang import guess_lang


@pytest.mark.parametrize(
    "text",
    [
        "Pois, nota-se que estás cansado. Fazes-me um favor? Fecha isso e vai comer qualquer coisa.",
        "Uma hora a arrumar o stash, a sério? Bora lá matar um boss antes que eu adormeça.",
        "Ok, isso foi mesmo limpo.",
        "Não.",
    ],
)
def test_portuguese(text):
    assert guess_lang(text) == "pt"


@pytest.mark.parametrize(
    "text",
    [
        "Okay, that was actually really clean. The dodge at the end? I'm impressed.",
        "It's almost two in the morning. Are you really opening another project folder right now?",
        "Is the boss still alive?",
    ],
)
def test_english(text):
    assert guess_lang(text) == "en"


@pytest.mark.parametrize("text", ["", "Hm.", "42", "Mario!"])
def test_nothing_to_go_on(text):
    assert guess_lang(text) is None
