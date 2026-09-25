import pytest

from felt_voice.config import VoiceConfig
from felt_voice.tags import strip_tags
from felt_voice.text_normalize import normalize_for_speech


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[laugh] No way, you actually pulled that off!", "No way, you actually pulled that off!"),
        ("Oh, Mario. [chuckle] That's adorable.", "Oh, Mario. That's adorable."),
        ("[sigh] Fine, I'll remind you again.", "Fine, I'll remind you again."),
        ("Wait [gasp], is that a drop?", "Wait, is that a drop?"),
        ("[clear throat] Ahem. [shush] Quiet.", "Ahem. Quiet."),
        ("Something [whatever] else.", "Something else."),
        ("[laugh]", ""),
        ("  [groan]  [sigh] ", ""),
        ("No tags here.", "No tags here."),
    ],
)
def test_default_strips_every_tag(text, expected):
    assert strip_tags(text) == expected


def test_allow_keeps_curatable_tags_inside_a_sentence():
    assert strip_tags("[sigh] Fine, I'll do it.", allow=True) == "[sigh] Fine, I'll do it."
    assert strip_tags("Not [Groan] this song again.", allow=True) == "Not [groan] this song again."
    assert strip_tags("Wait. [clear  throat] Ahem.", allow=True) == "Wait. [clear throat] Ahem."


def test_allow_still_bans_laugh_and_chuckle():
    assert strip_tags("[laugh] No way!", allow=True) == "No way!"
    assert strip_tags("Oh. [chuckle] Cute. [gasp] Wow.", allow=True) == "Oh. Cute. [gasp] Wow."


def test_allow_strips_unknown_tags():
    assert strip_tags("Hi [whisper] there.", allow=True) == "Hi there."


def test_allow_strips_standalone_tag_only_requests():
    assert strip_tags("[sigh]", allow=True) == ""
    assert strip_tags("[gasp] [groan]", allow=True) == ""


def test_allow_tags_config_defaults_off():
    assert VoiceConfig().allow_tags is False


def test_normalize_for_speech_leaves_tags_alone():
    assert normalize_for_speech("[sigh] It's 8.3 km/h out.") == "[sigh] It's 8.3 kilometers per hour out."
