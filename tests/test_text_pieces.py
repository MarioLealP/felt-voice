from felt_voice.chatterbox_engine import text_pieces

LONG = (
    "Okay so here's the thing about fried rice, and honestly I've thought about this way more than any "
    "sane person should, the rice has to be cold, the pan has to be screaming hot, the eggs go in first "
    "and come out early - then the chicken; then the rice."
)


def test_whole_text_when_off_or_short():
    assert text_pieces(LONG, 0) == [LONG]
    assert text_pieces("Fried rice!", 120) == ["Fried rice!"]


def test_pieces_fit_and_cut_at_clause_breaks():
    pieces = text_pieces(LONG, 120)
    assert all(len(p) <= 120 for p in pieces)
    assert " ".join(pieces) == LONG
    assert all(p.endswith((",", ";", ":", "-", ".")) for p in pieces[:-1])


def test_no_break_falls_back_to_a_space_then_a_hard_cut():
    assert text_pieces("aaa bbb ccc", 8) == ["aaa bbb", "ccc"]
    assert text_pieces("abcdefghij", 4) == ["abcd", "efgh", "ij"]
