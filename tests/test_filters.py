from transcript.config import FilterConfig
from transcript.filters import is_learnable

CFG = FilterConfig()


def keep(tok, cfg=CFG):
    return is_learnable(tok, cfg)[0]


def reason(tok, cfg=CFG):
    return is_learnable(tok, cfg)[1]


def test_regular_noun_is_kept(make_token):
    assert keep(make_token(text="Hund", lemma="Hund", pos="NOUN"))


def test_proper_noun_is_dropped(make_token):
    tok = make_token(text="Müller", lemma="Müller", pos="PROPN")
    assert not keep(tok)
    assert reason(tok) == "pos:PROPN"


def test_punctuation_symbols_space_x_are_dropped(make_token):
    for pos in ("PUNCT", "SYM", "SPACE", "X"):
        assert not keep(make_token(text=".", pos=pos, is_alpha=False))


def test_numerals_dropped_by_default_but_keepable(make_token):
    tok = make_token(text="zwei", lemma="zwei", pos="NUM")
    assert not keep(tok)
    cfg = FilterConfig(exclude_pos=frozenset({"PROPN"}))
    assert keep(tok, cfg)


def test_non_alpha_is_dropped(make_token):
    tok = make_token(text="2:30", pos="NOUN", is_alpha=False)
    assert not keep(tok)
    assert reason(tok) == "non-alpha"


def test_single_char_is_dropped(make_token):
    assert not keep(make_token(text="s", lemma="s", pos="NOUN"))


def test_person_location_org_entities_are_dropped(make_token):
    for ent in ("PER", "LOC", "ORG"):
        tok = make_token(text="Berlin", lemma="Berlin", pos="NOUN", ent_type=ent)
        assert not keep(tok)
        assert reason(tok) == f"entity:{ent}"


def test_misc_entity_is_kept_by_default(make_token):
    # nationality adjectives like "deutsch" are tagged MISC and are learnable
    assert keep(make_token(text="deutsch", lemma="deutsch", pos="ADJ", ent_type="MISC"))


def test_misc_can_be_excluded(make_token):
    cfg = FilterConfig(exclude_ent_types=frozenset({"PER", "LOC", "ORG", "MISC"}))
    assert not keep(make_token(pos="ADJ", ent_type="MISC"), cfg)


def test_stopwords_kept_by_default_dropped_when_enabled(make_token):
    tok = make_token(text="aber", lemma="aber", pos="ADV", is_stop=True)
    assert keep(tok)
    assert not keep(tok, FilterConfig(drop_stopwords=True))
