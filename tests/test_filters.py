from transcript.config import DEFAULT_EXCLUDE_POS, FilterConfig
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


def test_stopwords_dropped_by_default_keepable(make_token):
    tok = make_token(text="so", lemma="so", pos="ADV", is_stop=True)
    assert not keep(tok)
    assert reason(tok) == "stopword"
    assert keep(tok, FilterConfig(drop_stopwords=False))


def test_function_words_are_dropped(make_token):
    # the words that dominate any frequency list: der/sie/auf/sein/und/dass/zu
    for text, lemma, pos in (
        ("Der", "der", "DET"),
        ("sie", "sie", "PRON"),
        ("auf", "auf", "ADP"),
        ("ist", "sein", "AUX"),
        ("und", "und", "CCONJ"),
        ("dass", "dass", "SCONJ"),
        ("zu", "zu", "PART"),
    ):
        tok = make_token(text=text, lemma=lemma, pos=pos)
        assert not keep(tok), lemma
        assert reason(tok) == f"pos:{pos}"


def test_function_class_can_be_kept(make_token):
    tok = make_token(text="sich", lemma="sich", pos="PRON")
    cfg = FilterConfig(exclude_pos=DEFAULT_EXCLUDE_POS - {"PRON"})
    assert keep(tok, cfg)


def test_content_words_survive_the_function_word_rules(make_token):
    # ADV/VERB/ADJ are not closed classes, so only the stopword list touches
    # them: "plötzlich" stays while "so" goes.
    for lemma, pos in (("plötzlich", "ADV"), ("erzählen", "VERB"), ("müde", "ADJ")):
        assert keep(make_token(text=lemma, lemma=lemma, pos=pos)), lemma
