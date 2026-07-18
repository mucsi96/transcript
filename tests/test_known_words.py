from transcript.known_words import is_known, load_known_words, normalize_word


def write_list(tmp_path, content):
    path = tmp_path / "known.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_skips_blanks_comments_and_whitespace(tmp_path):
    path = write_list(tmp_path, "# my words\n\n  laufen  \nHaus\r\n\n# more\nschnell\n")
    assert load_known_words(path) == {"laufen", "haus", "schnell"}


def test_matching_is_case_insensitive(tmp_path):
    known = load_known_words(write_list(tmp_path, "haus\n"))
    assert is_known("Haus", known)
    assert is_known("HAUS", known)


def test_umlauts_are_not_conflated(tmp_path):
    known = load_known_words(write_list(tmp_path, "Baume\n"))
    assert not is_known("Bäume", known)


def test_eszett_matches_ss_spelling(tmp_path):
    # casefold maps ß -> ss, so "Strasse" in the list matches "Straße".
    known = load_known_words(write_list(tmp_path, "Strasse\n"))
    assert is_known("Straße", known)


def test_normalize_word():
    assert normalize_word("  Straße \n") == "strasse"
