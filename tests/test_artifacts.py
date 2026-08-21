import json

import pytest

from transcript.artifacts import SCHEMA_VERSION, load_json, require_list, save_json


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_round_trip(tmp_path):
    path = tmp_path / "artifact.json"
    save_json(path, {"schema_version": SCHEMA_VERSION, "segments": []})
    assert load_json(path)["segments"] == []


def test_truncated_artifact_names_the_file_and_the_fix(tmp_path):
    # What an interrupted run leaves behind.
    path = write(tmp_path, "analysis.json", '{"schema_version": 1, "sentences": [{"te')
    with pytest.raises(ValueError) as excinfo:
        load_json(path)
    message = str(excinfo.value)
    assert "analysis.json" in message
    assert "not valid JSON" in message
    assert "--force" in message


def test_empty_artifact_says_so(tmp_path):
    path = write(tmp_path, "analysis.json", "")
    with pytest.raises(ValueError, match="it is empty"):
        load_json(path)


def test_html_artifact_is_not_reported_as_a_json_column_number(tmp_path):
    path = write(tmp_path, "analysis.json", "<!DOCTYPE html>")
    with pytest.raises(ValueError) as excinfo:
        load_json(path)
    assert "analysis.json is not valid JSON" in str(excinfo.value)


def test_json_that_is_not_an_object_is_rejected(tmp_path):
    path = write(tmp_path, "analysis.json", json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="expected a JSON object, found list"):
        load_json(path)


def test_schema_mismatch_still_reported(tmp_path):
    path = write(tmp_path, "analysis.json", json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version 99"):
        load_json(path)


def test_missing_file_names_the_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="artifact not found"):
        load_json(tmp_path / "absent.json")


def test_require_list_reports_a_missing_field(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        require_list({"schema_version": 1}, "segments", tmp_path / "transcript.json")
    message = str(excinfo.value)
    assert "'segments' list" in message
    assert "missing" in message


def test_require_list_reports_a_wrong_type(tmp_path):
    with pytest.raises(ValueError, match="found a str"):
        require_list({"segments": "nope"}, "segments", tmp_path / "transcript.json")


def test_require_list_returns_the_value():
    assert require_list({"segments": [1]}, "segments", "t.json") == [1]
