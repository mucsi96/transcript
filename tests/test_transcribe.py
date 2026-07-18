import sys
from types import SimpleNamespace

from transcript.transcribe import pick_device


def test_explicit_cpu():
    assert pick_device("cpu") == ("cpu", "int8")


def test_explicit_cuda():
    assert pick_device("cuda") == ("cuda", "float16")


def test_auto_with_gpu(monkeypatch):
    fake = SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert pick_device("auto") == ("cuda", "float16")


def test_auto_without_gpu(monkeypatch):
    fake = SimpleNamespace(get_cuda_device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert pick_device("auto") == ("cpu", "int8")


def test_auto_when_ctranslate2_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    assert pick_device("auto") == ("cpu", "int8")
