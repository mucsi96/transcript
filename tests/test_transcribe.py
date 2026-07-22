import sys
from types import SimpleNamespace

from transcript.transcribe import format_progress, mlx_repo, pick_backend, pick_device


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


def test_format_progress_midway():
    # 900 of 9000 s of audio done in 600 s wall time -> 10%, 1.5x, 90 min left
    line = format_progress(done=900, total=9000, elapsed=600)
    assert line == "10.0% (900/9000 s of audio, 1.50x realtime, ~90 min left)"


def test_format_progress_no_elapsed_yet():
    line = format_progress(done=0, total=9000, elapsed=0)
    assert "ETA unknown" in line
    assert line.startswith("0.0%")


def test_format_progress_zero_total():
    assert "ETA unknown" in format_progress(done=0, total=0, elapsed=5)


def test_pick_backend_explicit_choices():
    assert pick_backend("faster-whisper") == "faster-whisper"
    assert pick_backend("mlx") == "mlx"


def test_pick_backend_auto_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace())
    assert pick_backend("auto") == "mlx"


def test_pick_backend_auto_apple_silicon_without_mlx(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    assert pick_backend("auto") == "faster-whisper"


def test_pick_backend_auto_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert pick_backend("auto") == "faster-whisper"


def test_mlx_repo_mapping():
    assert mlx_repo("large-v3") == "mlx-community/whisper-large-v3-mlx"
    assert mlx_repo("mlx-community/whisper-large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
