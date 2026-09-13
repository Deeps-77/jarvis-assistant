"""Whisper device/compute selection. Model construction is mocked."""

import speech
from speech import SpeechTranscriber


def test_auto_uses_configured_cuda_compute(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "auto")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8_float16")
    seen = []

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None):
            seen.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("no GPU here")

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    SpeechTranscriber()._load_sync()
    assert seen == [("cuda", "int8_float16"), ("cpu", "int8")]


def test_auto_defaults_cuda_int8_float16(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "auto")
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    seen = []

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None):
            seen.append((device, compute_type))

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    SpeechTranscriber()._load_sync()
    assert seen[0] == ("cuda", "int8_float16")


def test_explicit_device_uses_configured_compute(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    seen = []

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None):
            seen.append((device, compute_type))

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    SpeechTranscriber()._load_sync()
    assert seen == [("cpu", "int8")]
