import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from hermes.voice import (SAMPLE_RATE, BLOCK_SECONDS, HandsFree, Transcriber,
                          _HandsFreeStopped, _clean_for_speech, _read_wav_float32,
                          _silence_alsa_errors, contains_wake_phrase)


def test_silence_alsa_errors_is_noop():
    # No debe lanzar aunque no exista libasound (p. ej. CI sin ALSA).
    _silence_alsa_errors()


def test_clean_for_speech_strips_markup():
    assert _clean_for_speech('Hola **mundo**') == 'Hola mundo'
    assert _clean_for_speech('Dijo: "hola"') == 'Dijo: hola'
    assert _clean_for_speech('“comillas curvas”') == 'comillas curvas'


def _write_sine_wav(path: Path, seconds: float = 0.3) -> Path:
    n = int(SAMPLE_RATE * seconds)
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)))
        for i in range(n)
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(frames)
    return path


def test_contains_wake_phrase_detects_and_strips():
    ok, rest = contains_wake_phrase("hermes abre el navegador")
    assert ok is True
    assert rest == "abre el navegador"


def test_contains_wake_phrase_variants():
    ok, rest = contains_wake_phrase("Hermes dime la hora")
    assert ok is True
    assert rest == "dime la hora"
    ok, _ = contains_wake_phrase("Ermes, ¿qué hora es?")
    assert ok is True


def test_contains_wake_phrase_alone():
    ok, rest = contains_wake_phrase("hermes")
    assert ok is True
    assert rest == ""


def test_contains_wake_phrase_no_match():
    ok, rest = contains_wake_phrase("cuéntame un chiste")
    assert ok is False
    assert rest == "cuéntame un chiste"


def test_read_wav_float32(tmp_path):
    wav = _write_sine_wav(tmp_path / "sine.wav")
    audio = _read_wav_float32(wav)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(audio.max()) <= 1.0
    assert len(audio) == int(SAMPLE_RATE * 0.3)


# ------------------------------------------------------------ hands-free
class FakeTranscriber:
    """Devuelve transcripciones guionizadas (una por llamada)."""

    def __init__(self, responses):
        self.responses = list(responses)

    def transcribe_audio(self, audio):
        return self.responses.pop(0) if self.responses else ""


class FakeHF(HandsFree):
    """HandsFree sin micrófono: los bloques salen de una lista."""

    def __init__(self, blocks, responses):
        super().__init__(FakeTranscriber(responses), block_seconds=BLOCK_SECONDS,
                         silence_seconds=1.2)
        self._blocks = list(blocks)

    def _get_block(self):
        if not self._blocks:
            raise _HandsFreeStopped
        return self._blocks.pop(0)


def _loud_block() -> np.ndarray:
    return np.full((int(SAMPLE_RATE * BLOCK_SECONDS), 1), 0.3, dtype=np.float32)


def _silent_block() -> np.ndarray:
    return np.zeros((int(SAMPLE_RATE * BLOCK_SECONDS), 1), dtype=np.float32)


def test_hands_free_wake_and_command():
    # calibración en silencio + wake word + pausa + comando + silencio
    hf = FakeHF([_silent_block()] * 4 + [_loud_block()] * 5 + [_silent_block()] * 2
                + [_loud_block()] * 5 + [_silent_block()] * 8,
                responses=["hola hermes", "hola hermes abre el navegador"])
    woke = []
    command = hf._listen(on_wake=lambda: woke.append(True))
    assert woke, "la wake word no activó el callback"
    assert command == "abre el navegador"


def test_hands_free_short_wake_word():
    # «Hermes» solo dura ~0.4-0.6s: no debe descartarse por ser corto
    hf = FakeHF([_silent_block()] * 4 + [_loud_block()] * 3 + [_silent_block()] * 2
                + [_loud_block()] * 5 + [_silent_block()] * 8,
                responses=["hermes", "abre el navegador"])
    woke = []
    command = hf._listen(on_wake=lambda: woke.append(True))
    assert woke
    assert command == "abre el navegador"


def test_hands_free_command_in_same_breath():
    # petición dicha de corrido con la wake word, sin pausa
    hf = FakeHF([_silent_block()] * 4 + [_loud_block()] * 10 + [_silent_block()] * 8,
                responses=["hola hermes abre el navegador"])
    woke = []
    command = hf._listen(on_wake=lambda: woke.append(True))
    assert woke
    assert command == "abre el navegador"


def test_hands_free_ignores_non_wake_speech():
    hf = FakeHF([_silent_block()] * 4 + [_loud_block()] * 5 + [_silent_block()] * 3,
                responses=["hola mundo"])
    woke = []
    with pytest.raises(_HandsFreeStopped):
        hf._listen(on_wake=lambda: woke.append(True))
    assert not woke


def test_hands_free_wake_alone_keeps_listening():
    # wake word sola: no hay comando, se sigue escuchando
    hf = FakeHF([_silent_block()] * 4 + [_loud_block()] * 5 + [_silent_block()] * 8,
                responses=["hola hermes", "hola hermes"])
    with pytest.raises(_HandsFreeStopped):
        hf._listen(on_wake=lambda: None)


def test_hands_free_detects_saturated_mic():
    from hermes.voice import VoiceError

    sat = np.full((int(SAMPLE_RATE * BLOCK_SECONDS), 1), 0.9, dtype=np.float32)
    hf = FakeHF([sat] * 4, responses=[])
    with pytest.raises(VoiceError, match="saturado"):
        hf._listen(on_wake=None)


def test_listen_for_command_retries_with_fresh_queue(monkeypatch):
    """Tras saturación, el reintento recalibra con cola limpia (no con los
    bloques viejos del intento fallido)."""
    import queue as queue_mod

    import hermes.voice as voice_mod

    sat = np.full((int(SAMPLE_RATE * BLOCK_SECONDS), 1), 0.9, dtype=np.float32)
    first_attempt = [sat] * 6  # 4 de calibración + 2 sobrantes en la cola
    second_attempt = ([_silent_block()] * 4 + [_loud_block()] * 5
                      + [_silent_block()] * 2 + [_loud_block()] * 5
                      + [_silent_block()] * 8)
    responses = ["hola hermes", "abre el navegador"]

    class FakeStream:
        def __init__(self, hf):
            self.hf = hf
            self.calls = 0

        def __enter__(self):
            blocks = [first_attempt, second_attempt][self.calls]
            self.calls += 1
            for b in blocks:
                self.hf._queue.put(b)
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(voice_mod, "_restart_audio_stack", lambda: True)

    hf = HandsFree(FakeTranscriber(responses))
    hf._queue = queue_mod.Queue()
    fake = FakeStream(hf)
    monkeypatch.setattr("sounddevice.InputStream", lambda *a, **k: fake)

    command = hf.listen_for_command()
    assert command == "abre el navegador"
    assert fake.calls == 2, "debió reintentar tras la saturación"


def test_transcribe_audio_flattens_2d():
    tr = Transcriber("base")
    received = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            received["audio"] = audio
            return iter(()), None

    tr._model = FakeModel()  # sin descargar el modelo real
    assert tr.transcribe_audio(np.zeros((10, 1), dtype=np.float32)) == ""
    assert received["audio"].ndim == 1
