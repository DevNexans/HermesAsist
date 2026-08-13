"""Voz de Hermes: escuchar (micrófono + Whisper local) y hablar (espeak-ng).

El reconocimiento usa faster-whisper (modelo descargado en el primer uso).
La síntesis usa espeak-ng, ya instalado en el sistema, en un hilo en segundo
plano para no bloquear la conversación.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 30      # tope de grabación por turno
SILENCE_SECONDS = 1.6        # silencio que corta la grabación
BLOCK_SECONDS = 0.2
SPEAK_MAX_CHARS = 2000

# Frases de activación para el modo manos libres (normalizadas a minúsculas).
WAKE_PHRASES = (
    "hola hermes", "hey hermes", "hola ermes", "hey ermes", "ola hermes",
    "ahora hermes", "oye hermes", "oiga hermes",
)


class VoiceError(Exception):
    pass


# ------------------------------------------------------------------- TTS
class Speaker:
    def __init__(self, lang: str = "es"):
        self.lang = lang
        self._lock = threading.Lock()

    def say(self, text: str) -> None:
        """Habla el texto con espeak-ng en un hilo en segundo plano."""
        if not text.strip():
            return
        text = text[:SPEAK_MAX_CHARS]
        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()

    def _speak(self, text: str) -> None:
        try:
            with self._lock:
                subprocess.run(
                    ["espeak-ng", "-v", self.lang, "-s", "160", "-a", "150", text],
                    timeout=120, check=False,
                )
        except (OSError, subprocess.SubprocessError):
            pass  # la voz es un extra; no debe romper la sesión


# ------------------------------------------------------------------- STT
class Transcriber:
    """Transcripción local con faster-whisper (carga perezosa del modelo)."""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # import tardío
            except ImportError as exc:
                raise VoiceError(
                    "Falta faster-whisper. Instálalo con: pip install faster-whisper"
                ) from exc
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def transcribe(self, wav_path: str | Path) -> str:
        """Transcribe un archivo WAV."""
        return self.transcribe_audio(_read_wav_float32(wav_path))

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio float32 mono a 16 kHz (faster-whisper lo exige así)."""
        self._ensure_model()
        assert self._model is not None
        audio = np.asarray(audio).reshape(-1)  # acepta (N,) y (N,1)
        if len(audio) == 0:
            return ""
        segments, _info = self._model.transcribe(audio, language="es", beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()


def _read_wav_float32(path: str | Path) -> np.ndarray:
    """Lee un WAV mono y devuelve float32 en [-1, 1] a 16 kHz."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    if sr != SAMPLE_RATE:
        raise VoiceError(f"WAV a {sr} Hz; se esperaba {SAMPLE_RATE} Hz (usa 16 kHz mono).")
    return audio


def contains_wake_phrase(text: str) -> tuple[bool, str]:
    """Detecta la wake word en la transcripción.

    Devuelve (True, texto_sin_la_wake_word) si se detectó; si no, (False, texto).
    """
    t = text.lower().strip()
    for phrase in WAKE_PHRASES:
        if phrase in t:
            rest = t.split(phrase, 1)[1]
            rest = rest.strip(" .,;:!?¿¡-—")
            return True, rest
    return False, text


class _HandsFreeStopped(Exception):
    pass


class HandsFree:
    """Escucha continua con wake word («Hola Hermes»).

    Usa un único InputStream: mientras hay silencio no transcribe nada; cuando
    detecta voz, transcribe un segmento corto y comprueba si es la wake word.
    Si lo es, sigue grabando hasta el silencio, transcribe el comando y lo
    devuelve sin la wake word. Si no lo es, descarta y sigue escuchando.
    """

    WAKE_ENERGY = 0.02          # nivel pico que se considera «hay voz»
    WAKE_MIN_SECONDS = 0.8      # segmentos más cortos no pueden ser la wake word
    WAKE_MAX_SECONDS = 3.5      # tope del segmento analizado en busca de la wake

    def __init__(self, transcriber: Transcriber, block_seconds: float = BLOCK_SECONDS,
                 silence_seconds: float = SILENCE_SECONDS,
                 max_command_seconds: int = MAX_RECORD_SECONDS):
        self.tr = transcriber
        self.block_seconds = block_seconds
        self.silence_seconds = silence_seconds
        self.max_command_seconds = max_command_seconds
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    # ------------------------------------------------------------- stream
    def _callback(self, indata, _frames, _time, _status):  # pragma: no cover
        self._queue.put(indata.copy())

    def _get_block(self) -> np.ndarray:
        while True:
            try:
                return self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    raise _HandsFreeStopped

    def _level(self, block: np.ndarray) -> float:
        return float(np.max(np.abs(block)))

    # ----------------------------------------------------------- wake word
    def listen_for_command(self, on_wake=None) -> str | None:
        """Escucha hasta oír la wake word y devuelve el comando hablado.

        `on_wake` se invoca al detectar la wake word (antes de capturar el
        comando). Devuelve None si se detiene (Ctrl+C o error de audio).
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise VoiceError("Falta sounddevice. Instálalo con: pip install sounddevice") from exc

        self._queue = queue.Queue()
        self._stop.clear()
        blocksize = int(SAMPLE_RATE * self.block_seconds)
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                callback=self._callback, blocksize=blocksize):
                return self._listen(on_wake)
        except _HandsFreeStopped:
            return None
        except Exception as exc:  # noqa: BLE001 - sin dispositivo de audio
            raise VoiceError(f"No se pudo acceder al micrófono: {exc}") from exc

    def _listen(self, on_wake) -> str | None:
        quiet_to_stop = max(1, int(self.silence_seconds / self.block_seconds))
        max_cmd_blocks = int(self.max_command_seconds / self.block_seconds)

        while True:
            # --- fase 1: buscar la wake word -----------------------------
            speech: list[np.ndarray] = []
            quiet = 0
            while True:
                block = self._get_block()
                if self._level(block) > self.WAKE_ENERGY:
                    speech.append(block)
                    quiet = 0
                elif speech:
                    quiet += 1
                    if quiet >= 2:
                        break
                dur = sum(len(b) for b in speech) / SAMPLE_RATE
                if dur >= self.WAKE_MAX_SECONDS:
                    break

            if not speech:
                continue
            dur = sum(len(b) for b in speech) / SAMPLE_RATE
            if dur < self.WAKE_MIN_SECONDS:
                time.sleep(0.3)
                continue

            audio = np.concatenate(speech)
            text = self.tr.transcribe_audio(audio)
            ok, _rest = contains_wake_phrase(text)
            if not ok:
                time.sleep(0.8)  # cooldown: no re-analizar el mismo ruido
                continue
            if on_wake:
                on_wake()

            # --- fase 2: capturar el comando hasta el silencio ----------
            cmd_blocks: list[np.ndarray] = []
            quiet2 = 0
            while len(cmd_blocks) < max_cmd_blocks:
                block = self._get_block()
                cmd_blocks.append(block)
                if self._level(block) < 0.01:
                    quiet2 += 1
                else:
                    quiet2 = 0
                if quiet2 >= quiet_to_stop:
                    break

            if not cmd_blocks:
                continue
            cmd_audio = np.concatenate(cmd_blocks)
            cmd_text = self.tr.transcribe_audio(cmd_audio)
            _ok, command = contains_wake_phrase(cmd_text)
            command = re.sub(r"^\s*(dime|quiero que|puedes)?\s*", "", command)
            if command.strip():
                return command
            # la wake word sola: seguir escuchando por si viene el comando
            time.sleep(0.5)


# ---------------------------------------------------------------- record
def record_until_silence(
    max_seconds: int = MAX_RECORD_SECONDS,
    silence_seconds: float = SILENCE_SECONDS,
) -> str:
    """Graba el micrófono hasta detectar silencio y devuelve la ruta de un WAV."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise VoiceError(
            "Falta sounddevice. Instálalo con: pip install sounddevice"
        ) from exc

    blocks: list[np.ndarray] = []
    quiet_blocks = 0
    quiet_needed = max(1, int(silence_seconds / BLOCK_SECONDS))
    max_blocks = int(max_seconds / BLOCK_SECONDS)

    def callback(indata, _frames, _time, status):  # pragma: no cover - audio IO
        blocks.append(indata.copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            callback=callback, blocksize=int(SAMPLE_RATE * BLOCK_SECONDS)):
            while len(blocks) < max_blocks:
                sd.sleep(int(BLOCK_SECONDS * 1000))
                if len(blocks) < 4:
                    continue
                level = float(np.max(np.abs(np.concatenate(blocks[-3:]))))
                if level < 0.01:
                    quiet_blocks += 1
                    if quiet_blocks >= quiet_needed:
                        break
                else:
                    quiet_blocks = 0
    except Exception as exc:  # noqa: BLE001 - sin dispositivo de audio
        raise VoiceError(f"No se pudo acceder al micrófono: {exc}") from exc

    if not blocks:
        raise VoiceError("No se capturó audio del micrófono.")

    audio = np.concatenate(blocks)
    out = Path("/tmp") / f"hermes_{os.getpid()}_{time.time_ns()}.wav"
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return str(out)
