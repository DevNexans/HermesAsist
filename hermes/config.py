"""Configuración persistente de Hermes.

Los datos viven en HERMES_HOME (por defecto ~/.hermes):
  - config.json          preferencias (modelo, voz)
  - .env                 API key (HERMES_API_KEY)
  - memory/              hechos, resúmenes y conversaciones

El proveedor LLM es cualquier endpoint compatible con OpenAI (NVIDIA por
defecto; también Groq, OpenAI, Ollama...). Se configura con HERMES_BASE_URL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Modelos que no sirven para chat (STT/TTS/embeddings/guard/visión/imagen/audio...).
NON_CHAT_HINTS = (
    "whisper", "tts", "embedding", "embed", "guard", "stt", "rerank",
    "classifier", "transcribe", "sdxl", "stable-diffusion", "diffusion",
    "flux", "wav2vec", "clap", "dalle", "image", "audio", "tts-1",
    "vision", "video", "detector", "calibration", "deplot", "reward",
    "parse", "kosmos", "cosmos", "inpaint", "music", "speech",
)

# Nombres de variable para la API key (el primero gana).
KEY_ENV_NAMES = ("HERMES_API_KEY", "GROQ_API_KEY")


class Config:
    def __init__(self, home: str | Path | None = None):
        self.home = Path(home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        self.config_path = self.home / "config.json"
        self.memory_dir = self.home / "memory"
        self.sessions_dir = self.home / "memory" / "conversations"
        self._data = self._load()

    # ---------------------------------------------------------------- setup
    def ensure_dirs(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save(self) -> None:
        self.ensure_dirs()
        self.config_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------- settings
    @property
    def base_url(self) -> str:
        return self._data.get("base_url") or os.environ.get("HERMES_BASE_URL") or DEFAULT_BASE_URL

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._data["base_url"] = value
        self.save()

    @property
    def model(self) -> str:
        return self._data.get("model") or os.environ.get("HERMES_MODEL") or DEFAULT_MODEL

    @model.setter
    def model(self, value: str) -> None:
        self._data["model"] = value
        self.save()

    @property
    def voice_enabled(self) -> bool:
        return bool(self._data.get("voice", True))

    @voice_enabled.setter
    def voice_enabled(self, value: bool) -> None:
        self._data["voice"] = bool(value)
        self.save()

    @property
    def whisper_model(self) -> str:
        return os.environ.get("HERMES_WHISPER_MODEL") or "medium"

    @property
    def mic_device(self) -> str | None:
        """Dispositivo de micrófono (nombre o índice) para grabación."""
        return os.environ.get("HERMES_MIC_DEVICE") or None

    @property
    def voice_model(self) -> str | None:
        """Modelo de voz Piper (.onnx). Si no existe, se usa espeak-ng."""
        return os.environ.get("HERMES_VOICE") or str(
            Path.home() / ".hermes" / "voice" / "es_ES-sharvard-medium.onnx"
        )

    # ------------------------------------------------------------ api key
    def load_env(self) -> None:
        """Carga .env del cwd y de HERMES_HOME (las variables ya definidas ganan)."""
        load_dotenv(Path.cwd() / ".env", override=False)
        load_dotenv(self.home / ".env", override=False)

    @property
    def api_key(self) -> str | None:
        for name in KEY_ENV_NAMES:
            key = os.environ.get(name)
            if key:
                return key
        env_file = self.home / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                for name in KEY_ENV_NAMES:
                    if line.startswith(f"{name}="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        return None
