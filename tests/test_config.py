from pathlib import Path

from hermes.config import Config


def test_model_and_voice_persist(tmp_path: Path):
    cfg = Config(tmp_path)
    assert cfg.model  # default no vacío
    cfg.model = "qwen-qwq-32b"
    cfg.voice_enabled = False
    cfg2 = Config(tmp_path)
    assert cfg2.model == "qwen-qwq-32b"
    assert cfg2.voice_enabled is False


def test_api_key_from_home_env_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    env = tmp_path /".env"
    env.write_text('HERMES_API_KEY="clave-de-prueba"\n', encoding="utf-8")
    cfg = Config(tmp_path)
    assert cfg.api_key == "clave-de-prueba"


def test_groq_key_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    env = tmp_path /".env"
    env.write_text('GROQ_API_KEY="clave-groq"\n', encoding="utf-8")
    assert Config(tmp_path).api_key == "clave-groq"


def test_api_key_env_var_wins(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "desde-env")
    cfg = Config(tmp_path)
    assert cfg.api_key == "desde-env"


def test_nvidia_defaults():
    cfg = Config()
    assert cfg.base_url == "https://integrate.api.nvidia.com/v1"
    assert cfg.model.startswith("nvidia/")
    assert cfg.model.endswith("a12b")


def test_base_url_persists(tmp_path: Path):
    cfg = Config(tmp_path)
    cfg.base_url = "https://api.groq.com/openai/v1"
    assert Config(tmp_path).base_url == "https://api.groq.com/openai/v1"
