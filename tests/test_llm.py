"""Tests del cliente LLM (hermes.llm)."""

import pytest

from hermes.llm import GroqError, GroqLLM


class _FakeStream:
    """Itera y lanza APIError a mitad de la respuesta (proveedor 5xx)."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def __iter__(self):
        raise self.exc


class _FakeClient:
    """Replica la cadena client.chat.completions.create() de openai."""

    def __init__(self, stream_exc: Exception | None = None):
        self.chat = type("Chat", (), {
            "completions": type("Completions", (), {
                "create": self._create,
            })(),
        })()
        self.stream_exc = stream_exc

    def _create(self, **kwargs):  # imita la API de openai
        if self.stream_exc is not None:
            return _FakeStream(self.stream_exc)
        raise AssertionError("no debería llegar aquí")


def test_stream_error_raises_groqerror():
    """Un error del proveedor a mitad del stream no debe crashear: GroqError."""
    from openai import APIError
    from openai._exceptions import APIError as _APIError

    req = type("Req", (), {})()
    exc = _APIError.__new__(_APIError)
    exc.message = "Internal server error"
    exc.request = req

    llm = GroqLLM.__new__(GroqLLM)
    llm.client = _FakeClient(stream_exc=exc)

    with pytest.raises(GroqError, match="Error llamando al proveedor"):
        llm.run([{"role": "user", "content": "hola"}], tools=None, model="x")


def test_stream_timeout_raises_groqerror():
    """Timeout a mitad del stream también se traduce a GroqError."""
    from openai import APITimeoutError
    from openai._exceptions import APITimeoutError as _APITimeoutError

    req = type("Req", (), {})()
    exc = _APITimeoutError.__new__(_APITimeoutError)
    exc.message = "lento"
    exc.request = req

    llm = GroqLLM.__new__(GroqLLM)
    llm.client = _FakeClient(stream_exc=exc)

    with pytest.raises(GroqError, match="dejó de responder"):
        llm.run([{"role": "user", "content": "hola"}], tools=None, model="x")
