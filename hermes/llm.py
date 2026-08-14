"""Cliente LLM para Hermes (cualquier API compatible con OpenAI).

Por defecto usa NVIDIA (integrate.api.nvidia.com); también sirve para Groq,
OpenAI, Ollama, etc. cambiando HERMES_BASE_URL. Incluye streaming,
bucle de tool calling y un registro de modelos en vivo (para /modelos).
"""

from __future__ import annotations

import json
from typing import Callable, cast

from openai import APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionChunk

from hermes.config import DEFAULT_BASE_URL, NON_CHAT_HINTS

MAX_TOOL_ROUNDS = 8

# Si el proveedor no manda un solo byte en este tiempo (segundos), se aborta
# la llamada. Un float aplica el mismo tope a conexión, lectura y escritura.
CLIENT_TIMEOUT = 90.0


class GroqError(Exception):
    pass


class GroqLLM:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        if not api_key or api_key == "tu_clave_aqui":
            raise GroqError(
                "Falta la API key (HERMES_API_KEY). Consíguela en el panel de tu "
                "proveedor y guárdala en ~/.hermes/.env (HERMES_API_KEY=...) o como "
                "variable de entorno."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=CLIENT_TIMEOUT)

    # ------------------------------------------------------------- models
    def list_models(self) -> list[str]:
        """Devuelve los modelos de chat disponibles, ordenados alfabéticamente."""
        models = self.client.models.list()
        chat = []
        for m in models.data:
            mid = m.id.lower()
            if any(hint in mid for hint in NON_CHAT_HINTS):
                continue
            chat.append(m.id)
        return sorted(chat)

    # --------------------------------------------------------------- chat
    def run(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        on_delta: Callable[[str], None] | None = None,
        on_tool: Callable[[str, str], None] | None = None,
        dispatch: Callable[[str, str], str] | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Envia la conversación y devuelve el texto final.

        Si el modelo pide herramientas, se ejecutan (vía `dispatch`) y se
        reintenta, hasta MAX_TOOL_ROUNDS. `on_delta` recibe cada fragmento de
        texto; `on_tool` recibe (nombre, argumentos) antes de ejecutar.
        """
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    stream=True,
                    temperature=temperature,
                )
            except APITimeoutError as exc:
                raise GroqError(
                    "El proveedor no respondió a tiempo (modelo saturado o lento). "
                    "Prueba un modelo más rápido con: /modelo nvidia/llama-3.3-nemotron-super-49b-v1"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - se informa al usuario
                raise GroqError(f"Error llamando al proveedor: {exc}") from exc

            text_parts: list[str] = []
            tool_calls: dict[int, dict] = {}

            try:
                for chunk in stream:
                    chunk = cast(ChatCompletionChunk, chunk)
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        text_parts.append(delta.content)
                        if on_delta:
                            on_delta(delta.content)
                    if delta and delta.tool_calls:
                        for part in delta.tool_calls:
                            entry = tool_calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                            if part.id:
                                entry["id"] += part.id
                            if part.function:
                                if part.function.name:
                                    entry["name"] += part.function.name
                                if part.function.arguments:
                                    entry["arguments"] += part.function.arguments
            except APITimeoutError as exc:
                raise GroqError(
                    "El proveedor dejó de responder a mitad de la respuesta "
                    "(modelo saturado). Prueba: /modelo nvidia/llama-3.3-nemotron-super-49b-v1"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - se informa al usuario
                raise GroqError(f"Error llamando al proveedor: {exc}") from exc

            if not tool_calls:
                return "".join(text_parts)

            # Ronda de herramientas: registrar la llamada y ejecutarla.
            if dispatch is None:
                raise GroqError("El modelo pidió una herramienta pero no hay dispatch configurado.")

            assistant_msg = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls.values()
                ],
            }
            messages.append(assistant_msg)

            for tc in tool_calls.values():
                name, args = tc["name"], tc["arguments"]
                if on_tool:
                    on_tool(name, args)
                result = dispatch(name, args)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        raise GroqError("Demasiadas rondas de herramientas; detengo la conversación.")

    # ------------------------------------------------------------ summary
    def summarize(self, history: list[dict], model: str) -> str:
        """Resume una sesión para archivarla en memoria (sin streaming)."""
        prompt = (
            "Resume en español, en máximo 120 palabras, qué se hizo y qué se "
            "acordó en esta conversación, para recordarlo en el futuro. "
            "Incluye datos concretos (rutas, comandos, decisiones)."
        )
        msgs = [
            {"role": "system", "content": prompt},
            *history,
            {"role": "user", "content": "Genera el resumen."},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=model, messages=msgs,  # type: ignore[arg-type]
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - no debe romper el cierre
            return f"(No se pudo resumir la sesión: {exc})"
