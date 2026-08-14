"""CLI interactiva de Hermes.

Comandos disponibles dentro de la sesión:
  /ayuda                muestra esta ayuda
  /modelos              lista los modelos disponibles del proveedor
  /modelo <id>          cambia de modelo al instante (persistente)
  /voz                  activa/desactiva que Hermes hable (espeak-ng)
  /escuchar             graba el micrófono y envía lo transcrito
  /manoslibres          escucha continua: di «Hola Hermes» + tu petición
  /micro                diagnostica el micrófono (nivel de señal)
  /memoria              muestra los hechos guardados
  /recordar <texto>     guarda un hecho en la memoria a largo plazo
  /olvidar <texto>      borra un hecho de la memoria («todo» vacía toda la memoria)
  /limpiar              olvida la conversación actual (mantiene la memoria)
  /salir                termina la sesión
"""

from __future__ import annotations

import argparse
import readline
import sys
from datetime import datetime
from pathlib import Path

from hermes import __version__
from hermes.config import Config
from hermes.llm import GroqError, GroqLLM
from hermes.memory import Memory
from hermes.tools import TOOL_DEFS, ToolContext, dispatch
from hermes.voice import Speaker, Transcriber, VoiceError, record_until_silence

HELP_TEXT = """\
Comandos disponibles:
  /ayuda                muestra esta ayuda
  /modelos              lista los modelos disponibles del proveedor
  /modelo <id>          cambia de modelo al instante (persistente)
  /voz                  activa/desactiva que Hermes hable (espeak-ng)
  /escuchar             graba el micrófono y envía lo transcrito
  /manoslibres          escucha continua: di «Hola Hermes» + tu petición
  /micro                diagnostica el micrófono (nivel de señal)
  /memoria              muestra los hechos guardados
  /recordar <texto>     guarda un hecho en la memoria a largo plazo
  /olvidar <texto>      borra un hecho de la memoria («todo» vacía toda la memoria)
  /limpiar              olvida la conversación actual (mantiene la memoria)
  /salir                termina la sesión
"""

CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

SYSTEM_PROMPT = """Eres Hermes, el asistente personal del usuario que corre en su propia máquina, por CLI.
Habla siempre en español, de forma concisa y directa.

Tienes control de la máquina con herramientas: ejecuta comandos de shell cuando haga falta
(instalar programas, mover archivos, consultar el sistema, ejecutar scripts), lee y escribe
archivos, y usa tu memoria persistente.

Reglas:
- Usa herramientas cuando la respuesta requiera información real de la máquina; jamás inventes resultados.
- Los comandos de riesgo (borrar, instalar, git push, sudo...) requieren aprobación del usuario;
  el sistema se la pedirá automáticamente.
- Si algo falla, intenta diagnosticar y arreglarlo usando tus herramientas.
- Cuando el usuario diga algo que valga la pena recordar, guárdalo con memorize.
- Borra memoria SOLO cuando el usuario te lo pida explícitamente, con forget y su confirmación.

{memoria}

Entorno: directorio de trabajo {cwd} · modelo actual {model}."""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hermes", description="Hermes: asistente personal por CLI")
    p.add_argument("--home", help="directorio de datos de Hermes (por defecto ~/.hermes)")
    p.add_argument("--model", help="modelo inicial (persistente si se usa /modelo)")
    p.add_argument("--no-voice", action="store_true", help="no usar síntesis de voz")
    p.add_argument("--version", action="version", version=f"hermes {__version__}")
    return p.parse_args(argv)


def style(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{RESET}" if enabled else text


class HermesCLI:
    def __init__(self, args: argparse.Namespace):
        self.cfg = Config(args.home)
        self.cfg.load_env()
        if args.model:
            self.cfg.model = args.model
        self.cfg.ensure_dirs()
        self.memory = Memory(self.cfg.memory_dir, self.cfg.sessions_dir)
        self.session_id = self.memory.new_session()
        self.color = sys.stdin.isatty() and sys.stdout.isatty()
        self.voice = None if args.no_voice else Speaker(model=self.cfg.voice_model) if self.cfg.voice_enabled else None
        self.transcriber: Transcriber | None = None  # se crea al primer /escuchar
        self.llm: GroqLLM | None = None
        if self.cfg.api_key:
            try:
                self.llm = GroqLLM(self.cfg.api_key, base_url=self.cfg.base_url)
            except GroqError as exc:
                print(f"  {exc}")
        self.ctx = ToolContext(self.memory, confirm=self._confirm)
        self.context: list[dict] = []
        self._history: list[dict] = []  # para resumir al cerrar

    # ------------------------------------------------------------ helpers
    def _confirm(self, prompt: str) -> bool:
        if not sys.stdin.isatty():
            print(f"  [denegado automáticamente: entrada no interactiva]")
            return False
        try:
            return input(style(prompt, CYAN, self.color)).strip().lower() in ("s", "si", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _prompt(self) -> str:
        return style(f"\nHermes({self.cfg.model})> ", CYAN, self.color)

    def _rebuild_system(self) -> None:
        self.context = [{
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                memoria=self.memory.context_block(),
                cwd=self.ctx.cwd,
                model=self.cfg.model,
            ),
        }]
        self.context += self.memory.history(self.session_id)[-40:]

    def _say(self, text: str) -> None:
        if self.voice:
            self.voice.say(text)

    # ---------------------------------------------------------- commands
    def handle_command(self, raw: str) -> bool:
        cmd, _, arg = raw.partition(" ")
        cmd = cmd.lower()
        arg = arg.strip()

        if cmd in ("/salir", "/exit", "/quit"):
            return False
        if cmd in ("salir", "exit", "quit"):
            return False
        if cmd in ("/ayuda", "/help", "/?"):
            print(style(HELP_TEXT, DIM, self.color))
        elif cmd in ("/modelos", "/models"):
            self._cmd_models()
        elif cmd in ("/modelo", "/model"):
            self._cmd_model(arg)
        elif cmd == "/voz":
            self._cmd_voz()
        elif cmd == "/escuchar":
            self._cmd_escuchar()
        elif cmd in ("/manoslibres", "/manos"):
            self._cmd_manoslibres()
        elif cmd == "/micro":
            self._cmd_micro()
        elif cmd in ("/memoria", "/memory"):
            self._cmd_memoria()
        elif cmd in ("/recordar", "/remember"):
            self._cmd_recordar(arg)
        elif cmd in ("/olvidar", "/forget"):
            self._cmd_olvidar(arg)
        elif cmd == "/limpiar":
            self._rebuild_system()
            print(style("  Conversación olvidada. La memoria a largo plazo sigue intacta.", DIM, self.color))
        else:
            print(style(f"  Comando desconocido: {cmd} (usa /ayuda)", DIM, self.color))
        return True

    def _cmd_models(self) -> None:
        if self.llm is None:
            print("  No hay API key configurada; no puedo listar modelos.")
            return
        try:
            models = self.llm.list_models()
        except GroqError as exc:
            print(f"  {exc}")
            return
        print("  Modelos disponibles:")
        for m in models:
            marca = style(" <- actual", BOLD, self.color) if m == self.cfg.model else ""
            print(f"    - {m}{marca}")

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            print(f"  Modelo actual: {self.cfg.model}")
            return
        self.cfg.model = arg
        self._rebuild_system()
        print(style(f"  Modelo cambiado a: {arg}", BOLD, self.color))

    def _cmd_voz(self) -> None:
        self.cfg.voice_enabled = not self.cfg.voice_enabled
        self.voice = None if not self.cfg.voice_enabled else Speaker()
        print(f"  Voz {'activada' if self.cfg.voice_enabled else 'desactivada'} (espeak-ng).")

    def _cmd_memoria(self) -> None:
        facts = self.memory.facts()
        print("  Memoria a largo plazo:")
        print(facts if facts else style("    (vacía)", DIM, self.color))

    def _cmd_recordar(self, arg: str) -> None:
        if not arg:
            print("  Uso: /recordar <texto>")
            return
        print(f"  {self.memory.add_fact(arg) and 'OK: recordado.' or 'Ya estaba guardado.'}")

    def _cmd_olvidar(self, arg: str) -> None:
        if not arg:
            print("  Uso: /olvidar <texto>  o  /olvidar todo")
            return
        if arg.lower() in ("todo", "toda", "toda la memoria"):
            if not self._confirm("¿Borrar TODA la memoria a largo plazo (hechos y resúmenes)? [s/N] "):
                print("  Cancelado: no se borró nada.")
                return
            self.memory.clear_memory()
            print("  OK: memoria a largo plazo borrada por completo.")
            return
        if self.memory.delete_fact(arg):
            print(f"  OK: borrado de la memoria: {arg}")
        else:
            print(f"  No encontré en la memoria nada que coincida con: {arg}")

    def _cmd_escuchar(self) -> None:
        print(style("  Escuchando... (silencio para terminar)", DIM, self.color))
        try:
            wav = record_until_silence(device=self.cfg.mic_device)
        except VoiceError as exc:
            print(f"  Error de audio: {exc}")
            return
        if self.transcriber is None:
            self.transcriber = Transcriber(self.cfg.whisper_model)
        try:
            text = self.transcriber.transcribe(wav)
        except VoiceError as exc:
            print(f"  Error de transcripción: {exc}")
            return
        if not text:
            print(style("  No entendí nada.", DIM, self.color))
            return
        print(style(f"  🗣️ {text}", DIM, self.color))
        self.handle_message(text)

    def _cmd_manoslibres(self) -> None:
        if self.llm is None:
            print("  No hay HERMES_API_KEY configurada (ver ~/.hermes/.env).")
            return
        if self.transcriber is None:
            self.transcriber = Transcriber(self.cfg.whisper_model)
        from hermes.voice import HandsFree

        handsfree = HandsFree(self.transcriber, device=self.cfg.mic_device)
        print(style("  🎤 Modo manos libres: di «Hola Hermes» seguido de tu petición. "
                     "Ctrl+C para volver al modo normal.", BOLD, self.color))
        try:
            while True:
                try:
                    command = handsfree.listen_for_command(
                        on_wake=lambda: print(style("  🤖 Dime...", DIM, self.color))
                    )
                except VoiceError as exc:
                    print(style(f"  ⚠ {exc}", BOLD, self.color))
                    print(style("  Modo manos libres desactivado.", DIM, self.color))
                    return
                if command is None:
                    break
                if not command.strip():
                    print(style("  (no capté la petición tras la wake word)", DIM, self.color))
                    continue
                print(style(f"  🗣️ {command}", DIM, self.color))
                self.handle_message(command)
        except KeyboardInterrupt:
            print(style("\n  Modo manos libres desactivado.", DIM, self.color))

    # ---------------------------------------------------------- messages
    def handle_message(self, text: str) -> None:
        if self.llm is None:
            print("  No hay HERMES_API_KEY configurada (ver ~/.hermes/.env). "
                  "Consíguela en el panel de tu proveedor LLM.")
            return
        self._rebuild_system()
        user_msg = {"role": "user", "content": text}
        self.context.append(user_msg)
        self._history.append(user_msg)

        print()
        try:
            reply = self.llm.run(
                messages=self.context,
                tools=TOOL_DEFS,
                model=self.cfg.model,
                on_delta=lambda t: print(t, end="", flush=True),
                on_tool=self._on_tool,
                dispatch=lambda name, args: dispatch(self.ctx, name, args),
            )
        except GroqError as exc:
            print(style(f"\n  ✗ {exc}", BOLD, self.color))
            return
        print()
        self._say(reply)

        asst_msg = {"role": "assistant", "content": reply}
        self.context.append(asst_msg)
        self._history.append(asst_msg)
        self.memory.append(self.session_id, user_msg)
        self.memory.append(self.session_id, asst_msg)

    def _cmd_micro(self) -> None:
        """Diagnóstico del micrófono: nivel de señal y consejos."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            print("  Falta sounddevice.")
            return
        try:
            name = sd.query_devices(self.cfg.mic_device or sd.default.device[0])["name"]
        except Exception:  # noqa: BLE001
            name = "(desconocido)"
        print(style(f"  🎙 Entrada: {name}", DIM, self.color))
        print(style("  Grabando 3s — habla o haz ruido...", DIM, self.color))
        try:
            rec = sd.rec(int(3 * 16000), samplerate=16000, channels=1,
                         dtype="float32", device=self.cfg.mic_device)
            sd.wait()
            rms = float(np.sqrt(np.mean(rec**2)))
            peak = float(np.max(np.abs(rec)))
        except Exception as exc:  # noqa: BLE001
            print(f"  Error al grabar: {exc}")
            return
        print(f"  Nivel: peak={peak:.3f} | rms={rms:.4f}")
        if rms > 0.5:
            print(style("  ⚠ Micrófono SATURADO: señal al máximo constante. Baja la "
                         "ganancia (amixer / ajustes del sistema), reinicia o usa otro "
                         "dispositivo con HERMES_MIC_DEVICE.", BOLD, self.color))
        elif rms < 0.005:
            print(style("  ⚠ Nivel muy bajo: habla más cerca o sube el volumen del mic.", BOLD, self.color))
        else:
            print(style("  ✓ Nivel correcto para voz.", DIM, self.color))

    def _on_tool(self, name: str, args: str) -> None:
        try:
            shown = f"{name}({args[:80]})"
        except TypeError:
            shown = name
        print(style(f"\n  ⚙ {shown}", DIM, self.color))

    # ------------------------------------------------------------- loop
    def run(self) -> None:
        try:
            readline.read_history_file(str(self.cfg.home / "history"))
        except (OSError, ValueError):
            pass
        try:
            readline.set_history_length(1000)
        except AttributeError:
            pass

        print(style(f"Hermes v{__version__} — asistente personal (modelo: {self.cfg.model})", BOLD, self.color))
        print(style("  Escribe /ayuda para ver los comandos. /salir para terminar.", DIM, self.color))
        if self.llm is None:
            print(style("  ⚠ Sin HERMES_API_KEY: ponla en ~/.hermes/.env y reinicia.", BOLD, self.color))

        running = True
        while running:
            try:
                line = input(self._prompt())
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            line = line.strip()
            if not line:
                continue
            try:
                readline.add_history(line)
            except AttributeError:
                pass
            if line.startswith("/") or line.lower() in ("salir", "exit", "quit"):
                running = self.handle_command(line)
            else:
                self.handle_message(line)

        self._close()

    def _close(self) -> None:
        """Al cerrar, resume la sesión y la archiva en memoria."""
        if self.llm is not None and len(self._history) >= 4:
            print(style("  Guardando memoria de la sesión...", DIM, self.color))
            try:
                summary = self.llm.summarize(self._history[-20:], self.cfg.model)
                self.memory.add_summary(summary)
                print(style("  ✔ Memoria actualizada.", DIM, self.color))
            except GroqError:
                pass
        try:
            readline.write_history_file(str(self.cfg.home / "history"))
        except OSError:
            pass


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        HermesCLI(args).run()
    except KeyboardInterrupt:
        print(style("\n  Hasta luego.", DIM, False))
    except Exception as exc:  # noqa: BLE001 - nunca cerrar sin dejar rastro
        import traceback as tb
        tb.print_exc()
        try:
            log = Path.home() / ".hermes" / "error.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n--- {datetime.now().isoformat()} ---\n")
                tb.print_exc(file=fh)
            print(style(f"\n  Error inesperado. Detalle guardado en {log}", BOLD, False))
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
