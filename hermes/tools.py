"""Herramientas que Hermes puede usar para manejar la máquina.

La herramienta `shell` ejecuta comandos del sistema con confirmación previa
cuando el comando encaja con patrones de riesgo (borrados, instalaciones,
git push, sudo, etc.). `ToolContext.confirm` es inyectado por la CLI para
preguntar al usuario; en modo no interactivo se deniega automáticamente.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, cast

from hermes.memory import Memory

SHELL_TIMEOUT = 120          # segundos por comando
MAX_OUTPUT = 20000           # caracteres de salida devueltos al modelo
MAX_FILE_CHARS = 30000       # caracteres máximos al leer un archivo

# Patrones que requieren confirmación del usuario antes de ejecutarse.
DANGER_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+(-[a-z]*r[a-z]*\s+)?(-[a-z]*f[a-z]*\s+)?"),      # rm -rf / rm -r
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b.*\bof=/dev/"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bgit\s+push\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\b"),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b"),
    re.compile(r"\bwget\b.*\|\s*(ba)?sh\b"),
    re.compile(r"\b(shutdown|reboot|poweroff|halt|init\s+0)\b"),
    re.compile(r"\bchmod\s+-R\s+777\s+/\b"),
    re.compile(r"\bchown\b.*\s/\b"),
    re.compile(r"\b:\(\)\{"),
    re.compile(r"\b(pkexec|doas)\b"),
    re.compile(r"\b(apt|apt-get|dnf|yum|pacman|zypper)\s+(install|remove|purge|autoremove)\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bpasswd\b"),
    re.compile(r"\b(useradd|userdel|usermod|groupadd|groupdel)\b"),
    re.compile(r"\bmount\b"),
    re.compile(r"\bkill\s+-9\b"),
    re.compile(r"\b>\s*/dev/sd"),
]

# Palabras clave que hacen sospechoso un comando sin coincidir con lo anterior.
DANGER_WORDS = ("rm -rf", "rm -fr", "mkfs", "dd ", "sudo", "git push", "passwd",
                "shutdown", "reboot", "chmod -R 777", "curl | sh", "wget | sh")


class ToolContext:
    """Estado compartido por las herramientas durante la sesión."""

    def __init__(self, memory: Memory, cwd: Path | None = None,
                 confirm=None):
        self.memory = memory
        self.cwd = Path(cwd or Path.cwd())
        # confirm(prompt: str) -> bool; si es None, se deniega lo arriesgado.
        self.confirm = confirm or (lambda prompt: False)

    def is_dangerous(self, command: str) -> bool:
        if any(p.search(command) for p in DANGER_PATTERNS):
            return True
        low = command.lower()
        return any(w in low for w in DANGER_WORDS)


# ------------------------------------------------------------------ shell
def run_shell(ctx: ToolContext, command: str) -> str:
    command = command.strip()
    if not command:
        return "ERROR: comando vacío."

    # `cd` cambia el cwd persistente de la sesión.
    m = re.match(r"^cd\s+(\S.*)$", command)
    if m:
        target = Path(m.group(1).strip().strip('"').strip("'"))
        if not target.is_absolute():
            target = ctx.cwd / target
        if not target.exists() or not target.is_dir():
            return f"ERROR: no existe el directorio: {target}"
        ctx.cwd = target.resolve()
        return f"Directorio actual: {ctx.cwd}"

    if ctx.is_dangerous(command):
        if not ctx.confirm(f"¿Ejecutar comando de riesgo?\n  $ {command}\n[s/N] "):
            return "DENEGADO: el usuario no aprobó este comando de riesgo."
        ctx.memory.add_fact(f"El usuario aprobó ejecutar: {command[:120]}")

    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(ctx.cwd),
            capture_output=True, text=True, timeout=SHELL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: el comando superó {SHELL_TIMEOUT}s y fue cancelado."
    except OSError as exc:
        return f"ERROR al ejecutar: {exc}"

    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n... (salida truncada, {len(out)} caracteres)"
    status = f"\n[exit {proc.returncode}]"
    return (out + status) if out.strip() else f"(sin salida){status}"


# ------------------------------------------------------------------ files
def read_file(ctx: ToolContext, path: str, max_chars: int = MAX_FILE_CHARS) -> str:
    p = _resolve(ctx, path)
    if not p.exists():
        return f"ERROR: no existe: {p}"
    if p.is_dir():
        return f"ERROR: es un directorio, usa list_dir: {p}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR al leer {p}: {exc}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (archivo truncado, {len(text)} caracteres)"
    return text


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    p = _resolve(ctx, path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ERROR al escribir {p}: {exc}"
    return f"OK: escrito {p} ({len(content)} caracteres)"


def list_dir(ctx: ToolContext, path: str = ".") -> str:
    p = _resolve(ctx, path)
    if not p.exists():
        return f"ERROR: no existe: {p}"
    if not p.is_dir():
        return f"ERROR: no es un directorio: {p}"
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except OSError as exc:
        return f"ERROR al listar {p}: {exc}"
    lines = [f"{'[dir] ' if e.is_dir() else ''}{e.name}" for e in entries]
    if not lines:
        lines = ["(directorio vacío)"]
    return f"{p}:\n" + "\n".join(lines)


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ctx.cwd / p
    return p.resolve()


# ------------------------------------------------------------------ memory
def memorize(ctx: ToolContext, texto: str) -> str:
    if ctx.memory.add_fact(texto):
        return "OK: recordado a largo plazo."
    return "Ya tenía ese hecho guardado."


def recall(ctx: ToolContext, consulta: str) -> str:
    hits = ctx.memory.recall(consulta)
    return hits if hits else "No encontré nada en la memoria con esa consulta."


def forget(ctx: ToolContext, texto: str) -> str:
    """Borra memoria a largo plazo, SOLO con confirmación del usuario.

    `texto` puede ser un hecho concreto a borrar, o «todo» para vaciar toda
    la memoria (hechos y resúmenes de sesiones).
    """
    texto = texto.strip()
    if texto.lower() in ("todo", "toda", "toda la memoria", "todo lo que sabes"):
        if not ctx.confirm("¿Borrar TODA la memoria a largo plazo (hechos y resúmenes)? [s/N] "):
            return "Cancelado: no borré nada."
        ctx.memory.clear_memory()
        return "OK: memoria a largo plazo borrada por completo."
    if not texto:
        return "Indica qué borrar (un hecho concreto o «todo»)."
    if not ctx.confirm(f"¿Borrar de la memoria: «{texto[:80]}»? [s/N] "):
        return "Cancelado: no borré nada."
    if ctx.memory.delete_fact(texto):
        return f"OK: borrado de la memoria: {texto}"
    return f"No encontré en la memoria nada que coincida con: {texto}"


# ------------------------------------------------------------ definiciones
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Ejecuta un comando de shell en la máquina del usuario. "
                           "Úsalo para instalar programas, mover archivos, consultar "
                           "sistema, ejecutar scripts, etc. Los comandos de riesgo "
                           "piden confirmación al usuario.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string",
                                           "description": "Comando shell a ejecutar"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido de un archivo de texto (relativo al cwd actual).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Crea o sobrescribe un archivo con el contenido dado "
                           "(relativo al cwd actual). Crea los directorios si faltan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lista el contenido de un directorio (relativo al cwd actual).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memorize",
            "description": "Guarda un hecho a largo plazo sobre el usuario o el entorno. "
                           "Úsalo cuando el usuario pida recordar algo para siempre.",
            "parameters": {
                "type": "object",
                "properties": {"texto": {"type": "string",
                                         "description": "Hecho a recordar"}},
                "required": ["texto"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Busca en la memoria persistente (hechos y resúmenes de "
                           "sesiones anteriores) algo que coincida con la consulta.",
            "parameters": {
                "type": "object",
                "properties": {"consulta": {"type": "string"}},
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Borra memoria a largo plazo. ÚSALA SOLO cuando el usuario "
                           "lo pida explícitamente (por ejemplo «olvídate de...» o "
                           "«borra tu memoria»). Pide confirmación al usuario antes de "
                           "borrar; nunca borres memoria por tu cuenta. El texto «todo» "
                           "vacía toda la memoria.",
            "parameters": {
                "type": "object",
                "properties": {"texto": {
                    "type": "string",
                    "description": "Hecho concreto a borrar, o «todo» para vaciar toda la memoria"
                }},
                "required": ["texto"],
            },
        },
    },
]

HANDLERS = {
    "shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "memorize": memorize,
    "recall": recall,
    "forget": forget,
}


def dispatch(ctx: ToolContext, name: str, arguments: str) -> str:
    """Ejecuta una llamada a herramienta y devuelve el resultado como texto."""
    handler = HANDLERS.get(name)
    if handler is None:
        return f"ERROR: herramienta desconocida: {name}"
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return f"ERROR: argumentos inválidos para {name}: {exc}"
    if not isinstance(args, dict):
        return f"ERROR: argumentos inválidos para {name} (se esperaba un objeto)."
    try:
        handler = cast(Callable[..., str], handler)
        return str(handler(ctx, **args))
    except TypeError as exc:
        return f"ERROR: argumentos incorrectos para {name}: {exc}"
