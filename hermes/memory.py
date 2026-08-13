"""Memoria persistente de Hermes: hechos, resúmenes y conversaciones.

  memory/facts.md                hechos a largo plazo (lo que el usuario le pide recordar)
  memory/summaries.md            resúmenes de sesiones anteriores
  memory/conversations/*.jsonl   historial completo de cada sesión
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Número máximo de mensajes que se recuerdan del historial de la sesión actual.
MAX_HISTORY = 40


class Memory:
    def __init__(self, memory_dir: str | Path, sessions_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.sessions_dir = Path(sessions_dir)
        self.facts_path = self.memory_dir / "facts.md"
        self.summaries_path = self.memory_dir / "summaries.md"

    # ------------------------------------------------------------ facts
    def add_fact(self, text: str) -> bool:
        """Guarda un hecho a largo plazo. Devuelve False si ya existía."""
        text = text.strip()
        if not text:
            return False
        existing = self.facts_path.read_text(encoding="utf-8") if self.facts_path.exists() else ""
        if text in existing:
            return False
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self.facts_path.open("a", encoding="utf-8") as fh:
            fh.write(f"- [{stamp}] {text}\n")
        return True

    def facts(self) -> str:
        if not self.facts_path.exists():
            return ""
        return self.facts_path.read_text(encoding="utf-8").strip()

    def recall(self, query: str, limit: int = 10) -> str:
        """Busca coincidencias (sin distinguir mayúsculas) en hechos y resúmenes."""
        q = query.lower()
        hits: list[str] = []
        for path in (self.facts_path, self.summaries_path):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and q in line.lower():
                    hits.append(line.strip())
        return "\n".join(hits[-limit:])

    # ------------------------------------------------------ conversations
    def new_session(self) -> str:
        sid = datetime.now().strftime("%Y%m%d-%H%M%S")
        return sid

    def session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def append(self, session_id: str, message: dict) -> None:
        with self.session_path(session_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False) + "\n")

    def history(self, session_id: str, limit: int = MAX_HISTORY) -> list[dict]:
        path = self.session_path(session_id)
        if not path.exists():
            return []
        msgs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return msgs[-limit:]

    def clear_session(self, session_id: str) -> None:
        self.session_path(session_id).unlink(missing_ok=True)

    # -------------------------------------------------------- summaries
    def add_summary(self, summary: str) -> None:
        summary = summary.strip()
        if not summary:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self.summaries_path.open("a", encoding="utf-8") as fh:
            fh.write(f"## Sesión {stamp}\n{summary}\n\n")

    def recent_summaries(self, limit: int = 3) -> str:
        if not self.summaries_path.exists():
            return ""
        parts = self.summaries_path.read_text(encoding="utf-8").strip().split("\n## ")
        return "\n".join(p.strip() for p in parts[-limit:])

    # ------------------------------------------------- context block
    def context_block(self) -> str:
        """Bloque de contexto inyectado en el system prompt."""
        blocks: list[str] = []
        facts = self.facts()
        if facts:
            blocks.append("HECHOS QUE SÉ DEL USUARIO (memoria a largo plazo):\n" + facts)
        sums = self.recent_summaries()
        if sums:
            blocks.append("RESUMEN DE SESIONES ANTERIORES:\n" + sums)
        if not blocks:
            blocks.append("(Aún no hay memoria guardada de sesiones anteriores.)")
        return "\n\n".join(blocks)
