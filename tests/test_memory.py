from hermes.memory import Memory
from pathlib import Path


def make_memory(tmp_path: Path) -> Memory:
    mem = Memory(tmp_path / "memory", tmp_path / "memory" / "conversations")
    mem.memory_dir.mkdir(parents=True, exist_ok=True)
    mem.sessions_dir.mkdir(parents=True, exist_ok=True)
    return mem


def test_add_fact_and_dedupe(tmp_path):
    mem = make_memory(tmp_path)
    assert mem.add_fact("Me llamo Nexan") is True
    assert mem.add_fact("Me llamo Nexan") is False  # duplicado
    assert "Me llamo Nexan" in mem.facts()


def test_recall_finds_facts_and_summaries(tmp_path):
    mem = make_memory(tmp_path)
    mem.add_fact("Mi servidor favorito es Nginx")
    mem.add_summary("Se instaló Nginx en /etc/nginx")
    hits = mem.recall("nginx")
    assert "Nginx" in hits
    assert "/etc/nginx" in hits
    assert mem.recall("postgres") == ""


def test_session_history_and_clear(tmp_path):
    mem = make_memory(tmp_path)
    sid = mem.new_session()
    mem.append(sid, {"role": "user", "content": "hola"})
    mem.append(sid, {"role": "assistant", "content": "hola!"})
    hist = mem.history(sid)
    assert len(hist) == 2
    assert hist[-1]["content"] == "hola!"
    mem.clear_session(sid)
    assert mem.history(sid) == []


def test_context_block_includes_facts(tmp_path):
    mem = make_memory(tmp_path)
    assert "(Aún no hay memoria" in mem.context_block()
    mem.add_fact("El usuario odia el café")
    block = mem.context_block()
    assert "El usuario odia el café" in block
