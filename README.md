# Hermes 👽

Asistente personal por **CLI** con **memoria persistente**, **voz** (escucha y habla)
y **control total de la máquina** donde se ejecuta. Usa la API de **NVIDIA**
(compatible con OpenAI) y puede **cambiar de modelo en caliente** con un comando.

## Requisitos

- Linux con Python 3.11+
- `espeak-ng` (voz) — ya instalado en la mayoría de distribuciones
- Micrófono (para `/escuchar`)
- Una API key de un proveedor LLM compatible con OpenAI (por defecto [NVIDIA](https://build.nvidia.com))

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Guarda tu API key (la primera vez):
mkdir -p ~/.hermes
echo 'HERMES_API_KEY=tu_clave_aqui' > ~/.hermes/.env
```

## Uso

```bash
.venv/bin/hermes
```

Dentro de la sesión:

| Comando | Qué hace |
|---|---|
| `/modelos` | lista los modelos disponibles del proveedor |
| `/modelo <id>` | cambia de modelo al instante (queda guardado) |
| `/escuchar` | graba el micrófono, transcribe con Whisper local y lo envía |
| `/manoslibres` | escucha continua: di «Hola Hermes» + tu petición (Ctrl+C sale) |
| `/voz` | activa/desactiva que Hermes hable (espeak-ng) |
| `/recordar <texto>` | guarda un hecho en la memoria a largo plazo |
| `/memoria` | muestra los hechos guardados |
| `/limpiar` | olvida la conversación actual (mantiene la memoria) |
| `/ayuda` · `/salir` | ayuda · terminar |

Ejemplos de conversación:

```
Hermes(nvidia/nemotron-3-super-120b-a12b)> recuerda que mi proyecto favorito es HermesAsist
  ⚙ memorize({"texto": "El proyecto favorito del usuario es HermesAsist"})
  ✅ Recordado. Lo tendré presente en el futuro.
```

```
Hermes(nvidia/nemotron-3-super-120b-a12b)> ¿cuánto espacio libre tiene el disco?
  ⚙ shell(df -h /)
  ✅ Tienes 45G libres de 100G en la raíz.
```

## Qué hace Hermes

- **Memoria persistente** (`~/.hermes/memory/`): hechos a largo plazo (`facts.md`),
  resúmenes de cada sesión (`summaries.md`) que se inyectan al empezar, e historial
  completo de conversaciones.
- **Voz**: escucha con micrófono + **faster-whisper** local (español, sin internet)
  y habla con **espeak-ng** en un hilo en segundo plano. Modo **manos libres**
  (`/manoslibres`): escucha continua que solo transcribe cuando hay voz,
  se activa al oír «Hola Hermes» (o «Hey Hermes») y captura el comando hablado.
- **Control de la máquina**: herramientas de shell (con **confirmación para comandos
  de riesgo** como `rm -rf`, `sudo`, `git push`, instalaciones...), lectura/escritura
  de archivos y listado de directorios. El directorio de trabajo se mantiene entre
  comandos (soporta `cd`).
- **Modelos intercambiables**: `nvidia/nemotron-3-super-120b-a12b` por defecto, pero
  puedes cambiar a cualquiera de los modelos de chat de tu proveedor con `/modelo`
  (y de proveedor con `HERMES_BASE_URL`). Nota: el ultra-550b se queda encolado
  en el free tier; si lo eliges, un timeout te lo avisará y sugerirá uno más rápido.

## Estructura

```
hermes/
  cli.py     REPL interactiva, comandos, voz y bucle de herramientas
  llm.py     cliente LLM OpenAI-compatible (streaming + tool calling + modelos)
  tools.py   shell con filtro de riesgo, archivos, memoria
  memory.py  hechos, historial de sesiones y resúmenes
  voice.py   grabación de micrófono, Whisper local, espeak-ng
  config.py  configuración persistente (~/.hermes) y API key
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
