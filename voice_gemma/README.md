# Voice Gemma

Local **voice + text** assistant that chains:

1. **faster-whisper** — offline speech-to-text (multilingual)  
2. **LM Studio** — OpenAI-compatible chat API (e.g. Gemma GGUF)  
3. **edge-tts** — free neural text-to-speech for playback  

---

## Requirements

- Python 3.10+ (3.14 tested)  
- [LM Studio](https://lmstudio.ai/) with your model loaded and **Local Server** started  
- Microphone (for CLI recording or browser recording in the web UI)  
- Optional: **FFmpeg** on `PATH` for some audio formats (many paths work without it)  
- Internet for **edge-tts** (Microsoft’s edge read-aloud endpoint)

---

## Install

```bash
cd voice_gemma
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and edit:

| Variable | Description |
|----------|-------------|
| `LMSTUDIO_BASE_URL` | API base, usually `http://127.0.0.1:1234/v1` |
| `LMSTUDIO_API_KEY` | Key from LM Studio if enabled |
| `LMSTUDIO_MODEL` | Exact model id shown under **API Usage** in LM Studio |

The web UI can override these per session (stored in `localStorage`). Do not commit `.env`; it is gitignored.

---

## CLI (`main.py`)

**Voice + optional typed text (default: records from mic):**

```bash
python main.py
```

**Text only (no Whisper):**

```bash
python main.py --no-mic -t "Hello, how are you?" --language en
```

**Useful flags:** `--lm-url`, `--api-key`, `--chat-model`, `--whisper-model`, `--language`, `--no-tts`, `--reply-mp3 out.mp3`.

---

## Web UI (`server.py`)

```bash
python server.py
```

Open **http://127.0.0.1:8765** (change with env `VOICE_GEMMA_WEB_PORT`).

The UI follows a **ChatGPT-like** layout: sidebar (new chat, settings, voice transcript), main thread, and a bottom composer with **attachments**, text, mic, and send.

- **Attach**: `.txt`, `.md`, `.csv`, `.json`, code files, `.pdf` (needs `pypdf`), etc. Text is extracted on the server and prepended under `[Attached files]` in the prompt.  
- **POST /api/chat** with `multipart/form-data`: same field names as before, plus repeated `files` parts. JSON body is still supported (no file upload).

---

## API (for custom clients)

| Method | Path | Purpose |
|--------|------|--------|
| `GET` | `/api/health` | Liveness |
| `POST` | `/api/transcribe` | `multipart/form-data`: `file`, optional `language`, `whisper_model`, `device`, `compute_type` |
| `POST` | `/api/chat` | `application/json` **or** `multipart/form-data`: text fields + optional repeated `files` (uploads) |
| `POST` | `/api/tts` | `application/x-www-form-urlencoded` or form: `text`, `voice` |

---

## Troubleshooting

- **`connection refused` to LM Studio** — Start **Local Server** in LM Studio; match host/port to `LMSTUDIO_BASE_URL`.  
- **`ModuleNotFoundError`** — Run `pip install -r requirements.txt` inside your venv.  
- **`model not found`** — Use the exact **API Usage** model string from LM Studio in `LMSTUDIO_MODEL` / Settings.

---

## License

See repository root `LICENSE` if present.
