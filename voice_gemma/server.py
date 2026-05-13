"""Web UI and HTTP API for Voice Gemma (FastAPI)."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import edge_tts
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError
from pydantic import BaseModel

from attachments import attachment_blocks_from_uploads
from core import (
    ask_lm_studio,
    edge_voice_for_whisper_lang,
    lm_studio_connection_hint,
    load_local_env,
    make_whisper_model,
    merge_user_message,
    openai_client,
    resolve_whisper_device,
    settings_from_env_overrides,
    transcribe_file,
)

_static = Path(__file__).resolve().parent / "static"

DEFAULT_SYSTEM = (
    "You are a helpful, concise multilingual assistant. "
    "Answer in the same language as the user unless they ask otherwise."
)

_whisper_lock = threading.Lock()
_whisper_model = None
_whisper_key: tuple[str, str, str] | None = None


class ChatBody(BaseModel):
    typed: str | None = None
    voice_text: str | None = None
    language: str | None = None
    tts_voice: str | None = None
    system_prompt: str | None = None
    lm_url: str | None = None
    api_key: str | None = None
    model_id: str | None = None


def _form_str(form, key: str) -> str | None:
    v = form.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _get_whisper(model_size: str, device_flag: str, compute_type: str):
    global _whisper_model, _whisper_key
    dev = resolve_whisper_device(device_flag)
    ct = compute_type if dev == "cuda" else "int8"
    key = (model_size, dev, ct)
    with _whisper_lock:
        if _whisper_model is None or _whisper_key != key:
            _whisper_model = make_whisper_model(model_size, dev, ct)
            _whisper_key = key
    return _whisper_model


def create_app() -> FastAPI:
    load_local_env()
    app = FastAPI(title="Voice Gemma", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.post("/api/transcribe")
    async def api_transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        whisper_model: str = Form("small"),
        device: str = Form("auto"),
        compute_type: str = Form("int8"),
    ):
        model = _get_whisper(whisper_model, device, compute_type)
        name = file.filename or "rec.webm"
        suffix = Path(name).suffix if Path(name).suffix else ".webm"
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Empty audio upload")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            text, lang = transcribe_file(model, path, language)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return {"text": text, "language": lang}

    @app.post("/api/chat")
    async def api_chat(request: Request):
        ct = (request.headers.get("content-type") or "").lower()
        attach_block = ""
        if "application/json" in ct:
            data = await request.json()
            chat_in = ChatBody.model_validate(data)
        else:
            form = await request.form()
            chat_in = ChatBody(
                typed=_form_str(form, "typed"),
                voice_text=_form_str(form, "voice_text"),
                language=_form_str(form, "language"),
                tts_voice=_form_str(form, "tts_voice"),
                system_prompt=_form_str(form, "system_prompt"),
                lm_url=_form_str(form, "lm_url"),
                api_key=_form_str(form, "api_key"),
                model_id=_form_str(form, "model_id"),
            )
            uploads = [
                f for f in form.getlist("files") if isinstance(f, UploadFile)
            ]
            attach_block = await attachment_blocks_from_uploads(uploads)

        msg = merge_user_message(chat_in.typed, chat_in.voice_text)
        if attach_block:
            if msg.strip():
                msg = f"{msg.rstrip()}\n\n[Attached files]\n{attach_block}"
            else:
                msg = f"[Attached files]\n{attach_block}"
        if not msg.strip():
            raise HTTPException(
                400, "Provide a message, voice transcript, and/or text attachments"
            )

        base, key, mid = settings_from_env_overrides(
            chat_in.lm_url, chat_in.api_key, chat_in.model_id
        )
        client = openai_client(base, key)
        try:
            reply = ask_lm_studio(
                client,
                mid,
                msg,
                (chat_in.system_prompt or "").strip() or DEFAULT_SYSTEM,
            )
        except APIConnectionError:
            raise HTTPException(
                503,
                detail=lm_studio_connection_hint(base),
            ) from None
        voice = chat_in.tts_voice or edge_voice_for_whisper_lang(chat_in.language)
        return {"reply": reply, "tts_voice": voice, "model_id": mid}

    @app.post("/api/tts")
    async def api_tts(
        text: str = Form(...),
        voice: str = Form(...),
    ):
        if not text.strip():
            raise HTTPException(400, "TTS text is empty")
        comm = edge_tts.Communicate(text.strip(), voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
        try:
            await comm.save(path)
            data = Path(path).read_bytes()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return Response(content=data, media_type="audio/mpeg")

    @app.get("/")
    async def spa_index():
        index = _static / "index.html"
        if not index.is_file():
            raise HTTPException(500, "static/index.html not found")
        return FileResponse(index)

    app.mount(
        "/assets",
        StaticFiles(directory=str(_static)),
        name="assets",
    )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("VOICE_GEMMA_WEB_PORT", "8765"))
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
