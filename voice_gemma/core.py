"""Logic shared by CLI (main.py) and web UI (server.py)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from openai import OpenAI

DEFAULT_LMSTUDIO_CHAT_MODEL = (
    "nomic-ai/nomic-embed-text-v1.5-gguf/gemma-4-26b-a4b-it-ud-q4_k_m.gguf"
)

WHISPER_LANG_TO_EDGE_VOICE: dict[str, str] = {
    "en": "en-US-AriaNeural",
    "fa": "fa-IR-DilaraNeural",
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-SA-ZariyahNeural",
    "hi": "hi-IN-SwaraNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "uk": "uk-UA-PolinaNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "id": "id-ID-GadisNeural",
    "th": "th-TH-PremwadeeNeural",
    "cs": "cs-CZ-VlastaNeural",
    "el": "el-GR-AthinaNeural",
    "he": "he-IL-HilaNeural",
    "sv": "sv-SE-SofieNeural",
    "no": "nb-NO-PernilleNeural",
    "da": "da-DK-ChristelNeural",
    "fi": "fi-FI-SelmaNeural",
}


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if val.startswith('"') and val.endswith('"') and len(val) >= 2:
            val = val[1:-1]
        elif val.startswith("'") and val.endswith("'") and len(val) >= 2:
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def edge_voice_for_whisper_lang(lang_code: str | None) -> str:
    if not lang_code:
        return "en-US-AriaNeural"
    base = lang_code.split("-")[0].lower()
    return WHISPER_LANG_TO_EDGE_VOICE.get(base, "en-US-AriaNeural")


def merge_user_message(typed: str | None, voice_text: str | None) -> str:
    t = (typed or "").strip()
    v = (voice_text or "").strip()
    if t and v:
        return f"{t}\n\n[Speech]: {v}"
    if t:
        return t
    return v


def transcribe_ndarray(
    model: WhisperModel,
    audio: np.ndarray,
    sample_rate: int,
    language: str | None,
) -> tuple[str, str | None]:
    kwargs: dict = {"beam_size": 5, "vad_filter": True}
    if language:
        kwargs["language"] = language
    segments, info = model.transcribe(audio, **kwargs)
    parts: list[str] = []
    for seg in segments:
        parts.append(seg.text)
    text = " ".join(p.strip() for p in parts if p.strip()).strip()
    lang = info.language if hasattr(info, "language") else language
    return text, lang


def transcribe_file(
    model: WhisperModel,
    path: str | Path,
    language: str | None,
) -> tuple[str, str | None]:
    kwargs: dict = {"beam_size": 5, "vad_filter": True}
    if language:
        kwargs["language"] = language
    segments, info = model.transcribe(str(path), **kwargs)
    parts: list[str] = []
    for seg in segments:
        parts.append(seg.text)
    text = " ".join(p.strip() for p in parts if p.strip()).strip()
    lang = info.language if hasattr(info, "language") else language
    return text, lang


def ask_lm_studio(
    client: OpenAI,
    model: str,
    user_text: str,
    system_prompt: str | None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


def resolve_whisper_device(explicit: str) -> str:
    if explicit != "auto":
        return explicit
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    ct = compute_type if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=ct)


def lm_studio_connection_hint(base_url: str) -> str:
    return (
        f"Could not connect to LM Studio.\n"
        f"URL: {base_url}\n"
        "Open LM Studio, go to Local Server, and click Start. "
        "Match the port with LMSTUDIO_BASE_URL in your .env file."
    )


def openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)


def settings_from_env_overrides(
    lm_url: str | None,
    api_key: str | None,
    model_id: str | None,
) -> tuple[str, str, str]:
    base = (
        lm_url or os.environ.get("LMSTUDIO_BASE_URL") or "http://127.0.0.1:1234/v1"
    ).rstrip("/")
    key = api_key or os.environ.get("LMSTUDIO_API_KEY") or "lm-studio"
    mid = (
        (model_id or os.environ.get("LMSTUDIO_MODEL") or "").strip()
        or DEFAULT_LMSTUDIO_CHAT_MODEL
    )
    return base, key, mid
