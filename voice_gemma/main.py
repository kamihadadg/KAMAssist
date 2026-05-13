"""
Voice → Whisper (local) → LM Studio (Gemma) → edge-tts (free neural voices).

Requires: LM Studio local server running (OpenAI-compatible API).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import edge_tts

from core import (
    DEFAULT_LMSTUDIO_CHAT_MODEL,
    ask_lm_studio,
    edge_voice_for_whisper_lang,
    load_local_env,
    make_whisper_model,
    merge_user_message,
    openai_client,
    resolve_whisper_device,
    transcribe_ndarray,
)
from openai import APIConnectionError

try:
    from playsound import playsound
except ImportError:  # pragma: no cover
    playsound = None


def record_wav_float32(
    seconds: float, sample_rate: int, out_wav_path: Path | None = None
) -> tuple[np.ndarray, int]:
    frames = int(seconds * sample_rate)
    print(f"Recording — speak for up to {seconds:.0f} seconds…")
    data = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    flat = np.squeeze(data, axis=1)
    if out_wav_path:
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(flat, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(out_wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_i16.tobytes())
        print(f"Saved sample WAV: {out_wav_path}")
    return flat, sample_rate


def _lm_studio_refused_help(base_url: str) -> None:
    print()
    print("Could not connect to LM Studio (connection refused).")
    print(f"  URL used: {base_url}")
    print("  Checklist:")
    print("  1) Open LM Studio and start the Local Server (Developer tab).")
    print("  2) Confirm the port (default 1234). If different, set --lm-url or LMSTUDIO_BASE_URL, e.g.:")
    print('        python main.py --lm-url "http://127.0.0.1:PORT/v1" ...')
    print("  3) Ensure firewall/antivirus is not blocking localhost.")
    print("  4) Copy the exact URL/port from LM Studio after Start.")
    print(
        "     PowerShell: Test-NetConnection 127.0.0.1 -Port 1234   (TcpTestSucceeded should be True)"
    )
    print()


async def speak_edge_tts(
    text: str,
    voice: str,
    *,
    play: bool = True,
    reply_mp3: Path | None = None,
) -> None:
    if not text:
        return
    if play and playsound is None:
        raise RuntimeError(
            "playsound is required for audio playback. pip install playsound==1.2.2"
        )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    try:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(path)
        if reply_mp3:
            reply_mp3 = reply_mp3.resolve()
            reply_mp3.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, reply_mp3)
            print(f"Saved reply audio: {reply_mp3}")
        if play:
            playsound(os.path.abspath(path))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice + text chat with LM Studio, Whisper, and edge-tts")
    p.add_argument(
        "--lm-url",
        default=os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        help="OpenAI-compatible API base URL (LM Studio)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
        help="API key (LM Studio often accepts any string)",
    )
    p.add_argument(
        "--chat-model",
        default=os.environ.get("LMSTUDIO_MODEL", DEFAULT_LMSTUDIO_CHAT_MODEL),
        help=f"Model id in LM Studio (default: {DEFAULT_LMSTUDIO_CHAT_MODEL})",
    )
    p.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model size: tiny, base, small, medium, large-v2, large-v3",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, or auto",
    )
    p.add_argument(
        "--compute-type",
        default="int8",
        help="Usually int8 on CPU; float16 on GPU",
    )
    p.add_argument(
        "--language",
        default=None,
        help="STT language code (e.g. fa, en); omit for auto-detect",
    )
    p.add_argument(
        "--tts-voice",
        default=None,
        help="edge-tts voice (e.g. fa-IR-DilaraNeural); ignored if derived from transcript",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=12.0,
        help="Max recording length per turn (seconds)",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Mic sample rate (16 kHz is standard for Whisper)",
    )
    p.add_argument(
        "--save-wav",
        default=None,
        help="Optional path to save each recording as WAV (debug)",
    )
    p.add_argument(
        "--system",
        default="You are a helpful, concise multilingual assistant. Answer in the same language as the user unless they ask otherwise.",
        help="System prompt for chat",
    )
    p.add_argument(
        "--text",
        "-t",
        default=None,
        help="Optional typed text; combined with mic unless --no-mic",
    )
    p.add_argument(
        "--no-mic",
        action="store_true",
        help="Skip microphone; text only (Whisper not loaded)",
    )
    p.add_argument(
        "--reply-mp3",
        default=None,
        help="Save TTS reply as MP3",
    )
    p.add_argument(
        "--no-tts",
        action="store_true",
        help="Do not play TTS; still print text. With --reply-mp3, only save file",
    )
    return p.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()
    chat_model = (args.chat_model or "").strip() or DEFAULT_LMSTUDIO_CHAT_MODEL

    typed = (args.text or "").strip()
    if args.no_mic and not typed:
        raise SystemExit("--no-mic requires --text / -t.")

    detected_lang: str | None = None
    voice_text = ""

    if not args.no_mic:
        print("Loading Whisper… (first run may download weights)")
        device = resolve_whisper_device(args.device)
        whisper = make_whisper_model(args.whisper_model, device, args.compute_type)

        wav_path = Path(args.save_wav) if args.save_wav else None
        audio, sr = record_wav_float32(args.seconds, args.sample_rate, wav_path)
        voice_text, detected_lang = transcribe_ndarray(whisper, audio, sr, args.language)
        if not voice_text.strip() and not typed:
            print("No speech recognized and no typed text. Try again.")
            return
        if voice_text.strip():
            print(f"[Detected language: {detected_lang or args.language or 'auto'}]")
            print(f"You (voice): {voice_text}")

    if typed:
        print(f"You (text): {typed}")

    user_message = merge_user_message(typed or None, voice_text or None)

    client = openai_client(args.lm_url, args.api_key)
    base = args.lm_url.rstrip("/")
    print(f"LM Studio model id: {chat_model}")
    try:
        reply = ask_lm_studio(client, chat_model, user_message, args.system)
    except APIConnectionError:
        _lm_studio_refused_help(base)
        raise SystemExit(1) from None
    print(f"Assistant (text): {reply}")

    tts_lang = detected_lang or args.language
    voice = args.tts_voice or edge_voice_for_whisper_lang(tts_lang)
    print(f"TTS voice: {voice}")

    reply_mp3 = Path(args.reply_mp3) if args.reply_mp3 else None
    play_audio = not args.no_tts
    if play_audio or reply_mp3:
        asyncio.run(
            speak_edge_tts(reply, voice, play=play_audio, reply_mp3=reply_mp3)
        )


if __name__ == "__main__":
    main()
