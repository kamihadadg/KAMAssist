"""Extract plain text from uploaded files for chat context."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import UploadFile

MAX_TOTAL_CHARS = 120_000
MAX_PER_FILE_CHARS = 45_000

TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".py",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".log",
        ".sql",
        ".sh",
        ".bat",
        ".ps1",
    }
)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 40] + "\n... [truncated] ...\n"


def _pdf_to_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF: install pypdf — pip install pypdf]"
    reader = PdfReader(io.BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)


async def attachment_blocks_from_uploads(uploads: list[UploadFile]) -> str:
    """Return a single string of labeled sections for the LLM, or empty."""
    if not uploads:
        return ""
    sections: list[str] = []
    total = 0
    for up in uploads:
        name = up.filename or "attachment"
        suffix = Path(name).suffix.lower()
        try:
            raw = await up.read()
        except Exception as e:  # pragma: no cover
            sections.append(f"--- {name} ---\n[read error: {e}]")
            continue
        if not raw:
            continue
        try:
            if suffix == ".pdf":
                text = _pdf_to_text(raw)
            elif suffix in TEXT_SUFFIXES or suffix == "":
                text = raw.decode("utf-8", errors="replace")
            else:
                text = raw.decode("utf-8", errors="strict")
        except Exception:
            sections.append(
                f"--- {name} ---\n[binary or unsupported encoding; try .txt / .md / .pdf]"
            )
            continue
        text = _truncate(text.strip(), MAX_PER_FILE_CHARS)
        block = f"--- {name} ---\n{text}"
        if total + len(block) > MAX_TOTAL_CHARS:
            sections.append(
                f"--- {name} ---\n[skipped: total attachment budget {MAX_TOTAL_CHARS} chars exceeded]"
            )
            break
        sections.append(block)
        total += len(block)
    return "\n\n".join(sections)
