from __future__ import annotations

import re
from pathlib import Path

from interview_coach.exceptions import DocumentError

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    output: list[str] = []
    blank = False
    for line in lines:
        if line:
            output.append(line)
            blank = False
        elif output and not blank:
            output.append("")
            blank = True
    return "\n".join(output).strip()


def parse_document(*, filename: str, content: bytes, max_bytes: int = 5 * 1024 * 1024) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"Unsupported document type: {extension or 'missing extension'}")
    if not content:
        raise DocumentError("Document is empty")
    if len(content) > max_bytes:
        raise DocumentError(f"Document exceeds the {max_bytes}-byte limit")

    if extension == ".txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")
    else:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise DocumentError("PDF support requires the 'documents' extra (PyMuPDF)") from exc
        try:
            pdf = fitz.open(stream=content, filetype="pdf")
            if pdf.needs_pass:
                raise DocumentError("Encrypted PDFs are not supported")
            text = "\n\n".join(page.get_text("text") for page in pdf)
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("Could not read the PDF") from exc

    normalized = normalize_text(text)
    if not normalized:
        raise DocumentError("Document contains no readable text")
    return normalized


def parse_pasted_text(text: str, label: str, max_bytes: int = 5 * 1024 * 1024) -> str:
    if len(text.encode("utf-8")) > max_bytes:
        raise DocumentError(f"{label} exceeds the {max_bytes}-byte limit")
    normalized = normalize_text(text)
    if not normalized:
        raise DocumentError(f"{label} is empty")
    return normalized
