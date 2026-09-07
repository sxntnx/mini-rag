"""Extraccion de texto y chunking con solape.

El chunking es la decision que mas impacta la calidad del retrieval: chunks muy
grandes diluyen la senal, muy chicos pierden contexto. Se corta por parrafo y se
agrupa hasta CHUNK_SIZE, con solape para no partir una idea al medio.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str


def read_document(path: Path) -> str:
    """Lee .txt, .md o .pdf y devuelve texto plano normalizado."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Formato no soportado: {suffix}")
    return normalize(raw)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Agrupa parrafos hasta chunk_size; si un parrafo solo ya excede, lo parte por ventana."""
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_window(para, chunk_size, overlap))
            continue
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = _tail(buffer, overlap) + "\n\n" + para
    if buffer:
        chunks.append(buffer)
    return [c.strip() for c in chunks if c.strip()]


def _window(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def _tail(text: str, n: int) -> str:
    return text[-n:] if n and len(text) > n else text


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    return [Chunk(doc_id=doc_id, chunk_index=i, text=t) for i, t in enumerate(split_text(text))]
