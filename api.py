"""API REST con FastAPI. Levantar con: uvicorn api:app --reload"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from rag import RagPipeline

app = FastAPI(
    title="mini-RAG API",
    version="1.0.0",
    description="Busqueda semantica sobre documentos con respuestas citadas.",
)

pipeline = RagPipeline()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Cual es el limite de transferencia diario?"])
    top_k: int = Field(config.TOP_K, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    sources: list[Source]
    context_chars: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "embedding_backend": config.EMBEDDING_BACKEND}


@app.get("/stats")
def stats() -> dict:
    return pipeline.stats()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    try:
        return pipeline.answer(req.question, top_k=req.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/ingest")
def ingest(folder: str = "sample_docs") -> dict:
    path = Path(folder)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"No existe la carpeta {folder}")
    try:
        return pipeline.ingest_folder(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
