"""Vector store sobre SQLite.

Guarda chunks, metadatos y vectores (float32 en BLOB). La busqueda es coseno
exacto sobre la matriz completa: con decenas de miles de chunks es instantaneo
y evita agregar una dependencia de infraestructura al demo. El indice ANN
(pgvector / FAISS) es el paso siguiente si el corpus crece.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    n_chunks   INTEGER NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    n_chars     INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    UNIQUE (doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass
class Hit:
    doc_id: str
    chunk_index: int
    text: str
    score: float


class VectorStore:
    def __init__(self, path: str):
        self.path = path
        # check_same_thread=False + lock: uvicorn atiende cada request en un
        # thread distinto del pool, y sqlite3 rechaza por defecto una conexion
        # compartida entre threads.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- escritura ---
    def upsert_document(self, doc_id: str, source: str, texts: list[str], vectors: np.ndarray) -> int:
        if len(texts) != vectors.shape[0]:
            raise ValueError("texts y vectors deben tener el mismo largo")
        vectors = vectors.astype(np.float32)
        dim = int(vectors.shape[1])
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cur.execute(
                "INSERT INTO documents(doc_id, source, n_chunks) VALUES (?,?,?) "
                "ON CONFLICT(doc_id) DO UPDATE SET source=excluded.source, "
                "n_chunks=excluded.n_chunks",
                (doc_id, source, len(texts)),
            )
            cur.executemany(
                "INSERT INTO chunks(doc_id, chunk_index, text, n_chars, vector, dim) "
                "VALUES (?,?,?,?,?,?)",
                [(doc_id, i, t, len(t), vectors[i].tobytes(), dim) for i, t in enumerate(texts)],
            )
            self.conn.commit()
        return len(texts)

    def reset(self) -> None:
        with self._lock:
            self.conn.executescript("DELETE FROM chunks; DELETE FROM documents;")
            self.conn.commit()

    # --- lectura ---
    def all_texts(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT text FROM chunks ORDER BY id")]

    def stats(self) -> dict:
        with self._lock:
            docs, chunks, chars = self.conn.execute(
                "SELECT (SELECT COUNT(*) FROM documents), COUNT(*), "
                "COALESCE(SUM(n_chars),0) FROM chunks"
            ).fetchone()
        return {
            "documents": docs,
            "chunks": chunks,
            "avg_chunk_chars": round(chars / chunks, 1) if chunks else 0,
        }

    def _matrix(self) -> tuple[np.ndarray, list[tuple[str, int, str]]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT doc_id, chunk_index, text, vector, dim FROM chunks ORDER BY id"
            ).fetchall()
        if not rows:
            return np.zeros((0, 1), dtype=np.float32), []
        dim = rows[0][4]
        matrix = np.vstack([np.frombuffer(r[3], dtype=np.float32).reshape(1, dim) for r in rows])
        meta = [(r[0], r[1], r[2]) for r in rows]
        return matrix, meta

    def search(self, query_vector: np.ndarray, top_k: int, min_score: float = 0.0) -> list[Hit]:
        matrix, meta = self._matrix()
        if matrix.shape[0] == 0:
            return []
        scores = matrix @ query_vector.astype(np.float32).ravel()  # vectores ya normalizados
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(doc_id=meta[i][0], chunk_index=meta[i][1], text=meta[i][2], score=float(scores[i]))
            for i in order
            if scores[i] >= min_score
        ]

    def close(self) -> None:
        self.conn.close()
