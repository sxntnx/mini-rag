"""Orquestacion del RAG: ingesta, retrieval y armado del contexto."""
from __future__ import annotations

import logging
from pathlib import Path

import chunking
import config
from embeddings import LocalEmbedder, get_embedder
from store import Hit, VectorStore

logger = logging.getLogger("rag")

MODEL_PATH = Path("embedder_local.pkl")
SUPPORTED = {".txt", ".md", ".pdf"}


class RagPipeline:
    def __init__(self, db_path: str = config.DB_PATH, backend: str | None = None):
        self.backend = backend or config.EMBEDDING_BACKEND
        self.store = VectorStore(db_path)
        self.embedder = None

    # ---------- ingesta ----------
    def ingest_folder(self, folder: Path, reset: bool = True) -> dict:
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)
        if not files:
            raise FileNotFoundError(f"No hay documentos soportados en {folder}")
        if reset:
            self.store.reset()

        staged: list[tuple[str, str, list[str]]] = []
        skipped: list[str] = []
        for path in files:
            try:
                text = chunking.read_document(path)
            except Exception as exc:  # un archivo roto no debe frenar la corrida
                logger.warning("Se omite %s: %s", path.name, exc)
                skipped.append(path.name)
                continue
            pieces = chunking.split_text(text)
            if not pieces:
                skipped.append(path.name)
                continue
            staged.append((path.stem, str(path), pieces))

        corpus = [t for _, _, pieces in staged for t in pieces]

        # El embedder local necesita ver todo el corpus antes de vectorizar.
        self.embedder = get_embedder(self.backend)
        self.embedder.fit(corpus)
        if isinstance(self.embedder, LocalEmbedder):
            self.embedder.save(MODEL_PATH)

        total = 0
        for doc_id, source, pieces in staged:
            vectors = self.embedder.encode(pieces)
            total += self.store.upsert_document(doc_id, source, pieces, vectors)

        stats = self.store.stats()
        stats.update({"files_read": len(staged), "files_skipped": skipped, "chunks_indexed": total})
        return stats

    # ---------- consulta ----------
    def _ensure_embedder(self):
        if self.embedder is not None:
            return
        if self.backend == "local":
            if not MODEL_PATH.exists():
                raise RuntimeError("No hay indice entrenado. Ejecuta: python ingest.py")
            self.embedder = LocalEmbedder.load(MODEL_PATH)
        else:
            self.embedder = get_embedder(self.backend)

    def retrieve(self, question: str, top_k: int = config.TOP_K) -> list[Hit]:
        self._ensure_embedder()
        vector = self.embedder.encode([question])[0]
        return self.store.search(vector, top_k=top_k, min_score=config.MIN_SCORE)

    def answer(self, question: str, top_k: int = config.TOP_K) -> dict:
        """Devuelve contexto + fuentes. La generacion con LLM se enchufa aca.

        Si ni el mejor chunk supera ABSTAIN_THRESHOLD, el sistema se abstiene en
        vez de alucinar: en dominios regulados un "no se" es mejor que una
        respuesta inventada. El umbral se aplica al mejor resultado y no a todo
        el top-k, porque los chunks de respaldo aportan contexto util aunque
        tengan menor similitud.
        """
        hits = self.retrieve(question, top_k=top_k)
        if hits and hits[0].score < config.ABSTAIN_THRESHOLD:
            hits = []
        if not hits:
            return {
                "question": question,
                "answer": "No encontre informacion suficiente en los documentos indexados.",
                "grounded": False,
                "sources": [],
            }
        context = "\n\n---\n\n".join(f"[{h.doc_id}#{h.chunk_index}] {h.text}" for h in hits)
        return {
            "question": question,
            "answer": self._generate(question, context),
            "grounded": True,
            "sources": [
                {"doc_id": h.doc_id, "chunk_index": h.chunk_index, "score": round(h.score, 4)}
                for h in hits
            ],
            "context_chars": len(context),
        }

    def _generate(self, question: str, context: str) -> str:
        """Punto de enchufe del LLM.

        Sin OPENAI_API_KEY devuelve el pasaje mas relevante (modo extractivo),
        para que el demo sea util y auditable sin costo de API.
        """
        if not config.OPENAI_API_KEY:
            return context.split("\n\n---\n\n")[0]
        import httpx

        prompt = (
            "Responde unicamente con la informacion del CONTEXTO. "
            "Si el contexto no alcanza, deci que no hay informacion suficiente. "
            "Cita las fuentes entre corchetes.\n\n"
            f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"
        )
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def stats(self) -> dict:
        return self.store.stats()
