"""Capa de embeddings intercambiable.

El resto del sistema (store, retrieval, API) no sabe que backend se usa: solo
pide vectores normalizados. Eso permite correr el demo sin API key y cambiar a
un modelo de produccion con una variable de entorno.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

import config


class Embedder(ABC):
    dim: int

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Ajusta el backend al corpus (no-op en backends pre-entrenados)."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Devuelve una matriz (n, dim) de vectores L2-normalizados."""


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalEmbedder(Embedder):
    """TF-IDF + SVD (LSA). Sin dependencias externas ni API key.

    No es un modelo de lenguaje: captura co-ocurrencia, no semantica profunda.
    Sirve para que el pipeline sea reproducible de punta a punta y para medir
    el retrieval de forma deterministica.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.dim = dim
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._svd_cls = TruncatedSVD
        self._svd = None

    def fit(self, corpus: list[str]) -> None:
        tfidf = self._vectorizer.fit_transform(corpus)
        n_components = min(self.dim, max(2, min(tfidf.shape) - 1))
        self._svd = self._svd_cls(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)
        self.dim = n_components

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("El embedder local requiere fit() antes de encode()")
        return l2_normalize(self._svd.transform(self._vectorizer.transform(texts)))

    def save(self, path: Path) -> None:
        import pickle

        path.write_bytes(pickle.dumps({"vec": self._vectorizer, "svd": self._svd, "dim": self.dim}))

    @classmethod
    def load(cls, path: Path) -> "LocalEmbedder":
        import pickle

        state = pickle.loads(path.read_bytes())
        obj = cls(dim=state["dim"])
        obj._vectorizer = state["vec"]
        obj._svd = state["svd"]
        obj.dim = state["dim"]
        return obj


class OpenAIEmbedder(Embedder):
    """text-embedding-3-small. Requiere OPENAI_API_KEY. Batching + reintentos."""

    def __init__(self, model: str = config.OPENAI_EMBEDDING_MODEL, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self.dim = 1536
        if not config.OPENAI_API_KEY:
            raise RuntimeError("Falta OPENAI_API_KEY para el backend 'openai'")

    def fit(self, corpus: list[str]) -> None:  # modelo pre-entrenado
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()["data"]
            vectors.extend(item["embedding"] for item in sorted(payload, key=lambda d: d["index"]))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def get_embedder(backend: str | None = None) -> Embedder:
    backend = backend or config.EMBEDDING_BACKEND
    if backend == "local":
        return LocalEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Backend desconocido: {backend}")
