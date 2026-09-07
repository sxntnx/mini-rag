"""Tests minimos del pipeline. Correr con: pytest -q"""
from pathlib import Path

import numpy as np
import pytest

import chunking
from embeddings import LocalEmbedder, l2_normalize
from store import VectorStore

TEXT = "\n\n".join(f"Parrafo {i} sobre control de calidad de datos y validacion." for i in range(30))


def test_chunking_respeta_tamano_y_solape():
    chunks = chunking.split_text(TEXT, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # margen por el solape agregado


def test_chunking_rechaza_overlap_invalido():
    with pytest.raises(ValueError):
        chunking.split_text(TEXT, chunk_size=100, overlap=100)


def test_vectores_normalizados():
    matrix = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0)
    assert np.isclose(np.linalg.norm(matrix[1]), 0.0)  # el vector nulo no rompe


def test_store_guarda_y_recupera(tmp_path: Path):
    store = VectorStore(str(tmp_path / "t.db"))
    textos = ["limite de transferencia diario", "clasificacion de incidencias"]
    embedder = LocalEmbedder(dim=2)
    embedder.fit(textos)
    store.upsert_document("doc", "doc.md", textos, embedder.encode(textos))
    assert store.stats()["chunks"] == 2
    hits = store.search(embedder.encode(["limite de transferencia"])[0], top_k=1)
    assert hits and hits[0].text == textos[0]
    store.close()


def test_reingesta_no_duplica(tmp_path: Path):
    store = VectorStore(str(tmp_path / "t.db"))
    textos = ["uno", "dos"]
    embedder = LocalEmbedder(dim=2)
    embedder.fit(textos)
    vectors = embedder.encode(textos)
    store.upsert_document("doc", "doc.md", textos, vectors)
    store.upsert_document("doc", "doc.md", textos, vectors)
    assert store.stats()["chunks"] == 2
    store.close()
