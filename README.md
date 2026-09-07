# mini-RAG — Búsqueda semántica sobre documentos con respuestas citadas

Sistema RAG (Retrieval-Augmented Generation) de punta a punta: ingesta de documentos,
chunking con solape, embeddings, vector store, recuperación por similitud coseno y una
API REST que devuelve la respuesta **con sus fuentes** o se abstiene si no hay evidencia.

Construido sobre un corpus ficticio de documentación operativa de una fintech
(políticas de transferencias, onboarding digital, prevención de fraude, arquitectura
de datos y gestión de incidencias).

> **EN —** End-to-end RAG pipeline: document ingestion, overlap chunking, pluggable
> embeddings, SQLite vector store, cosine retrieval and a FastAPI service that answers
> with cited sources or abstains when the corpus has no evidence. Retrieval is measured
> against a labeled question set (recall@k / MRR), not assumed.

---

## Resultados medidos

Evaluación sobre un golden set de 15 preguntas etiquetadas (`evaluate.py`),
backend de embeddings local, 5 documentos / 10 chunks:

| Métrica | Valor |
|---|---|
| recall@1 | **0.933** |
| recall@3 | **1.000** |
| MRR | **0.967** |
| Latencia de ingesta | 1.1 s |
| Tests | 5/5 en verde |

**Separación de abstención:** las preguntas dentro del corpus recuperan con score ≥ 0.93;
las preguntas fuera del corpus no superan 0.67. El umbral quedó fijado en 0.80, con lo
que el sistema se abstiene en el 100% de las consultas fuera de dominio del set de prueba
sin perder ninguna respuesta válida.

Reproducir:

```bash
python make_sample_docs.py
python ingest.py
python evaluate.py 3
```

---

## Cómo funciona

```
documentos (.pdf/.md/.txt)
    │
    ├─ chunking.py     extracción + normalización + chunks de 900 car. con 150 de solape
    ├─ embeddings.py   backend intercambiable → vectores L2-normalizados
    ├─ store.py        SQLite: chunks + metadatos + vectores (BLOB float32)
    ├─ rag.py          retrieval por coseno, umbral de abstención, armado de contexto
    └─ api.py          FastAPI: /query /ingest /stats /health
```

### Decisiones de diseño

**Chunking por párrafo con solape.** Cortar por longitud fija parte ideas al medio y
degrada el retrieval. Se agrupan párrafos hasta el tamaño objetivo y se arrastra un
solape para que una definición que cae en el borde siga siendo recuperable.

**Embeddings desacoplados.** El store y el retrieval no saben qué modelo los generó.
Hay dos backends: `local` (TF-IDF + SVD, sin API key, determinístico — permite correr
el proyecto y medirlo sin costo) y `openai` (`text-embedding-3-small`, calidad de
producción). Se cambia con una variable de entorno, sin tocar el resto del código.

**Abstención antes que alucinación.** El umbral se aplica al mejor resultado, no a todo
el top-k: si ni el chunk más parecido llega a 0.80, la pregunta está fuera del corpus y
la API responde `grounded: false`. Los chunks de respaldo con menor score sí entran al
contexto, porque aportan detalle aunque no sean el match principal. En un dominio
regulado, un "no sé" auditable vale más que una respuesta plausible e incorrecta.

**Búsqueda exacta, no aproximada.** Con este volumen, el coseno sobre la matriz completa
es instantáneo y evita sumar una dependencia de infraestructura. El índice ANN
(pgvector o FAISS) es el paso siguiente cuando el corpus lo justifique — no antes.

**Cada respuesta cita su fuente.** La API devuelve `doc_id`, `chunk_index` y score de
cada pasaje usado, para que la respuesta sea verificable contra el documento original.

---

## Uso

```bash
pip install -r requirements.txt
python make_sample_docs.py     # genera el corpus de ejemplo
python ingest.py               # indexa → rag_store.db
uvicorn api:app --reload       # docs interactivos en http://127.0.0.1:8000/docs
```

Consulta:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "a partir de que puntaje se bloquea una transaccion", "top_k": 2}'
```

```json
{
  "question": "a partir de que puntaje se bloquea una transaccion",
  "grounded": true,
  "sources": [
    {"doc_id": "prevencion_fraude", "chunk_index": 0, "score": 0.9293},
    {"doc_id": "prevencion_fraude", "chunk_index": 1, "score": 0.4682}
  ]
}
```

Para indexar tus propios documentos: `python ingest.py ruta/a/tu/carpeta`

### Con LLM y embeddings de producción

```bash
export OPENAI_API_KEY=sk-...
export RAG_EMBEDDING_BACKEND=openai
python ingest.py && uvicorn api:app
```

Con la key configurada, `/query` genera una respuesta redactada y citada con `gpt-4o-mini`
sobre el contexto recuperado; sin key, devuelve el pasaje más relevante en modo extractivo.

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y backend activo |
| GET | `/stats` | Documentos, chunks y tamaño promedio de chunk |
| POST | `/ingest` | Reindexa una carpeta |
| POST | `/query` | Pregunta → respuesta + fuentes citadas |

## Tests

```bash
pytest -q
```

Cubren chunking (tamaño, solape y validación de parámetros), normalización de vectores,
persistencia y recuperación en el store, y reingesta idempotente.

## Stack

Python · FastAPI · Pydantic · scikit-learn · NumPy · SQLite · pypdf · pytest

## Roadmap

- [ ] Índice ANN con pgvector para corpus de mayor volumen
- [ ] Búsqueda híbrida (BM25 + denso) con reranking
- [ ] Evaluación de la generación, no solo del retrieval (faithfulness)
- [ ] Contenedor Docker y despliegue

---

**Santino Spelzini** — Ingeniero Industrial · Data Analyst
[LinkedIn](https://www.linkedin.com/in/santino-spelzini) · [GitHub](https://github.com/sxntnx)
