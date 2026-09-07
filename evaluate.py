"""Evaluacion del retrieval sobre un set de preguntas etiquetadas.

Sin medicion no hay mejora posible: este script fija una linea base
reproducible (recall@k y MRR) para poder comparar cambios de chunking,
de dimension o de backend de embeddings.
"""
from __future__ import annotations

import json
import statistics
import sys

from rag import RagPipeline

# (pregunta, doc_id esperado)
GOLDEN_SET: list[tuple[str, str]] = [
    ("cual es el limite de transferencia diario para personas fisicas", "politica_transferencias"),
    ("hasta que hora se procesan las transferencias interbancarias", "politica_transferencias"),
    ("en cuantos dias puedo pedir la reversa de una transferencia", "politica_transferencias"),
    ("que documentos necesita una persona moral para abrir cuenta", "onboarding_digital"),
    ("cuantas veces puedo reintentar si me rechazan por calidad de imagen", "onboarding_digital"),
    ("cuales son las etapas del alta remota de clientes", "onboarding_digital"),
    ("a partir de que puntaje se bloquea una transaccion", "prevencion_fraude"),
    ("cual es la tasa de falsos positivos tolerada del modelo", "prevencion_fraude"),
    ("cada cuanto se reentrena el modelo de riesgo", "prevencion_fraude"),
    ("que controles de calidad debe tener un pipeline de datos", "arquitectura_datos"),
    ("cuanto tiempo se conservan los datos de la capa raw", "arquitectura_datos"),
    ("cuando hay que enmascarar informacion personal", "arquitectura_datos"),
    ("cual es el tiempo de respuesta para una incidencia de severidad 1", "soporte_incidencias"),
    ("cuando se requiere un analisis post mortem", "soporte_incidencias"),
    ("como se clasifican las incidencias por severidad", "soporte_incidencias"),
]


def evaluate(top_k: int = 3) -> dict:
    pipeline = RagPipeline()
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    failures: list[dict] = []

    for question, expected_doc in GOLDEN_SET:
        results = pipeline.retrieve(question, top_k=top_k)
        docs = [h.doc_id for h in results]
        if expected_doc in docs:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (docs.index(expected_doc) + 1))
        else:
            reciprocal_ranks.append(0.0)
            failures.append(
                {"question": question, "expected": expected_doc, "retrieved": docs[:top_k]}
            )

    total = len(GOLDEN_SET)
    return {
        "questions": total,
        "top_k": top_k,
        f"recall@{top_k}": round(hits_at_k / total, 3),
        "mrr": round(statistics.mean(reciprocal_ranks), 3),
        "failures": failures,
    }


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(json.dumps(evaluate(k), indent=2, ensure_ascii=False))
