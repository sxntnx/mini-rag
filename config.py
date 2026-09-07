"""Configuracion central del proyecto. Todo se puede sobreescribir por variables de entorno."""
import os

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 900))        # caracteres por chunk
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 150))  # solape entre chunks

# --- Embeddings ---
# "local"  -> TF-IDF + SVD (sin API key, reproducible, para correr el demo)
# "openai" -> text-embedding-3-small (calidad de produccion, requiere OPENAI_API_KEY)
EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "local")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", 256))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("RAG_OPENAI_MODEL", "text-embedding-3-small")

# --- Store ---
DB_PATH = os.getenv("RAG_DB_PATH", "rag_store.db")

# --- Retrieval ---
TOP_K = int(os.getenv("RAG_TOP_K", 4))
# Piso de ruido: descarta chunks irrelevantes del top-k.
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", 0.10))
# Umbral de abstencion: se aplica SOLO al mejor resultado. Si ni el chunk mas
# parecido llega a este valor, la pregunta esta fuera del corpus y el sistema
# responde que no sabe. Calibrado sobre el golden set con el backend local
# (preguntas dentro del corpus: >=0.93; fuera del corpus: <=0.67).
ABSTAIN_THRESHOLD = float(os.getenv("RAG_ABSTAIN_THRESHOLD", 0.80))
