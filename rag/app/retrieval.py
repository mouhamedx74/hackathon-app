"""Recherche vectorielle : client Ollama (embeddings), collection Chroma, filtrage par score.

Ce module ne dépend pas de FastAPI : il lève des exceptions métier
que l'API traduit en codes HTTP (502, 503...).
"""

from __future__ import annotations

import chromadb
import ollama

from . import config


# --- Erreurs métier (traduites en codes HTTP par api.py) -------------------

class OllamaUnavailableError(Exception):
    """Ollama est injoignable ou le modèle demandé est absent (-> 502)."""


class EmptyCollectionError(Exception):
    """La base vectorielle ne contient aucun morceau (-> 503)."""


class EmbedModelMismatchError(Exception):
    """La collection a été indexée avec un autre modèle d'embeddings (-> 409)."""


# --- Clients partagés (un seul par processus, pas un par requête) ----------

_ollama = ollama.Client(host=config.OLLAMA_HOST)
_chroma: chromadb.ClientAPI | None = None


def get_ollama_client() -> ollama.Client:
    return _ollama


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma


def get_collection():
    """Renvoie la collection (créée si besoin) en vérifiant le modèle d'embeddings.

    Des vecteurs calculés par deux modèles différents ne sont pas comparables :
    si la collection a été remplie avec un autre modèle que EMBED_MODEL,
    on lève une erreur claire plutôt que de renvoyer des résultats absurdes.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except Exception:  # la collection n'existe pas encore (le type d'erreur varie selon les versions)
        # Distance cosinus : adaptée aux embeddings de texte
        return client.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL},
        )

    stored_model = (collection.metadata or {}).get("embed_model")
    if stored_model and stored_model != config.EMBED_MODEL:
        raise EmbedModelMismatchError(
            f"La collection '{config.COLLECTION_NAME}' a été indexée avec '{stored_model}' "
            f"mais EMBED_MODEL vaut '{config.EMBED_MODEL}'. "
            "Lance `python -m cli.clean` puis `python -m cli.ingest`."
        )
    return collection


# --- Embeddings -------------------------------------------------------------

def embed(texts: list[str], prefix: str = "") -> list[list[float]]:
    """Calcule les embeddings via Ollama.

    `prefix` sert aux modèles qui distinguent requête et document
    (ex. nomic-embed-text : "search_query: " / "search_document: ").
    """
    inputs = [prefix + t for t in texts]
    try:
        return _ollama.embed(model=config.EMBED_MODEL, input=inputs)["embeddings"]
    except Exception as exc:  # Ollama arrêté, modèle non téléchargé...
        raise OllamaUnavailableError(f"Erreur Ollama (embeddings '{config.EMBED_MODEL}') : {exc}") from exc


def _uses_nomic_prefixes() -> bool:
    return "nomic" in config.EMBED_MODEL


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embed(texts, "search_document: " if _uses_nomic_prefixes() else "")


def embed_query(text: str) -> list[float]:
    return embed([text], "search_query: " if _uses_nomic_prefixes() else "")[0]


# --- Recherche --------------------------------------------------------------

def retrieve(question: str, top_k: int = config.TOP_K, min_score: float = config.MIN_SCORE
             ) -> list[tuple[str, dict, float]]:
    """Renvoie les passages pertinents sous forme de tuples (texte, métadonnées, score).

    Le score est la similarité cosinus (1 - distance) : seuls les passages
    dont le score est >= min_score sont gardés.
    """
    collection = get_collection()
    count = collection.count()  # un seul appel
    if count == 0:
        raise EmptyCollectionError("La base est vide : lance d'abord `python -m cli.ingest`.")

    query_embedding = embed_query(question)  # lève OllamaUnavailableError si Ollama est arrêté
    result = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, count))

    results = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        score = round(1 - dist, 4)
        if score >= min_score:
            results.append((doc, meta, score))
    return results
