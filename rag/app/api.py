"""API de questions/réponses sur l'histoire (RAG + Ollama).

Lancement (depuis la racine du projet) :
    uvicorn app.api:app --reload
Swagger : http://localhost:8000/docs

Codes d'erreur :
    400 : question invalide (vide)
    409 : la base a été indexée avec un autre modèle d'embeddings
    502 : Ollama injoignable ou modèle absent
    503 : base vide (ingestion pas encore faite)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config, generation, retrieval
from .cache import cache

app = FastAPI(
    title="Story RAG API",
    description="Pose des questions sur « L'Aventure magique de Léo et Éclair ». "
                "Les réponses sont générées par un LLM Ollama à partir des passages retrouvés.",
    version="2.0.0",
)

# CORS ouvert : l'API est pédagogique et tourne en local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
def invalid_input(request: Request, exc: RequestValidationError):
    """Corps de requête invalide (question vide, top_k hors limites...) -> 400 au lieu de 422."""
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


# --- Modèles Pydantic -------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Qui est Éclair dans l'histoire ?"])
    top_k: int = Field(config.TOP_K, ge=1, le=20, description="Nombre de passages à retrouver")


class Source(BaseModel):
    chapter: str
    text: str
    score: float = Field(description="Similarité cosinus (1 = identique)")


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    sources: list[Source]
    cached: bool = Field(False, description="true si la réponse vient du cache")


# --- Fonctions utilitaires --------------------------------------------------

def _clean_question(question: str) -> str:
    """Refuse les questions composées uniquement d'espaces (400)."""
    question = question.strip()
    if not question:
        raise HTTPException(400, "La question ne peut pas être vide.")
    return question


def _retrieve(question: str, top_k: int) -> list[tuple[str, dict, float]]:
    """Recherche vectorielle avec traduction des erreurs métier en codes HTTP."""
    try:
        return retrieval.retrieve(question, top_k)
    except retrieval.EmptyCollectionError as exc:
        raise HTTPException(503, str(exc)) from exc
    except retrieval.OllamaUnavailableError as exc:
        raise HTTPException(502, str(exc)) from exc
    except retrieval.EmbedModelMismatchError as exc:
        raise HTTPException(409, str(exc)) from exc


def _to_sources(results: list[tuple[str, dict, float]]) -> list[Source]:
    return [Source(chapter=meta["chapter"], text=doc, score=score) for doc, meta, score in results]


# --- Routes -----------------------------------------------------------------

@app.get("/health", tags=["système"])
def health():
    """Vérifie que la base est chargée et indique les modèles utilisés."""
    try:
        chunks = retrieval.get_collection().count()
    except retrieval.EmbedModelMismatchError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "chunks": chunks,
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "min_score": config.MIN_SCORE,
    }


@app.post("/search", response_model=list[Source], tags=["histoire"])
def search(req: AskRequest):
    """Retourne seulement les passages pertinents (sans appeler le LLM)."""
    return _to_sources(_retrieve(_clean_question(req.question), req.top_k))


@app.post("/ask", response_model=AskResponse, tags=["histoire"])
def ask(req: AskRequest):
    """Répond à une question sur l'histoire à partir des passages retrouvés.

    Les réponses sont gardées en cache (CACHE_TTL secondes) : une question
    déjà posée est renvoyée directement, avec `cached: true`.
    """
    question = _clean_question(req.question)
    cache_key = f"{req.top_k}|{question}"  # top_k fait partie de la clé : il change les sources

    # 1. Question déjà posée récemment ?
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.model_copy(update={"cached": True})

    # 2. Pipeline normal : recherche + génération
    results = _retrieve(question, req.top_k)
    try:
        answer = generation.chat(generation.build_messages(question, results))
    except retrieval.OllamaUnavailableError as exc:
        raise HTTPException(502, str(exc)) from exc

    response = AskResponse(
        question=question, answer=answer, model=config.LLM_MODEL,
        sources=_to_sources(results), cached=False,
    )
    # 3. Mise en cache avant de renvoyer
    cache.set(cache_key, response)
    return response


@app.post("/ask/stream", tags=["histoire"])
def ask_stream(req: AskRequest):
    """Comme /ask, mais renvoie la réponse en texte brut au fil de la génération (sans cache)."""
    question = _clean_question(req.question)
    results = _retrieve(question, req.top_k)
    try:
        tokens = generation.chat_stream(generation.build_messages(question, results))
    except retrieval.OllamaUnavailableError as exc:
        raise HTTPException(502, str(exc)) from exc
    return StreamingResponse(tokens, media_type="text/plain; charset=utf-8")


@app.get("/cache/stats", tags=["cache"])
def cache_stats():
    """Nombre d'entrées en cache (total et encore valides) et durée de vie."""
    return cache.stats()


@app.delete("/cache", tags=["cache"])
def cache_clear():
    """Vide le cache des réponses."""
    return {"cleared": cache.clear()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
