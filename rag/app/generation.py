"""Génération de la réponse : prompt système, contexte, appel du LLM (normal ou en streaming)."""

from __future__ import annotations

from collections.abc import Iterator

from . import config
from .retrieval import OllamaUnavailableError, get_ollama_client

SYSTEM_PROMPT = """Tu es un conteur pour enfants qui répond aux questions sur l'histoire de Léo et Éclair.

Règles :
- Réponds UNIQUEMENT à partir des extraits fournis ci-dessous
- Si la réponse n'est pas dans les extraits, dis "Je ne trouve pas cette information dans l'histoire"
- Utilise un langage simple adapté aux enfants
- Cite le chapitre quand c'est pertinent
- Réponds en français, en quelques phrases maximum
"""


def build_context(results: list[tuple[str, dict, float]]) -> str:
    """Assemble les passages retrouvés, chacun précédé de son chapitre.

    On utilise le titre complet ("Chapitre 1 : Le Grand Départ") plutôt que
    chapter_index, qui commence à 0 et ne correspond pas au numéro affiché.
    """
    return "\n\n".join(f"[{meta['chapter']}] {doc}" for doc, meta, score in results)


def build_messages(question: str, results: list[tuple[str, dict, float]]) -> list[dict]:
    context = build_context(results) or "(aucun extrait pertinent trouvé)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extraits de l'histoire :\n\n{context}\n\nQuestion : {question}"},
    ]


def chat(messages: list[dict]) -> str:
    """Appelle le LLM et renvoie la réponse complète."""
    try:
        response = get_ollama_client().chat(
            model=config.LLM_MODEL,
            messages=messages,
            options={"temperature": 0.2},
        )
    except Exception as exc:  # Ollama injoignable, modèle absent...
        raise OllamaUnavailableError(f"Erreur Ollama (LLM '{config.LLM_MODEL}') : {exc}") from exc
    return response["message"]["content"].strip()


def chat_stream(messages: list[dict]) -> Iterator[str]:
    """Appelle le LLM en streaming et renvoie les morceaux de texte au fil de l'eau.

    Le premier morceau est demandé tout de suite : si Ollama est arrêté,
    l'erreur est levée avant le début de la réponse HTTP (et peut donc devenir une 502).
    """
    try:
        stream = get_ollama_client().chat(
            model=config.LLM_MODEL,
            messages=messages,
            options={"temperature": 0.2},
            stream=True,
        )
        first = next(stream, None)
    except Exception as exc:
        raise OllamaUnavailableError(f"Erreur Ollama (LLM '{config.LLM_MODEL}') : {exc}") from exc

    def _tokens() -> Iterator[str]:
        if first is not None:
            yield first["message"]["content"]
        for part in stream:
            yield part["message"]["content"]

    return _tokens()
