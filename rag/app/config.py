"""Configuration partagée par l'API et les scripts CLI.

Chaque réglage peut être surchargé par une variable d'environnement
(voir `.env.example` et le README).
"""

import os
from pathlib import Path

# Racine du projet (le dossier qui contient app/, cli/ et data/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Base vectorielle (ChromaDB persistante sur disque)
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "story")

# Source et découpage
STORY_PATH = Path(os.getenv("STORY_PATH", BASE_DIR / "data" / "story.txt"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # en caractères
CHUNK_OVERLAP_PARAGRAPHS = int(os.getenv("CHUNK_OVERLAP_PARAGRAPHS", "1"))

# Recherche
TOP_K = int(os.getenv("TOP_K", "4"))
# Seuil de pertinence : les passages dont la similarité cosinus est en dessous sont ignorés
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.3"))

# Cache des réponses de /ask
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # en secondes (5 min)
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "100"))  # nombre de réponses gardées
