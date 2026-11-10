"""Charge l'histoire dans la base vectorielle.

Usage (depuis la racine du projet) :
    python -m cli.ingest                      # charge data/story.txt
    python -m cli.ingest autre_histoire.txt   # charge un autre fichier
"""

import sys
from pathlib import Path

from app import config
from app.ingestion import ingest
from app.retrieval import EmbedModelMismatchError, OllamaUnavailableError


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.STORY_PATH
    try:
        ingest(path)
    except (OllamaUnavailableError, EmbedModelMismatchError) as exc:
        sys.exit(f"Échec de l'ingestion : {exc}")


if __name__ == "__main__":
    main()
