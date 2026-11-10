"""Liste le contenu de la base vectorielle.

Usage (depuis la racine du projet) :
    python -m cli.list_docs          # aperçu de chaque morceau
    python -m cli.list_docs --full   # texte complet de chaque morceau
"""

import argparse

from app import config, retrieval

PREVIEW_CHARS = 120


def main() -> None:
    parser = argparse.ArgumentParser(description="Liste les morceaux stockés dans la base.")
    parser.add_argument("--full", action="store_true", help="affiche le texte complet de chaque morceau")
    args = parser.parse_args()

    client = retrieval.get_chroma_client()
    # Selon la version de Chroma, list_collections() renvoie des objets ou des noms
    names = [getattr(c, "name", c) for c in client.list_collections()]
    if config.COLLECTION_NAME not in names:
        print(f"La collection '{config.COLLECTION_NAME}' n'existe pas : lance `python -m cli.ingest`.")
        return

    collection = client.get_collection(config.COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    rows = sorted(zip(data["ids"], data["documents"], data["metadatas"]), key=lambda r: r[0])

    print(f"Collection : {config.COLLECTION_NAME} ({config.CHROMA_DIR})")
    print(f"Modèle d'embeddings : {(collection.metadata or {}).get('embed_model', 'inconnu')}")
    print(f"Morceaux : {len(rows)}")

    current_chapter = None
    for id_, doc, meta in rows:
        if meta.get("chapter") != current_chapter:
            current_chapter = meta.get("chapter")
            print(f"\n=== {current_chapter} ===")
        if args.full:
            print(f"\n[{id_}] ({len(doc)} caractères)\n{doc}")
        else:
            preview = " ".join(doc.split())
            if len(preview) > PREVIEW_CHARS:
                preview = preview[:PREVIEW_CHARS] + "..."
            print(f"[{id_}] ({len(doc)} car.) {preview}")


if __name__ == "__main__":
    main()
