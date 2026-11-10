"""Vide la base vectorielle pour pouvoir recharger l'histoire (ou changer de modèle d'embeddings).

Usage (depuis la racine du projet) :
    python -m cli.clean
"""

from app import config, retrieval


def main() -> None:
    client = retrieval.get_chroma_client()
    # Selon la version de Chroma, list_collections() renvoie des objets ou des noms
    existing = [getattr(c, "name", c) for c in client.list_collections()]
    if config.COLLECTION_NAME not in existing:
        print(f"Rien à nettoyer : la collection '{config.COLLECTION_NAME}' n'existe pas.")
        return
    count = client.get_collection(config.COLLECTION_NAME).count()
    client.delete_collection(config.COLLECTION_NAME)
    print(f"Collection '{config.COLLECTION_NAME}' supprimée ({count} morceaux).")


if __name__ == "__main__":
    main()
