# Story RAG API

Pose des questions sur une histoire pour enfants (`data/story.txt`, « L'Aventure magique
de Léo et Éclair », 6 chapitres) via une API FastAPI. Les passages pertinents sont
retrouvés dans une base vectorielle (ChromaDB), puis un LLM Ollama rédige la réponse.

## Lancement rapide

```bash
# Copier et remplir le fichier d'env
cp .env.example .env
# Mettre ton NGROK_AUTHTOKEN dans .env (token gratuit : https://dashboard.ngrok.com)

docker compose up --build
```

Au premier lancement, comptez ~2 min pour le téléchargement des modèles
(`nomic-embed-text` et `qwen2.5:1.5b`). L'histoire est ensuite indexée
automatiquement au démarrage de l'API.

- Swagger local : http://localhost:8000/docs
- Open WebUI : http://localhost:3000
- Dashboard ngrok : http://localhost:4040 (requêtes en temps réel)
- URL publique : visible sur http://localhost:4040 ou via `docker compose logs ngrok | grep url=`
- Ollama : http://localhost:11434

Sans `NGROK_AUTHTOKEN`, le service ngrok s'arrête avec un message et le reste de
la stack fonctionne normalement (en local uniquement).

### Exposition publique avec ngrok

ngrok rend l'API accessible depuis Internet sans configurer de redirection de port.
Sans `NGROK_DOMAIN`, l'URL est aléatoire et change à chaque redémarrage du service ;
avec un domaine réservé sur ngrok, elle reste fixe.

À savoir :
- **L'API n'a pas d'authentification** : toute personne qui a l'URL peut interroger
  le LLM (et vider le cache). N'exposez l'API que le temps d'une démo.
- Avec un compte gratuit, ngrok affiche une page d'avertissement aux navigateurs.
  Les appels `curl` ou depuis du code ne sont pas concernés ; pour un navigateur,
  ajoutez l'en-tête `ngrok-skip-browser-warning: 1`.
- Seule l'API est exposée : Ollama et Open WebUI restent locaux.

Les modèles, la base vectorielle et les données d'Open WebUI sont gardés dans des
volumes Docker : les lancements suivants sont immédiats. Pour tout effacer :
`docker compose down -v`.

## Services Docker

| Service       | Rôle |
|---------------|------|
| `ollama`      | Serveur de modèles (embeddings + LLM). |
| `ollama-pull` | Télécharge les modèles dans le volume d'Ollama, puis s'arrête. |
| `api`         | L'API FastAPI. ChromaDB tourne dans ce conteneur (mode embarqué, volume `chroma_data`). Lance l'ingestion puis l'API. |
| `open-webui`  | Interface de chat branchée directement sur Ollama. |
| `ngrok`       | Tunnel public vers l'API (dev). Dashboard sur le port 4040. |

## Structure du projet

```
app/
  config.py      tous les réglages (variables d'environnement)
  cache.py       cache en mémoire des réponses (dict + TTL)
  retrieval.py   embeddings Ollama, collection Chroma, recherche + filtrage par score
  generation.py  prompt système, contexte, appel du LLM (normal ou streaming)
  ingestion.py   découpage par chapitre en morceaux, embeddings, upsert
  api.py         routes FastAPI
cli/
  ingest.py      indexe l'histoire
  list_docs.py   affiche le contenu de la base
  clean.py       vide la base
data/
  story.txt      l'histoire (titre en 1re ligne, chapitres « Chapitre N : … » ou « I. … »)
```

## Commandes CLI

Dans le conteneur (`docker compose exec api …`) ou en local, depuis la racine du projet :

```bash
python -m cli.ingest              # indexe data/story.txt (relançable sans doublons)
python -m cli.ingest autre.txt    # indexe un autre fichier
python -m cli.list_docs           # contenu de la base (--full pour le texte complet)
python -m cli.clean               # vide la base
```

Après une modification de `data/story.txt` (monté dans le conteneur) :
`docker compose restart api` suffit, l'ingestion est relancée au démarrage.

## Endpoints

| Méthode | Route          | Rôle |
|---------|----------------|------|
| GET     | `/health`      | Nombre de morceaux en base, modèles utilisés, seuil de score |
| POST    | `/search`      | Passages retrouvés seulement (sans LLM), pour déboguer |
| POST    | `/ask`         | Réponse du LLM + passages sources. Les réponses sont mises en cache (`cached: true` si elle vient du cache) |
| POST    | `/ask/stream`  | Comme `/ask` mais la réponse arrive en texte brut au fil de la génération (sans cache) |
| GET     | `/cache/stats` | Entrées en cache (total, encore valides), TTL, taille max |
| DELETE  | `/cache`       | Vide le cache |

Corps de requête : `{"question": "Qui est Éclair dans l'histoire ?", "top_k": 4}`

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Qui est Éclair dans l'\''histoire ?"}'
```

Codes d'erreur :

| Code | Cas |
|------|-----|
| 400  | Requête invalide (question vide, `top_k` hors de 1–20) |
| 409  | La base a été indexée avec un autre modèle d'embeddings que `EMBED_MODEL` |
| 502  | Ollama injoignable ou modèle absent |
| 503  | Base vide : l'ingestion n'a pas encore été faite |

## Configuration (variables d'environnement)

Copiez `.env.example` en `.env` pour les surcharger : `docker compose` le lit automatiquement.

| Variable                   | Défaut                   | Rôle |
|----------------------------|--------------------------|------|
| `OLLAMA_HOST`              | `http://localhost:11434` | Adresse d'Ollama (`http://ollama:11434` dans Docker, imposé par le compose) |
| `EMBED_MODEL`              | `nomic-embed-text`       | Modèle d'embeddings |
| `LLM_MODEL`                | `qwen2.5:1.5b`           | Modèle qui rédige les réponses |
| `CHROMA_DIR`               | `./chroma_db`            | Dossier de la base vectorielle (`/app/chroma_db` dans Docker) |
| `COLLECTION_NAME`          | `story`                  | Nom de la collection Chroma |
| `STORY_PATH`               | `./data/story.txt`       | Fichier indexé par défaut |
| `CHUNK_SIZE`               | `800`                    | Taille visée d'un morceau, en caractères |
| `CHUNK_OVERLAP_PARAGRAPHS` | `1`                      | Paragraphes répétés d'un morceau au suivant |
| `TOP_K`                    | `4`                      | Nombre de passages retrouvés par défaut |
| `MIN_SCORE`                | `0.3`                    | Similarité cosinus minimale pour garder un passage |
| `CACHE_TTL`                | `300`                    | Durée de vie d'une réponse en cache, en secondes |
| `CACHE_MAX_SIZE`           | `100`                    | Nombre maximal de réponses en cache |
| `NGROK_AUTHTOKEN`          | *(vide)*                 | Token ngrok, nécessaire pour l'URL publique |
| `NGROK_DOMAIN`             | *(vide)*                 | Domaine fixe ngrok (optionnel), ex. `mon-rag.ngrok-free.app` |

Si vous changez `EMBED_MODEL` ou `LLM_MODEL`, le service `ollama-pull` télécharge le
nouveau modèle au prochain `docker compose up`. Pour `EMBED_MODEL`, videz aussi la base
(`docker compose exec api python -m cli.clean` puis `docker compose restart api`) :
les vecteurs de deux modèles différents ne sont pas comparables, et l'API renvoie
une erreur 409 tant que la base n'est pas réindexée.

## Sans Docker (développement local)

Prérequis : Python 3.11 et [Ollama](https://ollama.com) lancé en local avec les deux modèles
(`ollama pull nomic-embed-text` et `ollama pull qwen2.5:1.5b`).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m cli.ingest
uvicorn app.api:app --reload
```
