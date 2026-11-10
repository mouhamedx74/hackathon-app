"""Cache en mémoire (dict + TTL), sans dépendance externe.

Stocke les réponses récentes pour éviter de refaire embedding + LLM
sur des questions déjà posées. Le cache est perdu au redémarrage de l'API.
"""

import time

from .config import CACHE_MAX_SIZE, CACHE_TTL


class SimpleCache:
    """Cache clé -> valeur avec durée de vie (TTL) et taille maximale."""

    def __init__(self, ttl: int = CACHE_TTL, max_size: int = CACHE_MAX_SIZE):
        self._store = {}  # clé normalisée -> (timestamp, valeur)
        self._ttl = ttl
        self._max_size = max_size

    def _normalize(self, key: str) -> str:
        # "  Qui est Éclair ? " et "qui est éclair ?" donnent la même clé
        return key.strip().lower()

    def get(self, key: str):
        k = self._normalize(key)
        if k in self._store:
            ts, value = self._store[k]
            if time.time() - ts < self._ttl:
                return value
            del self._store[k]  # entrée expirée
        return None

    def set(self, key: str, value) -> None:
        k = self._normalize(key)
        # Si le cache est plein, supprimer l'entrée la plus ancienne
        if k not in self._store and len(self._store) >= self._max_size:
            oldest = min(self._store, key=lambda x: self._store[x][0])
            del self._store[oldest]
        self._store[k] = (time.time(), value)

    def clear(self) -> int:
        """Vide le cache et renvoie le nombre d'entrées supprimées."""
        count = len(self._store)
        self._store.clear()
        return count

    def stats(self) -> dict:
        now = time.time()
        valid = sum(1 for ts, _ in self._store.values() if now - ts < self._ttl)
        return {"total": len(self._store), "valid": valid, "ttl": self._ttl, "max_size": self._max_size}


# Instance partagée par toute l'API
cache = SimpleCache()
