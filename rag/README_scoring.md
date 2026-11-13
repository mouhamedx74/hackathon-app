# Scoring des tendances (TrendPulse) : modèles, validation, limites

## Modèles
- **V1 : scoring à règles.** Durabilité = somme pondérée de 5 piliers (persistance 30 %, croissance 20 %, diversité 20 %, diffusion 15 %, adoption 10 %) × (1 − malus anti-hype lié au pic viral). Opportunité = 0.45·durabilité + 0.40·fit + 0.15·timing. Alerte si opportunité ≥ 75. Une tendance est jugée « durable » si durabilité ≥ 60. Validation humaine toujours requise.
- **V2 : régression logistique** (StandardScaler + LogisticRegression) sur les mêmes variables, évaluée en validation croisée stratifiée à 5 plis.
- **Pourquoi ces choix :** modèles explicables, adaptés à un jeu de référence de 30 cas (pas assez pour un modèle complexe), sans dépendance à un service externe.

## Protocole de validation
- Jeu de référence : `data/cases.csv`, 30 tendances réelles (15 durables, 15 buzz) issues de la parfumerie, des saveurs, de la food et de la beauté.
- Labels rétrospectifs : « durable » = la tendance a duré dans le temps ; « buzz » = elle a disparu rapidement.
- Notes des piliers (0–100) et `viral_spike` (0–1) : estimées par l'équipe Data à partir de connaissances publiques.
- Métriques : matrice de confusion, accuracy, précision, rappel, F1.
- Reproduire : `python -m cli.evaluate`

## Résultats
| Modèle | Accuracy | Précision | Rappel | F1 | Erreurs |
|---|---|---|---|---|---|
| V1 règles (seuil 60) | 0.93 | 1.00 | 0.87 | 0.93 | 2 faux négatifs (cardamome 58.5, kombucha 55.7) |
| V2 régression logistique (CV 5 plis) | 1.00 | 1.00 | 1.00 | 1.00 | 0 |

## Limites et biais
- **Taille :** 30 cas seulement, donc des métriques instables. Le 1.00 de V2 est très probablement du sur-apprentissage et ne se généralise pas.
- **Biais de construction :** labels rétrospectifs (biais du survivant) et notes de piliers estimées en connaissant déjà le label. Les résultats sont optimistes.
- **Poids simulés, non validés :** les pondérations de V1 sont issues des slides, pas calibrées sur des données réelles.
- **Seuil :** le seuil de 60 est à calibrer ; V1 rate deux tendances durables à score limite.
- **Couverture :** domaines limités (parfumerie, saveurs, food, beauté).
- **Suites :** valider les labels et les notes avec Product/Insights, remplacer les estimations par des mesures (volumes de recherche, publications, ventes), puis recalibrer seuil et poids.

Les résultats sont indicatifs et non généralisables ; les décisions restent soumises à validation humaine.
