# Évaluation IA & Résultats — ErgoNet v2.0

*Dernière mise à jour : 2026-05-30 — Après l'exécution d'entraînement (500 époques, dataset_TMS_enriched.csv)*

---

## 1. Configuration de l'entraînement

| Paramètre | Valeur |
|---|---|
| **Script** | `ai/train_v2.py` |
| **Jeu de données** | `ai/data/dataset_TMS_enriched.csv` (~20 000 échantillons) |
| **Caractéristiques d'entrée** | 12 angles articulaires (bilatéraux, normalisés Z-score) |
| **Têtes de sortie** | 4 (risk_score, severity_code, location_code, condition_code) |
| **Architecture** | MLP : 12 → 512 (ReLU) → 4 |
| **Optimiseur** | Descente de gradient, lr = 0,005 |
| **Époques** | 500 |
| **Fonction de perte** | Erreur Quadratique Moyenne (MSE) |

---

## 2. Résultats d'entraînement (Exécution complète)

| Métrique | Valeur |
|---|---|
| **Perte finale d'entraînement (MSE)** | **0,2742** |
| **Précision finale d'entraînement** | **97,14 %** |
| **Perte finale de validation** | **0,2971** |
| **Précision finale de validation** | **94,22 %** |
| **Fichier modèle** | `ai/models/ergo_net_v2.pkl` |
| **Journal d'entraînement** | `ai/data/training_log.json` |

### Points saillants de la progression par époque

| Époque | Perte entraînement | Précision entraînement | Précision validation |
|---|---|---|---|
| 1 | ~0,843 | ~0,851 | ~0,826 |
| 100 | ~0,322 | ~0,935 | ~0,907 |
| 200 | ~0,303 | ~0,950 | ~0,921 |
| 300 | ~0,290 | ~0,957 | ~0,929 |
| 400 | ~0,281 | ~0,963 | ~0,934 |
| **500** | **0,274** | **0,971** | **0,942** |

---

## 3. Performance d'inférence

| Métrique | Valeur |
|---|---|
| **Latence d'inférence** | ~8 ms (CPU Jetson Orin) |
| **Empreinte RAM** | ~15 Mo (modèle + tampons) |
| **Dépendances** | `numpy` uniquement |
| **Débit** | Compatible avec un flux caméra 30 FPS |

---

## 4. Qualité des sorties diagnostiques

| Tête de sortie | Description | Qualité |
|---|---|---|
| `risk_score` | Magnitude de risque continue 0,0–10,0 | Lisse, bien calibrée |
| `severity_code` | Sévérité catégorielle 0–4 | Accord élevé avec la référence RULA/REBA |
| `location_code` | Région anatomique | Identifie correctement l'articulation à risque dominant |
| `condition_code` | Type de condition TMS | Fort sur les patterns asymétriques |

---

## 5. Logique opérationnelle

L'**IA Opérationnelle** (`ai/operation/`) est le moteur d'inférence de production « figé ». Elle :
1. Charge le `ergo_net_v2.pkl` pré-entraîné au démarrage du serveur.
2. Accepte un vecteur d'angles à 12 éléments depuis `pose/skeleton.py`.
3. Applique la normalisation `X_mean`/`X_std` stockée.
4. Exécute une seule passe avant à 2 couches (< 8 ms).
5. Dénormalise la sortie en utilisant `y_mean`/`y_std` stockés.
6. Émet les résultats via Socket.IO vers la page du tableau de bord `/ai`.

Cette inférence déterministe et sans apprentissage assure une **reproductibilité à 100 %** pour la même posture d'entrée sur toutes les sessions.

---

*Documenté par l'Équipe IA ErgoVision · 2026*
