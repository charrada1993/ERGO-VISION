# ERGO-VISION : Documentation Avancée de l'IA

*Dernière mise à jour : 2026-05-11 — Entraînement ErgoNet v2.0 terminé.*

Ce document explique l'architecture IA personnalisée, la logique du jeu de données et le pipeline d'entraînement utilisés pour l'évaluation automatisée des risques TMS (Troubles Musculosquelettiques) sur le **NVIDIA Jetson Orin (reComputer J3011)**.

---

## 1. Architecture IA : ErgoNet v2.0

Étant donné que l'environnement de déploiement nécessite des dépendances minimales et une vitesse maximale, l'IA est implémentée en tant que **Réseau de Neurones from scratch en NumPy pur**.

### Topologie du réseau

| Couche | Forme | Activation |
|---|---|---|
| Entrée | 12 angles articulaires | Normalisé Z-score |
| Cachée | 512 nœuds | ReLU |
| Sortie | 4 têtes diagnostiques | Linéaire (régression) |

### Têtes prédictives (Sortie Multi-Tâches)

| Tête | Sortie | Description |
|---|---|---|
| `risk_score` | Float 0,0–10,0 | Magnitude globale du risque ergonomique |
| `severity_code` | Int 0–4 | Niveau de sévérité : Sain → Critique |
| `location_code` | Int | Région anatomique à risque le plus élevé |
| `condition_code` | Int | Condition TMS (Tendinite, Entorse, etc.) |

### Améliorations clés en v2.0

- **Entrées basées sur les angles** (remplace les landmarks bruts) : les angles articulaires sont invariants à l'échelle et à la distance de la caméra, améliorant considérablement la stabilité en conditions réelles.
- **Couche cachée de 512 nœuds** (vs 256 en v1) : capacité plus élevée pour les patterns d'interaction multi-articulaire.
- **Jeu de données TMS enrichi** : 20 000+ échantillons vs 15 000 en v1, avec augmentation spécifique aux conditions.

---

## 2. Résultats d'entraînement (v2.0 — 2026-05-11)

| Métrique | Valeur |
|---|---|
| Jeu de données | `dataset_TMS_enriched.csv` (~20 000 échantillons) |
| Époques | 500 |
| Taux d'apprentissage | 0,005 |
| **Perte finale d'entraînement (MSE)** | **0,2742** |
| **Précision finale d'entraînement** | **97,14 %** |
| **Précision finale de validation** | **94,22 %** |
| Fichier modèle | `ai/models/ergo_net_v2.pkl` |

---

## 3. Jeu de données : Synthetic-Ergo-3D (TMS Enrichi)

Parce que les jeux de données ergonomiques professionnels avec des étiquettes cliniques précises sont rares, le modèle est entraîné sur un **Jeu de données synthétique haute-fidélité** généré via `ai/synthetic_gen.py`.

### Logique de génération

- **Contraintes cinématiques** : Simule le mouvement humain dans les limites anatomiques (ex., Cou : -10° à +60°).
- **Espace de coordonnées** : Les configurations de landmarks 3D correspondent au système de coordonnées MediaPipe.
- **Étiquetage automatique** : Les scores RULA/REBA de vérité terrain sont calculés pour chaque trame à partir de tables cliniques officielles basées uniquement sur les angles.
- **Enrichissement TMS** : Surreprésente les postures à haut risque et les patterns spécifiques aux conditions pour la sensibilité clinique.

---

## 4. Documentation détaillée

Pour des explications techniques plus approfondies, consultez le dossier `docs/` :

| Fichier | Contenu |
|---|---|
| [architecture.md](architecture.md) | Topologie du réseau, mathématiques de la passe avant, initialisation |
| [dataset.md](dataset.md) | Structure du jeu de données TMS enrichi et méthodologie de génération |
| [model_report.md](model_report.md) | Résultats d'entraînement complets, métriques et schéma du modèle sauvegardé |
| [forecasting.md](forecasting.md) | Logique de prévision de risque LSTM sur 10 jours |
| [jetson_optimization.md](jetson_optimization.md) | Optimisation matérielle, mode de puissance, optimisation mémoire |
| [onnx_guide.md](onnx_guide.md) | Voie d'export future ONNX → TensorRT |
| [evaluation.md](evaluation.md) | Progression epoch par epoch et benchmarks d'inférence |
| [operation.md](operation.md) | Détails du moteur d'inférence opérationnel |

---

## 🎓 Préparation à la soutenance
Pour des questions et réponses de niveau master concernant l'architecture IA ErgoNet v2.0, l'inférence NumPy personnalisée, la génération bootstrap de jeu de données synthétique, et plus encore, consultez le **[Guide Q&A de défense devant le jury](jury_questions_answers.md)**.

---

*Documenté par l'Équipe IA ErgoVision · 2026*
