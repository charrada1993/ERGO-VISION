# Architecture IA : Analyse approfondie d'ErgoNet v2.0

*Dernière mise à jour : 2026-05-11*

**ErgoNet v2.0** est le moteur d'inférence actif alimentant le pipeline ergonomique en temps réel d'ERGO-VISION. C'est un réseau de neurones personnalisé implémenté en **NumPy pur**, conçu pour fonctionner sur le **NVIDIA Jetson Orin Nano (reComputer J3011)** sans nécessiter PyTorch, TensorFlow ou Keras.

---

## 1. Philosophie de conception

Contrairement aux modèles d'estimation de pose standard qui produisent des coordonnées articulaires, ErgoNet v2.0 est un **MLP de régression multi-tâches**. Une seule passe avant produit simultanément quatre sorties diagnostiques cliniques. C'est plus efficace que d'exécuter quatre modèles séparés et capture les relations ergonomiques inter-articulaires dans une représentation partagée.

### Pourquoi NumPy pur ?
- **Pas de surcouche de bibliothèques** : Évite la surcharge ~2 Go de RAM de PyTorch/TensorFlow au démarrage.
- **Optimisé ARM64** : Exploite les instructions **OpenBLAS avec ARM v8.2 NEON** de Jetson pour la multiplication matricielle vectorisée.
- **Latence d'inférence** : Une seule passe avant se termine en **< 8 ms** sur le CPU Jetson.
- **Portable** : Le fichier modèle `.pkl` figé fonctionne dans n'importe quel environnement Python 3.10+ — sans CUDA requis.

---

## 2. Topologie du réseau

```
Entrée (12)  →  Cachée (512, ReLU)  →  Sortie (4)
```

| Couche | Forme | Activation |
|---|---|---|
| Entrée | `(N, 12)` | — (angles normalisés Z-score) |
| Cachée | `(12 → 512)` | ReLU |
| Sortie | `(512 → 4)` | Linéaire (régression) |

### Initialisation des poids
L'initialisation Xavier (Glorot) est utilisée pour maintenir la variance entre les couches :
```python
W1 = np.random.randn(12, 512) * np.sqrt(1.0 / 12)
W2 = np.random.randn(512, 4)  * np.sqrt(1.0 / 512)
```
Cela empêche la disparition/explosion du gradient et assure une convergence propre sur le Jetson.

---

## 3. Entrée : 12 angles articulaires biomécaniques

ErgoNet v2.0 est passé des coordonnées de landmarks MediaPipe brutes (x, y, z) aux **angles articulaires calculés**. C'est l'amélioration clé par rapport à v1.0 :

| Caractéristique d'entrée | Articulation |
|---|---|
| `Neck_Flexion_deg` | Inclinaison avant/arrière de la colonne cervicale |
| `Trunk_Flexion_deg` | Inclinaison avant lombaire |
| `R_Shoulder_Flexion_deg` | Élévation de l'épaule droite |
| `L_Shoulder_Flexion_deg` | Élévation de l'épaule gauche |
| `R_Elbow_Flexion_deg` | Angle du coude droit |
| `L_Elbow_Flexion_deg` | Angle du coude gauche |
| `R_Wrist_Deviation_deg` | Déviation radiale/ulnaire du poignet droit |
| `L_Wrist_Deviation_deg` | Déviation radiale/ulnaire du poignet gauche |
| `R_Hip_Flexion_deg` | Flexion de la hanche droite |
| `L_Hip_Flexion_deg` | Flexion de la hanche gauche |
| `R_Knee_Flexion_deg` | Angle du genou droit |
| `L_Knee_Flexion_deg` | Angle du genou gauche |

**Pourquoi les angles plutôt que les landmarks bruts ?**
Les coordonnées de landmarks brutes (x, y, z) changent en fonction de la distance de la caméra et de la perspective. Les angles sont **invariants à l'échelle et à la perspective**, rendant ErgoNet v2.0 significativement plus robuste dans les déploiements industriels réels.

---

## 4. Sortie : 4 têtes diagnostiques

| Tête | Type | Description |
|---|---|---|
| `risk_score` | Continu (0,0–10,0) | Magnitude globale du risque ergonomique |
| `severity_code` | Int catégoriel | 0=Sain, 1=Faible, 2=Modéré, 3=Élevé, 4=Critique |
| `location_code` | Int catégoriel | Segment anatomique à risque le plus élevé |
| `condition_code` | Int catégoriel | Condition TMS prédite (Tendinite, Entorse, etc.) |

---

## 5. Passe avant

```python
def forward(self, X):
    self.z1 = np.dot(X, self.W1) + self.b1   # Transformation linéaire
    self.a1 = np.maximum(0, self.z1)           # ReLU
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    return self.z2
```

---

## 6. Résumé de l'entraînement

| Paramètre | Valeur |
|---|---|
| Jeu de données | `dataset_TMS_enriched.csv` (~20 000 échantillons) |
| Époques | 500 |
| Taux d'apprentissage | 0,005 |
| Fonction de perte | Erreur Quadratique Moyenne (MSE) |
| Perte finale d'entraînement | **0,2742** |
| Précision finale d'entraînement | **97,14 %** |
| Précision finale de validation | **94,22 %** |

---

## 7. Activation & Stabilité mathématique

- **ReLU** : Permet au réseau d'apprendre des frontières de décision ergonomiques non linéaires.
- **Normalisation Z-score** : Les entrées et sorties sont normalisées avant l'entraînement et dénormalisées à l'inférence.
- **Normalisation des caractéristiques à l'inférence** : Les statistiques `X_mean` et `X_std` stockées depuis l'entraînement sont appliquées aux données de caméra en direct.

---

*Documenté par l'Équipe IA ErgoVision · 2026*
