# ErgoNet v2.0 : Rapport technique du modèle 🧠

*Dernière mise à jour : 2026-05-11 — Entraînement ErgoNet v2.0 terminé.*

---

## 1. Type de modèle & Architecture

**ErgoNet v2.0** est un **Perceptron Multi-Couches (MLP) à sorties multiples**. Contrairement aux modèles de classification traditionnels qui prédisent une seule étiquette, cette architecture utilise une couche d'extraction de caractéristiques partagée pour prédire simultanément quatre résultats ergonomiques distincts à partir d'une seule passe avant.

### Spécifications

| Composant | Détails |
|---|---|
| **Couche d'entrée** | 12 nœuds — angles articulaires biomécaniques normalisés |
| **Couche cachée** | 512 nœuds — activation ReLU |
| **Couche de sortie** | 4 nœuds — têtes de régression multi-tâches |
| **Initialisation** | Xavier (Glorot) — prévient la disparition/explosion du gradient |
| **Optimiseur** | Descente de gradient, lr = 0,005, 500 époques |
| **Framework** | NumPy pur — zéro dépendance ML externe |

### Caractéristiques d'entrée (12 angles articulaires)

| Caractéristique | Description |
|---|---|
| `Neck_Flexion_deg` | Inclinaison avant/arrière du cou |
| `Trunk_Flexion_deg` | Inclinaison avant du tronc |
| `R/L_Shoulder_Flexion_deg` | Élévation de l'épaule (bilatérale) |
| `R/L_Elbow_Flexion_deg` | Angle du coude (bilatéral) |
| `R/L_Wrist_Deviation_deg` | Déviation radiale/ulnaire du poignet (bilatérale) |
| `R/L_Hip_Flexion_deg` | Angle de la hanche (bilatéral) |
| `R/L_Knee_Flexion_deg` | Angle du genou (bilatéral) |

### Têtes de sortie (Multi-Tâches)

| Tête | Type | Plage |
|---|---|---|
| **Score de risque** | Régression continue | 0,0 – 10,0 |
| **Code de sévérité** | Catégoriel | 0 (Sain) → 4 (Critique) |
| **Code de localisation** | ID de segment anatomique | 0 – N |
| **Code de condition** | Prédiction diagnostique | ex. Tendinite, Entorse |

---

## 2. Justification : Pourquoi cette architecture ?

### A. Déploiement sans dépendance (NumPy)
Le modèle est implémenté en **NumPy pur**. C'est critique pour l'environnement **NVIDIA Jetson Orin** parce que :
- Élimine le besoin de PyTorch/TensorFlow/Keras (~2 Go de RAM au chargement).
- Contourne les conflits de dépendances complexes sur l'architecture ARM64 de Jetson.
- Le fichier modèle `.pkl` est léger (<5 Mo) et se charge en millisecondes.

### B. Inférence haute vitesse
Les multiplications matricielles vectorisées dans NumPy via OpenBLAS sur ARM v8.2 NEON donnent **~8 ms** de latence d'inférence.

### C. Invariance basée sur les angles (Amélioration v2.0)
ErgoNet v1.0 utilisait les landmarks MediaPipe bruts (x, y, z). La v2.0 passe aux **angles articulaires calculés**.
- Les landmarks bruts changent en fonction de la distance du sujet à la caméra. Les angles articulaires sont **mathématiquement invariants** à la distance et à la distorsion de perspective.

---

## 3. Logique de la passe avant

```python
# Couche 1
z1 = X @ W1 + b1      # (N, 12) × (12, 512) → (N, 512)
a1 = ReLU(z1)          # Activation non linéaire

# Couche 2 (Sortie)
output = a1 @ W2 + b2  # (N, 512) × (512, 4) → (N, 4)
```

Le vecteur de sortie est ensuite dénormalisé en utilisant les `y_mean` et `y_std` stockés et découpé en quatre têtes diagnostiques, qui sont servies au tableau de bord via Socket.IO.

---

## 4. Pipeline d'entraînement (`ai/train_v2.py`)

```bash
cd ~/ERGO-VISION/ai
python3 train_v2.py
```

Le script :
1. Charge `ai/data/dataset_TMS_enriched.csv` (~20 000 échantillons).
2. Normalise Z-score les entrées (X) et les sorties (y).
3. Exécute une boucle de descente de gradient personnalisée sur 500 époques.
4. Sauvegarde le modèle figé dans `ai/models/ergo_net_v2.pkl`.
5. Écrit le journal d'entraînement époque par époque dans `ai/data/training_log.json`.
6. Affiche les courbes de précision/perte d'entraînement et de validation via matplotlib.

---

## 5. Métriques de performance (v2.0 — Exécution complète)

| Métrique | Valeur |
|---|---|
| **Jeu de données d'entraînement** | 20 000+ échantillons TMS enrichis (`dataset_TMS_enriched.csv`) |
| **Époques** | 500 |
| **Perte finale d'entraînement (MSE)** | **0,2742** |
| **Précision finale d'entraînement** | **97,14 %** |
| **Perte finale de validation** | **0,2971** |
| **Précision finale de validation** | **94,22 %** |
| **Latence d'inférence** | ~8 ms (CPU Jetson Orin) |
| **Fichier modèle** | `ai/models/ergo_net_v2.pkl` |

---

## 6. Structure du modèle sauvegardé

```python
state = {
    'version': '2.0',
    'W1': np.ndarray,   # (12, 512)
    'b1': np.ndarray,   # (1, 512)
    'W2': np.ndarray,   # (512, 4)
    'b2': np.ndarray,   # (1, 4)
    'X_mean': np.ndarray,   # moyenne d'entrée par caractéristique
    'X_std':  np.ndarray,   # écart-type d'entrée par caractéristique
    'y_mean': np.ndarray,   # moyenne de sortie par cible
    'y_std':  np.ndarray,   # écart-type de sortie par cible
    'input_cols':  list,    # 12 noms de colonnes d'angles
    'target_cols': list     # ['risk_score', 'severity_code', 'location_code', 'condition_code']
}
```

---

*Documenté par l'Équipe IA ErgoVision · 2026*
