# Jeu de données : Jeu de données ergonomique enrichi TMS

*Dernière mise à jour : 2026-05-11*

ErgoNet v2.0 est entraîné sur le **Jeu de données TMS Enrichi** (`ai/data/dataset_TMS_enriched.csv`), un jeu de données ergonomique synthétique haute-fidélité généré spécifiquement pour la prédiction du risque de troubles musculosquelettiques (TMS).

---

## 1. Aperçu du jeu de données

| Propriété | Valeur |
|---|---|
| **Fichier** | `ai/data/dataset_TMS_enriched.csv` |
| **Taille** | ~18,6 Mo |
| **Échantillons** | 20 000+ enregistrements de postures uniques |
| **Caractéristiques d'entrée** | 12 angles articulaires (bilatéraux, biomécaniques) |
| **Caractéristiques cibles** | 4 sorties cliniques (risque, sévérité, localisation, condition) |

---

## 2. Caractéristiques d'entrée (12 colonnes d'angles)

Tous les angles sont en **degrés** et représentent de vraies mesures d'articulations biomécaniques :

| Nom de colonne | Articulation | Plage |
|---|---|---|
| `Neck_Flexion_deg` | Flexion/extension cervicale | −10° à +60° |
| `Trunk_Flexion_deg` | Inclinaison avant lombaire | −10° à +90° |
| `R_Shoulder_Flexion_deg` | Élévation de l'épaule droite | 0° à 180° |
| `L_Shoulder_Flexion_deg` | Élévation de l'épaule gauche | 0° à 180° |
| `R_Elbow_Flexion_deg` | Coude droit | 0° à 150° |
| `L_Elbow_Flexion_deg` | Coude gauche | 0° à 150° |
| `R_Wrist_Deviation_deg` | Déviation radiale/ulnaire du poignet droit | −30° à +30° |
| `L_Wrist_Deviation_deg` | Déviation radiale/ulnaire du poignet gauche | −30° à +30° |
| `R_Hip_Flexion_deg` | Hanche droite | 0° à 120° |
| `L_Hip_Flexion_deg` | Hanche gauche | 0° à 120° |
| `R_Knee_Flexion_deg` | Genou droit | 0° à 135° |
| `L_Knee_Flexion_deg` | Genou gauche | 0° à 135° |

---

## 3. Caractéristiques cibles (4 étiquettes cliniques)

| Nom de colonne | Type | Description |
|---|---|---|
| `risk_score` | Float (0,0–10,0) | Magnitude globale du risque ergonomique |
| `severity_code` | Int (0–4) | Sévérité : 0=Sain, 1=Faible, 2=Modéré, 3=Élevé, 4=Critique |
| `location_code` | Int | Région anatomique à risque le plus élevé |
| `condition_code` | Int | Condition TMS : Tendinite, Douleur dorsale, Entorse, etc. |

---

## 4. Méthodologie de génération synthétique

Les jeux de données ergonomiques de qualité d'entraînement avec des étiquettes cliniques précises sont rarement disponibles publiquement. ERGO-VISION utilise une approche **Bootstrap Synthétique** via `ai/synthetic_gen.py`.

### A. Échantillonnage aléatoire de mouvement
Le simulateur échantillonne des angles articulaires aléatoires dans des plages anatomiquement valides :
- **Cou** : Flexion/Extension (−10° à +60°), Latéral (±35°).
- **Tronc** : Flexion (−10° à +90°).
- **Épaules** : Abduction/Flexion (0° à 180°).
- **Coudes** : Flexion (0° à 150°).
- **Poignets** : Déviation (±30°).
- **Hanches/Genoux** : Plage physiologique.

### B. Projection de landmarks
Pour chaque ensemble d'angles, le système calcule les **coordonnées XYZ 3D** de tous les 33 landmarks MediaPipe, créant une vérité terrain parfaite qui relie la géométrie des landmarks aux valeurs d'angles précises.

### C. Étiquetage automatique RULA/REBA
Chaque posture générée est passée à travers le **moteur de scoring RULA/REBA officiel basé sur les angles** pour attribuer des scores cliniques (sans métriques de charge/effort subjectifs). L'IA apprend ensuite : *« Quand je vois ces angles, l'état de risque est X. »*

### D. Enrichissement TMS
Le jeu de données est enrichi avec des **patterns spécifiques aux conditions** :
- Surreprésentation des postures à haut risque (RULA 5+) pour améliorer la sensibilité.
- Patterns d'asymétrie bilatérale (déséquilibre gauche/droite) associés au développement réel de TMS.
- Micro-variations répétitives de postures pour améliorer la généralisation.

---

## 5. Avantages des données synthétiques

| Avantage | Détails |
|---|---|
| **Pas d'annotation manuelle** | Aucun clinicien n'est nécessaire pour évaluer des milliers d'images |
| **Couverture extrême** | Peut générer des postures trop douloureuses pour les humains lors de la collecte de données |
| **Zéro erreur d'étiquette** | Les étiquettes sont calculées mathématiquement — aucune erreur d'annotation humaine |
| **Distribution contrôlable** | Peut suréchantillonner les conditions rares à haut risque selon les besoins |
| **Conforme à la vie privée** | Aucune donnée réelle de patient ou de travailleur impliquée |

---

## 6. Prétraitement à l'entraînement

Avant l'entraînement, la normalisation suivante est appliquée :

```python
# Normalisation des entrées (Z-score)
X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-6
X_norm = (X - X_mean) / X_std

# Normalisation des sorties (Z-score)
y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6
y_norm = (y - y_mean) / y_std
```

Les statistiques `X_mean`, `X_std`, `y_mean` et `y_std` sont sauvegardées dans le fichier modèle `.pkl` et appliquées de façon identique au moment de l'inférence.

---

*Documenté par l'Équipe IA ErgoVision · 2026*
