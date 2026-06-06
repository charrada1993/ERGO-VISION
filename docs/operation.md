# Opération de l'IA : Moteur d'Inférence Uniquement

*Dernière mise à jour : 2026-05-30 — Intégration avec la visualisation des angles articulaires en temps réel*

Ce dossier contient le moteur **IA Opérationnelle** — la version d'ErgoNet v2.0 utilisée dans le tableau de bord en direct. Contrairement aux scripts d'entraînement, ce code ne peut ni apprendre ni changer. Il est conçu pour la **fiabilité, la vitesse et la stabilité sans dépendance**.

---

## Modèle : ErgoNet v2.0

| Propriété | Valeur |
|---|---|
| **Fichier du Modèle** | `ai/models/ergo_net_v2.pkl` |
| **Architecture** | MLP : 12 → 512 (ReLU) → 4 |
| **Précision d'Entraînement** | 97.14% |
| **Précision de Validation** | 94.22% |
| **Latence d'Inférence** | ~8 ms (CPU Jetson Orin) |
| **Dépendances** | `numpy` uniquement |

---

## Caractéristiques Clés

| Propriété | Détail |
|---|---|
| **Zéro Surcharge** | Pas de logique d'entraînement, pas de calcul de gradient, pas de rétropropagation |
| **Déterministe** | La même posture en entrée produit toujours le même diagnostic en sortie |
| **Sans Dépendances** | Seul `numpy` est requis — immunisé contre les conflits de versions de bibliothèques |
| **Matériel Agnostique** | Optimisé pour Jetson ARM64, mais fonctionne sur n'importe quel processeur avec Python 3.10+ |

---

## Comment ça Fonctionne

1. Au démarrage du serveur, `inference.py` charge `ergo_net_v2.pkl` (poids + statistiques de normalisation).
2. Pour chaque image de la caméra, `pose/skeleton.py` calcule un vecteur d'angles articulaires à 12 éléments.
3. Le vecteur d'angles est normalisé par Z-score à l'aide des moyennes/écarts-types (`X_mean` / `X_std`) stockés.
4. Une passe avant à 2 couches est exécutée :
   ```python
   a1 = np.maximum(0, X_norm @ W1 + b1)   # Couche cachée ReLU
   output = a1 @ W2 + b2                  # Sortie linéaire
   ```
5. La sortie est dé-normalisée à l'aide des moyennes/écarts-types (`y_mean` / `y_std`) stockés.
6. Les 4 têtes de diagnostic sont extraites et émises via Socket.IO vers le tableau de bord.

---

## Schéma d'Entrée / Sortie

**Entrée** : 12 angles articulaires normalisés (vecteur float32)
```
[Neck_Flexion, Trunk_Flexion, R_Shoulder, L_Shoulder,
 R_Elbow, L_Elbow, R_Wrist, L_Wrist,
 R_Hip, L_Hip, R_Knee, L_Knee]
```

**Sortie** : 4 valeurs de diagnostic
```
[risk_score (0–10), severity_code (0–4), location_code, condition_code]
```

---

## Fichiers dans ce Dossier

| Fichier | Objectif |
|---|---|
| `inference.py` | Moteur d'inférence de production (charge le fichier `.pkl`, exécute la passe avant) |
| `export_onnx.py` | Optionnel : exporte le modèle au format ONNX pour une utilisation future avec TensorRT |

---

*Documenté par l'Équipe IA ErgoVision · 2026*
