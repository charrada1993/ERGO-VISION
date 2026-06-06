# ERGO-VISION 🦺

> **Système d'évaluation ergonomique de la posture en temps réel** alimenté par les caméras de profondeur OAK-D, l'estimation de pose MediaPipe, et le **Moteur Neuronal ErgoNet v2.0** — entraîné à **97,14 % de précision** sur plus de 20 000 échantillons TMS pour des diagnostics musculosquelettiques de qualité clinique.

*Dernière mise à jour : 2026-06-01 — Ajout du [Guide de défense devant le jury](jury_questions_answers.md) pour la présentation finale du projet.*

---

## 📋 Table des matières

1. [Aperçu général](#aperçu-général)
2. [Pile technologique](#pile-technologique)
3. [Architecture système](#architecture-système)
4. [Fonctionnalités](#fonctionnalités)
5. [Configuration matérielle requise](#configuration-matérielle-requise)
6. [Structure du projet](#structure-du-projet)
7. [Installation](#installation)
8. [Démarrage rapide](#démarrage-rapide)
9. [Angles articulaires calculés](#angles-articulaires-calculés)
10. [Référence API](#référence-api)
11. [Référence de scoring RULA / REBA](#référence-de-scoring-rula--reba)
12. [Configuration](#configuration)
13. [Dépannage](#dépannage)
14. [Guide de défense & Q&A technique](jury_questions_answers.md)
15. [Contribuer](#contribuer)


---

## Aperçu général

**ERGO-VISION** est une plateforme open-source d'évaluation des risques ergonomiques en temps réel pour la santé au travail. Pour un aperçu complet des dépendances logicielles et matérielles, consultez [tech.md](tech.md).

---

## Pile technologique

> [!NOTE]
> Pour une description complète de toutes les bibliothèques, de l'intégration matérielle et des versions logicielles, consultez la [Documentation de la pile technologique (tech.md)](tech.md).

### 🖥️ Plateforme matérielle

| Composant | Technologie | Rôle |
|---|---|---|
| Ordinateur Edge AI | **NVIDIA Jetson Orin** (reComputer J3011) | Hôte principal d'inférence (ARM Cortex-A78AE + CUDA) |
| Caméra de profondeur | **Luxonis OAK-D** (OpenCV AI Kit with Depth) | RGB 1280×720 + profondeur stéréo alignée |
| USB | **USB 2.0 / USB 3.0** | Transport des données caméra via le protocole XLink |

---

### 🐍 Backend — Python

| Bibliothèque | Version | Rôle |
|---|---|---|
| **Python** | 3.10+ | Langage principal |
| **depthai** | 2.24+ | SDK caméra OAK-D — pipeline, files XLink, post-traitement de la profondeur |
| **mediapipe** | 0.10+ | Estimation de la pose humaine — 33 points clés corporels (x, y, z) |
| **opencv-python** | 4.8+ | Traitement d'image, flot optique Lucas-Kanade, streaming MJPEG |
| **numpy** | 1.24+ | Tous les calculs vectoriels 3D, calcul d'angles, lissage EMA |
| **flask** | 3.0+ | Serveur web et routes HTTP |
| **flask-socketio** | 5.3+ | Communication WebSocket en temps réel (Socket.IO v4) |
| **simple-websocket** | — | Transport WebSocket backend pour Flask-SocketIO |
| **pandas** | 2.0+ | Journalisation des sessions CSV et export de données |
| **matplotlib** | 3.7+ | Génération de graphiques de séries temporelles pour les rapports PDF |
| **reportlab** | 4.0+ | Générateur de rapports PDF ergonomiques |

---

### 🌐 Frontend — Tableau de bord Web

| Technologie | Version | Rôle |
|---|---|---|
| **HTML5** | — | Structure des pages et disposition sémantique |
| **Vanilla CSS3** | — | Thème sombre glassmorphique, propriétés CSS personnalisées (jetons de design) |
| **JavaScript (ES2020)** | — | Toute la logique côté client, manipulation du DOM |
| **Socket.IO Client** | 4.5.0 | Réception des données WebSocket en temps réel depuis le backend Flask |
| **Chart.js** | 4.4.0 | Graphiques linéaires déroulants en temps réel (angles articulaires, scores RULA/REBA, 12 sparklines) |
| **Three.js** | r128 | Visionneuse de squelette 3D interactive à `/3d` |
| **Font Awesome** | 6.0 | Icônes vectorielles dans l'interface |
| **Google Fonts** | — | Typographie Inter / JetBrains Mono |

---

### 🧠 IA / Apprentissage automatique

| Technologie | Rôle |
|---|---|
| **ErgoNet v2.0** | MLP à 4 têtes personnalisé (basé sur les angles) : score de risque, sévérité, code de localisation, code de condition |
| **NumPy MLP** | Moteur d'inférence sans dépendance (optimisé NEON sur ARM) — aucun TensorFlow/PyTorch requis |
| **Générateur de données synthétiques** | `ai/synthetic_gen.py` — fabrication de jeu de données ergonomique TMS haute-fidélité (20 000+ échantillons) |
| **Lissage temporel EMA** | α=0,15 — pondération historique à 85 %, élimine le bruit des points clés aux seuils de scoring RULA |
| **Hystérésis des scores** | Empêche le scintillement RULA/REBA aux frontières de niveau de risque |
| **Maintien en cas d'occultation** | Conserve les derniers angles valides pendant 6 images manquées (~0,75 s) lors d'une perte de suivi |

**Architecture ErgoNet v2.0 :**
```
Entrée (12 angles articulaires) → Dense(512, ReLU) → Dense(256, ReLU) → Dense(128, ReLU)
→ 4 têtes de sortie :
    risk_score     (continu 0–10)
    severity_code  (classe 0–4 : Sain → Critique)
    location_code  (classe 0–8 : région corporelle)
    condition_code (classe 0–17 : condition TMS)
```

**Résultats d'entraînement :** Perte `0,2742` | Précision d'entraînement `97,14 %` | Précision de validation `94,22 %`

---

### 📐 Standards de biomécanique / Ergonomie

| Standard | Implémentation |
|---|---|
| **RULA** (Rapid Upper Limb Assessment) | Score complet à 7 niveaux via les tables de consultation officielles exactes |
| **REBA** (Rapid Entire Body Assessment) | Score complet à 15 niveaux via les tables de consultation officielles exactes |
| **McAtamney & Corlett (1993)** | Référence de la méthodologie RULA |
| **Hignett & McAtamney (2000)** | Référence de la méthodologie REBA |

---

### ⚙️ DevOps / Déploiement

| Technologie | Rôle |
|---|---|
| **Bash** (`run.sh`) | Script de démarrage Jetson : nvpmodel MAXN, jetson_clocks, gestion alimentation USB, rotation des logs |
| **taskset** | Assignation d'affinité CPU aux cœurs 0–3 (grands cœurs Cortex-A78) |
| **Git / GitHub** | Contrôle de version et dépôt distant |
| **MJPEG sur HTTP** | Point de terminaison de streaming vidéo (`/video_feed`, `/depth_feed`) à 8 fps |

---

## Architecture système

```
┌─────────────────────────────────────────────────────────────┐
│                  Caméra OAK-D (USB 2.0)                     │
│         RGB 1280×720 + Profondeur stéréo alignée @ 8 fps    │
└────────────────────────┬────────────────────────────────────┘
                         │  DepthAI XLink
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  camera/manager.py  —  Pipeline DepthAI                     │
│  • Files non bloquantes (taille=1)  • tryGetAll() fraîcheur │
│  • Profondeur : filtres speckle + temporel + spatial + seuil│
│  • Mode simulation (MockCamera) sans OAK-D                  │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────┐
│  pose/estimator.py        │  MediaPipe Pose → 33 points clés
│  pose/fusion.py           │  Fusion multi-caméra de landmarks
│  pose/skeleton.py         │  Mathématiques vectorielles 3D → 30+ angles
│  (patch médian profondeur 3×3) │  EMA α=0,15 + maintien occultation
└───────────────┬───────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌────────────┐   ┌────────────┐
│ rula.py    │   │ reba.py    │
│ Score 1–7  │   │ Score 1–15 │
└────────────┘   └────────────┘
       │                 │
       └────────┬────────┘
                ▼
┌─────────────────────────────────────────────────────────────┐
│  web/socket_events.py  — Thread d'arrière-plan @ ~8 Hz      │
│  ai/operation/inference.py — Inférence ErgoNet v2.0         │
│  Émet : pose_update, skeleton_3d via Socket.IO              │
└────────────────────────┬────────────────────────────────────┘
                         │  WebSocket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Tableau de bord Flask (web/routes.py)                      │
│  ├── / tableau de bord  — Graphiques Chart.js + carte corps │
│  ├── /camera            — Flux vidéo MJPEG + profondeur     │
│  ├── /rula              — Détail des sous-scores RULA       │
│  ├── /reba              — Détail des sous-scores REBA       │
│  ├── /3d                — Visionneuse squelette 3D Three.js │
│  ├── /ai                — Courbes d'entraînement ErgoNet    │
│  ├── /collection        — Enregistreur de sessions CSV      │
│  └── /report            — Générateur de rapports PDF        │
└─────────────────────────────────────────────────────────────┘
```

---

## Fonctionnalités

| Fonctionnalité | Détails |
|---|---|
| 🎥 **Caméra OAK-D** | RGB + profondeur stéréo, alignée, post-traitée |
| 🦴 **MediaPipe Pose** | 33 points clés corporels, CPU uniquement, temps réel |
| 📐 **Angles articulaires complets** | Flexion, extension, flexion latérale et rotation calculées par articulation |
| 📊 **RULA / REBA** | Scoring par tables de consultation officielles, Groupes A+B, sous-scores |
| 🧠 **ErgoNet v2.0** | MLP basé sur les angles, précision 97,14 %, 18 classes de conditions TMS |
| 📈 **12 sparklines en direct** | Graphiques de tendance par articulation avec badges numériques anatomiques |
| 🌀 **IMU visuel** | Estimation du mouvement et de l'orientation par flot optique Lucas-Kanade |
| 🌐 **Tableau de bord Socket.IO** | Streaming WebSocket en temps réel vers n'importe quel navigateur |
| 🎬 **Flux MJPEG** | RGB + carte de couleurs de profondeur en direct à 8 fps via HTTP |
| 📄 **Rapports PDF** | Rapports de risque ergonomique automatisés à partir des sessions CSV |
| ⚡ **Optimisé Jetson** | Affinité CPU, mode MAXN, files non bloquantes, limite 8 fps |
| 🌙 **Mode sombre / clair** | Bascule de thème avec synchronisation Chart.js |
| 🔄 **Mode simulation** | MockCamera sans OAK-D connectée |

---

## Configuration matérielle requise

| Composant | Minimum | Recommandé |
|---|---|---|
| Caméra | 1× OAK-D Lite | 1–3× OAK-D (IMU visuel via flot optique) |
| Hôte | 4 cœurs ARM/x86, 4 Go de RAM | NVIDIA Jetson Orin, 8 Go de RAM |
| USB | USB 2.0 | USB 3.0 |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

---

## Structure du projet

```
ERGO-VISION/
├── app.py                      # Point d'entrée — orchestre tous les modules
├── config.py                   # Configuration globale (FPS, chemins, seuils)
├── requirements.txt            # Dépendances Python
├── run.sh                      # Script de démarrage Jetson
│
├── camera/
│   ├── manager.py              # Pipeline DepthAI OAK-D + MockCamera
│   ├── imu_manager.py          # IMU visuel via flot optique Lucas-Kanade
│   └── calibration.py          # Intrinsèques / extrinsèques RGB
│
├── pose/
│   ├── estimator.py            # MediaPipe Pose → 33 landmarks
│   ├── fusion.py               # Fusion de landmarks multi-caméra
│   └── skeleton.py             # Calcul d'angles articulaires 3D (30+ clés)
│
├── ergonomics/
│   ├── rula.py                 # Calculateur RULA (score 1–7)
│   ├── reba.py                 # Calculateur REBA (score 1–15)
│   └── risk.py                 # Détecteur d'anomalies
│
├── ai/
│   ├── models/ergo_net_v2.pkl  # Poids ErgoNet v2.0
│   ├── data/                   # Jeu de données TMS + logs d'entraînement
│   ├── operation/              # Moteur d'inférence + export ONNX
│   ├── train_v2.py             # Pipeline d'entraînement
│   └── synthetic_gen.py        # Générateur de données synthétiques
│
├── web/
│   ├── routes.py               # Routes Flask
│   ├── socket_events.py        # Socket.IO + thread de traitement
│   ├── static/css/style.css    # Thème glassmorphique sombre/clair
│   └── static/js/
│       ├── dashboard.js        # Graphiques Chart.js + client Socket.IO
│       └── 3d_viewer.js        # Squelette 3D Three.js
│   └── templates/              # Pages HTML Jinja2
│
├── reporting/
│   ├── report_generator.py     # Générateur PDF (ReportLab)
│   └── graphs.py               # Helpers graphiques Matplotlib
│
└── docs/
    ├── README.md               # Ce fichier (présentation générale)
    ├── architecture.md         # Architecture IA ErgoNet v2.0
    ├── dataset.md              # Jeu de données TMS enrichi
    ├── model_report.md         # Rapport technique du modèle
    ├── forecasting.md          # Prévision de risque sur 10 jours
    ├── jetson_optimization.md  # Optimisation Jetson Orin
    ├── onnx_guide.md           # Guide d'export ONNX
    ├── tech.md                 # Pile technologique complète
    ├── ai_documentation.md     # Documentation avancée de l'IA
    ├── jury_questions_answers.md # Guide de défense devant le jury
    ├── evaluation.md           # Évaluation & résultats ErgoNet v2.0
    ├── operation.md            # Moteur d'inférence opérationnel
    └── examples.md             # Exemples d'utilisation DepthAI
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/charrada1993/ERGO-VISION.git
cd ERGO-VISION

# 2. Créer un environnement virtuel
python3 -m venv venv && source venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Linux) Règles udev pour OAK-D
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Démarrage rapide

```bash
# Option A — Lanceur Jetson (fixe les horloges, gère l'alimentation USB)
bash run.sh

# Option B — Python direct
python3 app.py
```

Ouvrez **http://localhost:5000** (ou l'adresse IP de votre Jetson sur le port 5000 depuis n'importe quel appareil du réseau).

| URL | Page |
|---|---|
| `/` | Tableau de bord principal — graphiques en direct, carte des risques corporels, analyse IA |
| `/camera` | Flux MJPEG RGB + profondeur |
| `/rula` | Détail des sous-scores RULA |
| `/reba` | Détail des sous-scores REBA |
| `/3d` | Squelette 3D interactif Three.js |
| `/ai` | Courbes d'entraînement ErgoNet + moniteur d'inférence en direct |
| `/collection` | Enregistreur de sessions CSV |
| `/report` | Générateur de rapports PDF automatisé |

---

## Angles articulaires calculés

`pose/skeleton.py` émet les clés d'angles suivantes via `pose_update` :

| Clé | Description |
|---|---|
| `neck` | Flexion/extension du cou |
| `neck_lateral_flexion` | Inclinaison latérale de la tête / flexion |
| `neck_rotation` | Rotation de la tête |
| `trunk` | Flexion/extension du tronc |
| `trunk_lateral_flexion` | Flexion latérale du tronc |
| `trunk_rotation` | Rotation du tronc |
| `upper_arm_left/right` | Flexion/extension de l'épaule |
| `abd_l/r` | Abduction/adduction de l'épaule |
| `elbow_left/right` | Flexion/extension du coude |
| `elb_rotation_l/r` | Rotation de l'avant-bras |
| `wrist_left/right` | Flexion/extension du poignet |
| `wri_deviation_l/r` | Déviation radiale/ulnaire du poignet |
| `wri_rotation_l/r` | Rotation de l'avant-bras au niveau du poignet |
| `hip_left/right` | Flexion/extension de la hanche |
| `thi_abduction_adduction_l/r` | Abduction/adduction de la cuisse |
| `thi_rotation_l/r` | Rotation de la hanche |
| `knee_left/right` | Flexion/extension du genou |

---

## Référence API

### Événements Socket.IO (Serveur → Client)

| Événement | Description |
|---|---|
| `config` | `{mode, usb3}` — nombre de caméras et vitesse USB |
| `pose_update` | Charge utile de posture complète à ~8 Hz |
| `skeleton_3d` | `{landmarks: [[x,y,z]×33]}` pour la visionneuse Three.js |

**Charge utile `pose_update` :**
```json
{
  "angles":       {"neck": 12.3, "neck_rotation": -5.1, "trunk": 8.2, ...},
  "rula":         4,
  "reba":         7,
  "risk_level":   "Moyen",
  "anomalies":    ["Flexion du cou : 42,0° (>40°)"],
  "rula_details": {"score_a": 3, "score_b": 4, "score_c": 4},
  "reba_details": {"table_a": 5, "table_b": 6, "score_c": 7},
  "ai_results":   {"risk_score": 3.2, "severity_code": 2, "condition_code": 9}
}
```

### Points de terminaison REST

| Point de terminaison | Méthode | Description |
|---|---|---|
| `/api/config` | GET | Mode caméra et configuration matérielle |
| `/api/sessions` | GET | Liste des sessions CSV enregistrées |
| `/api/reports` | GET | Liste des rapports PDF générés |
| `/api/download_csv/<fichier>` | GET | Télécharger un CSV de session |
| `/api/download_report/<fichier>` | GET | Télécharger un rapport PDF |
| `/api/generate_report/<fichier>` | GET | Générer un PDF à partir d'un CSV |
| `/api/training_log` | GET | JSON de l'historique d'entraînement ErgoNet v2.0 |
| `/video_feed` | GET | Flux MJPEG RGB (8 fps) |
| `/depth_feed` | GET | Flux MJPEG carte de couleurs de profondeur (4 fps) |

---

## Référence de scoring RULA / REBA

### Score final RULA

| Score | Risque | Action |
|---|---|---|
| 1–2 | ✅ Acceptable | Aucune action |
| 3–4 | ⚠️ Faible | Surveiller |
| 5–6 | 🔶 Moyen | Changement nécessaire |
| 7 | 🔴 Très élevé | Action immédiate |

### Score final REBA

| Score | Risque | Action |
|---|---|---|
| 1 | ✅ Négligeable | Aucune action |
| 2–3 | ⚠️ Faible | Surveiller |
| 4–7 | 🔶 Moyen | Amélioration nécessaire |
| 8–10 | 🔴 Élevé | Intervention rapide |
| 11–15 | 🆘 Très élevé | Action immédiate |

---

## Configuration

Modifier `config.py` :

```python
class Config:
    MAX_CAMERAS      = 3      # Nombre maximum de caméras OAK-D
    PROCESSING_FPS   = 10     # Fréquence d'estimation de pose (Hz)
    LOG_INTERVAL     = 0.5    # Intervalle de journalisation CSV (s)
    LOAD_KG_DEFAULT  = 0      # Charge portée (kg)

class JetsonConfig:
    VIDEO_WIDTH      = 320    # Largeur du flux MJPEG
    VIDEO_HEIGHT     = 240    # Hauteur du flux MJPEG
    VIDEO_STREAM_FPS = 8      # Limite FPS MJPEG
    VIDEO_JPEG_QUALITY = 50   # Qualité JPEG (0–100)
```

---

## Dépannage

| Problème | Solution |
|---|---|
| `Aucun appareil OAK-D trouvé` | Vérifier USB, lancer `lsusb \| grep 03e7`, installer les règles udev |
| `Port 5000 déjà utilisé` | Lancer `fuser -k 5000/tcp` puis redémarrer |
| Graphiques vides / pas de données | Vérifier la console navigateur pour les erreurs JS ; s'assurer que la caméra est connectée |
| Flux vidéo vide | Caméra en mode simulation — connecter l'OAK-D |
| PDF non généré | S'assurer que `reports/` est accessible en écriture ; `pip install reportlab matplotlib` |
| MediaPipe sans landmarks | Améliorer l'éclairage ; le sujet doit être entièrement visible dans le cadre |

---

## 🎓 Guide de défense & Q&A technique

Pour préparer votre soutenance académique ou professionnelle, consultez :
👉 **[JURY_QUESTIONS_ANSWERS.md](jury_questions_answers.md)**

Ce guide complet inclut des questions et réponses de niveau expert concernant :
* Optimisations embarquées sur Jetson Orin (`nvpmodel`, `jetson_clocks`, affinité `taskset`).
* Alignement de profondeur stéréoscopique 3D & filtration du bruit MediaPipe.
* Mathématiques d'angles biomécaniques & implémentations des tables officielles RULA/REBA.
* Structure du **Moteur IA ErgoNet v2.0** & passe avant NumPy pur.
* Compromis système et évolutivité future du produit.

---

## Contribuer

1. Forker le dépôt
2. Créer une branche : `git checkout -b feature/votre-fonctionnalite`
3. Committer : `git commit -m "feat: votre fonctionnalité"`
4. Pousser : `git push origin feature/votre-fonctionnalite`
5. Ouvrir une Pull Request

---

## Licence

**Licence MIT** — voir [LICENSE](LICENSE) pour les détails.

---

## Remerciements

- [Luxonis DepthAI](https://docs.luxonis.com/) — SDK caméra OAK-D
- [Google MediaPipe](https://developers.google.com/mediapipe) — Estimation de la pose humaine
- [Chart.js](https://www.chartjs.org/) — Visualisation de données en temps réel
- [Three.js](https://threejs.org/) — Visionneuse de squelette 3D
- McAtamney & Corlett (1993) — Méthodologie RULA
- Hignett & McAtamney (2000) — Méthodologie REBA

---

*Construit avec ❤️ pour la santé et la sécurité au travail sur NVIDIA Jetson Orin.*
