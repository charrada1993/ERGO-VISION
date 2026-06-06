# Pile Technologique ERGO-VISION 🛠️

Ce document répertorie toutes les technologies, frameworks, composants matériels et bibliothèques logicielles utilisés dans le système **ERGO-VISION** d'évaluation ergonomique de la posture en temps réel.

---

## 📋 Aperçu de la pile technologique

```
                        ┌──────────────────────────────────────┐
                        │          NVIDIA Jetson Orin          │
                        │      (Mode MAXN, Affinité CPU)       │
                        └──────────────────┬───────────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        ▼                                  ▼                                  ▼
┌──────────────┐                  ┌──────────────┐                   ┌──────────────┐
│  Matériel    │                  │   Backend    │                   │   Frontend   │
│  • Cam OAK-D │                  │  (Python 3)  │                   │  (JS/HTML5)  │
│  • USB 3.0   │                  │  • Flask     │                   │  • Vanilla   │
│              │                  │  • Socket.IO │                   │    CSS       │
└──────────────┘                  │  • MediaPipe │                   │  • Chart.js  │
                                  │  • NumPy MLP │                   │  • Three.js  │
                                  └──────────────┘                   └──────────────┘
```

---

## 🖥️ 1. Plateforme matérielle

| Technologie | Rôle & Intégration | Détails |
|---|---|---|
| **NVIDIA Jetson Orin** (reComputer J3011) | **Ordinateur hôte principal** | Cœurs CPU ARM Cortex-A78AE (architecture ARM64) sous NVIDIA JetPack Linux. |
| **Luxonis OAK-D / OAK-D Lite** | **Détection de profondeur stéréo** | Capture RGB (1280×720) synchronisé + profondeur stéréo alignée. |
| **USB 3.0 / USB 2.0** | **Protocole de transport de données** | Communication hôte–caméra via le protocole XLink de Luxonis. |

---

## 🐍 2. Moteur Backend (Python 3.10)

### Bibliothèques principales
- **`depthai`** : SDK DepthAI de Luxonis — pipeline OAK-D, files non bloquantes.
- **`mediapipe`** : 33 points clés corporels 3D en temps réel depuis le flux RGB.
- **`numpy`** : Calculs d'angles 3D, filtrage EMA, inférence MLP sans dépendance.
- **`opencv-python-headless`** : Manipulation de trames, flot optique Lucas-Kanade, flux MJPEG.
- **`pandas`** : Journalisation des sessions en CSV structurés.
- **`reportlab`** : Génération de rapports PDF ergonomiques.
- **`matplotlib`** : Graphiques de séries temporelles pour rapports PDF.

### Réseau & Serveur Web
- **`flask`** : Framework web micro hébergeant les endpoints REST.
- **`flask-socketio`** : Communication WebSocket bidirectionnelle temps réel (Socket.IO v4).
- **`simple-websocket`** : Transport WebSocket rapide sous Flask-SocketIO.

---

## 🌐 3. Tableau de bord Frontend

- **HTML5 sémantique** : Structure les pages du tableau de bord, moniteur IA, lecteur 3D et reporting.
- **Vanilla CSS3** : Design glassmorphique, modes sombre/clair via variables CSS.
- **JavaScript (ES2020)** : Manipulation DOM, rendu télémétrie temps réel, gestion WebSocket.
- **Socket.IO Client (v4.5.0)** : Abonnement aux événements temps réel (`pose_update`, `skeleton_3d`, `config`).
- **Chart.js (v4.4.0)** : Sparklines et graphiques d'historique en temps réel sur 30 secondes.
- **Three.js (r128)** : Squelette 3D WebGL interactif à la vue `/3d`.
- **Font Awesome (v6.0)** : Icônes vectorielles réactives.
- **Google Fonts (Inter & JetBrains Mono)** : Typographie soignée pour la lisibilité.

---

## 🧠 4. Intelligence artificielle & Biomécanique

### Moteur Neuronal ErgoNet v2.0
- **Architecture** : MLP à têtes multiples mappant 12 entrées articulaires vers 4 sorties diagnostiques :
  - **`risk_score`** : Continu (0,0 à 10,0).
  - **`severity_code`** : Catégoriel (Sain, Faible, Moyen, Élevé, Critique).
  - **`location_code`** : Région anatomique.
  - **`condition_code`** : 18 troubles musculosquelettiques.
- **Moteur NumPy MLP** : Inférence sans dépendance, optimisée via ARM NEON.
- **Pipeline ONNX** : Export vers ONNX pour accélération future via TensorRT.

### Stabilisation temporelle & Standards biomécaniques
- **Filtre EMA** ($\alpha = 0,15$) : Élimine le bruit des landmarks MediaPipe.
- **Logique d'hystérésis** : Lisse les mises à jour RULA/REBA aux frontières de risque.
- **Maintien d'occultation** : Dernier état valide conservé 6 trames lors d'une perte visuelle.
- **Formules biomécaniques** : Tables de consultation officielles exactes RULA et REBA.

---

## ⚙️ 5. DevOps & Optimisation de l'hôte

- **`run.sh`** : Mode MAXN, `jetson_clocks`, désactivation suspension USB, rotation journaux.
- **`taskset`** : Assignation des cœurs CPU 0–3 au pipeline temps réel.
- **Rotation des journaux** : Limite la taille des logs pour les installations embarquées.

---

## 🎓 Préparation à la soutenance
Pour un ensemble complet de questions/réponses de niveau master, consultez le **[Guide Q&A de défense devant le jury](jury_questions_answers.md)**.
