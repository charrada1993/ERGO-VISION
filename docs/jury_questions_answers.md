# 🎓 ERGO-VISION : Guide de Défense Finale & Q&A Technique

> **Ce document est conçu pour vous aider à réussir votre soutenance finale.** Il compile les questions les plus avancées, rigoureuses et techniques qu'un jury d'experts, de professeurs et de professionnels de l'industrie pourrait poser concernant vos décisions d'ingénierie, l'architecture système, les formules biomécaniques, le modèle IA personnalisé et les optimisations embarquées.

---

## 💡 Conseils rapides pour la soutenance orale

1. **Soyez confiant & humble :** Assumez vos choix de conception. Si vous ne connaissez pas une réponse, guidez-les à travers votre *logique d'ingénierie* plutôt que de deviner.
2. **Focalisez-vous sur les compromis :** Les jurys aiment entendre *pourquoi* vous avez choisi une technologie plutôt qu'une autre (ex. NumPy pur au lieu de PyTorch, ou CSS vanilla au lieu de frameworks lourds). Expliquez les compromis en termes d'**empreinte mémoire, surcharge CPU et contraintes Edge AI**.
3. **Soulignez l'impact réel :** Reliez l'implémentation technique directement à la santé et la sécurité au travail (prévention des Troubles Musculosquelettiques / TMS) pour montrer la valeur de votre travail.

---

## 📋 Table des matières
1. [Catégorie A : Architecture système & Plateforme matérielle](#-catégorie-a--architecture-système--plateforme-matérielle)
2. [Catégorie B : Vision par ordinateur & Détection de profondeur 3D](#-catégorie-b--vision-par-ordinateur--détection-de-profondeur-3d)
3. [Catégorie C : Biomécanique & Standards ergonomiques](#-catégorie-c--biomécanique--standards-ergonomiques)
4. [Catégorie D : Apprentissage automatique personnalisé (ErgoNet v2.0 & Moteur NumPy)](#-catégorie-d--apprentissage-automatique-personnalisé-ergonet-v20--moteur-numpy)
5. [Catégorie E : Systèmes Web & Pipeline de télémétrie dynamique](#-catégorie-e--systèmes-web--pipeline-de-télémétrie-dynamique)
6. [Catégorie F : Systèmes embarqués & Optimisations Jetson Orin](#-catégorie-f--systèmes-embarqués--optimisations-jetson-orin)
7. [Catégorie G : Limitations du projet & Travaux futurs](#-catégorie-g--limitations-du-projet--travaux-futurs)

---

## 🖥️ Catégorie A : Architecture système & Plateforme matérielle

### Q1 : Pourquoi avez-vous choisi le NVIDIA Jetson Orin Nano au lieu d'un Raspberry Pi standard ou d'un serveur cloud ?
> [!IMPORTANT]
> **Concepts clés :** Confidentialité du calcul Edge, boucles à faible latence et optimisation ARM.
* **Réponse :** « Une solution cloud est inappropriée pour les environnements industriels ou cliniques en raison des **réglementations strictes sur la confidentialité des données (RGPD/HIPAA)** et des contraintes de coût/fiabilité de la bande passante. Transmettre des flux vidéo haute résolution de travailleurs vers le cloud viole la confidentialité.
* Un Raspberry Pi est trop faible computationnellement pour gérer simultanément le décodage des trames de profondeur OAK-D, l'extraction des points clés de pose MediaPipe, les calculs d'angles articulaires à 12 axes, l'inférence MLP ErgoNet v2.0 et un serveur WebSocket multi-canaux en temps réel.
* Le **NVIDIA Jetson Orin Nano** fournit la plateforme Edge AI idéale. Il permet de garder toutes les données localement sur site, garantit un pipeline constant à faible latence de **8+ images par seconde**, et consomme moins de **15 W de puissance**. »

### Q2 : Comment la caméra Luxonis OAK-D s'intègre-t-elle dans votre système, et pourquoi une caméra de profondeur est-elle nécessaire plutôt qu'une webcam 2D standard ?
> [!NOTE]
> **Concepts clés :** Ambiguïté d'échelle en 2D, coordonnées 3D réelles et correspondance stéréo matérielle.
* **Réponse :** « Une caméra 2D standard perd toutes les informations de profondeur, introduisant une **ambiguïté d'échelle en perspective**. La **Luxonis OAK-D** dispose de caméras de profondeur stéréo aux côtés d'un capteur RGB haute résolution. En alignant la carte de profondeur sur la trame RGB au niveau matériel, nous mappons chaque landmark MediaPipe 2D $(x, y)$ directement à sa profondeur physique réelle $Z$ en utilisant un patch médian $3\times3$ autour de la coordonnée pixel de l'articulation. Cela donne de **vraies coordonnées 3D métriques $(X_{m}, Y_{m}, Z_{m})$** par rapport à l'objectif de la caméra, rendant nos calculs articulaires invariants à la distance et à l'échelle. »

---

## 🦴 Catégorie B : Vision par ordinateur & Détection de profondeur 3D

### Q3 : Comment extrayez-vous les points clés corporels 3D en temps réel, et comment gérez-vous le bruit/fluctuations de profondeur autour de ces points ?
* **Réponse :** « Nous utilisons un pipeline **MediaPipe Pose** optimisé pour extraire 33 landmarks corporels distincts. Cependant, les cartes de profondeur brutes sont très bruitées en raison de l'éclairage, des réflexions des vêtements et des ombres de correspondance stéréo.
* Notre `pose/estimator.py` utilise un **filtre médian spatial $3\times3$** centré autour de la coordonnée pixel $(x, y)$ de chaque articulation pour récupérer la profondeur.
* Temporellement, nous appliquons un filtre **EMA** avec un facteur de lissage $\alpha = 0,15$ :
  $$S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1}$$
  Cela supprime le bruit de coordonnées haute fréquence avec une latence négligeable (moins de 50 ms à 8 FPS). »

### Q4 : Les caméras OAK-D peuvent exécuter des réseaux de neurones sur leur VPU Myriad X interne. Pourquoi avez-vous choisi d'exécuter MediaPipe et ErgoNet sur le CPU Jetson ?
> [!TIP]
> **Concepts clés :** Flexibilité du pipeline CPU, vectorisation NEON ARM et extensibilité multi-caméra.
* **Réponse :** « Exécuter des pipelines sur le VPU Myriad X nécessite de compiler les graphes de modèles en fichiers `.blob` personnalisés, ce qui nous restreint à des versions plus anciennes et empêche les modifications dynamiques à l'exécution.
* En exécutant **MediaPipe** et notre **ErgoNet v2.0** sur le CPU Jetson, nous maintenons un contrôle total. Notre **ErgoNet v2.0** est implémenté comme un **MLP NumPy pur vectorisé**, utilisant les **jeux d'instructions ARM NEON** via OpenBLAS, permettant l'inférence en **moins de 8 ms** avec seulement 15 Mo de RAM. »

---

## 📐 Catégorie C : Biomécanique & Standards ergonomiques

### Q5 : Expliquez comment vous calculez un angle articulaire 3D (comme la Flexion du Cou ou du Tronc) à partir des coordonnées 3D retournées par votre pipeline de vision.
* **Réponse :** « Nous modélisons les segments corporels comme des vecteurs 3D. L'angle entre deux vecteurs 3D $\vec{u}$ et $\vec{v}$ est calculé avec la formule du produit scalaire :
  $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right)$$
* Pour les articulations complexes comme l'**Abduction de l'Épaule** ou la **Rotation du Tronc**, nous projetons les vecteurs sur des plans anatomiques spécifiques (plans sagittal, frontal ou transversal). »

### Q6 : Les scores RULA et REBA scintillent ou sautent rapidement aux conditions limites. Comment votre système résout-il ce problème ?
> [!IMPORTANT]
> **Concepts clés :** Logique d'hystérésis, filtrage EMA et persistance des transitions.
* **Réponse :** « Nous résolvons cela en deux couches :
  1. **Pré-lissage EMA ($\alpha = 0,15$)** : Lisse les angles articulaires bruts dans le temps.
  2. **Hystérésis & Logique de Persistance des Scores** : Un score RULA/REBA n'est autorisé à monter ou descendre que s'il reste stable pendant au moins **5 trames consécutives** (environ 600 ms). »

---

## 🧠 Catégorie D : Apprentissage automatique personnalisé (ErgoNet v2.0 & Moteur NumPy)

### Q7 : Pourquoi avez-vous entraîné un réseau de neurones personnalisé (ErgoNet v2.0) si vous avez déjà les tables de consultation officielles RULA et REBA ?
* **Réponse :** « RULA et REBA sont des standards de scoring fortement discrétisés et rigides développés dans les années 1990. **ErgoNet v2.0** sert de **compagnon de diagnostic clinique**. Il utilise un MLP à têtes multiples pour :
  1. Prédire un **Score de Risque continu (0,0 à 10,0)**.
  2. Effectuer une **Classification Multi-Tâches** pour prédire la localisation anatomique et classifier **18 différents Troubles Musculosquelettiques (TMS)** réels.
* L'IA ne remplace pas RULA/REBA ; elle travaille aux côtés d'eux. »

### Q8 : Comment le jeu de données d'entraînement pour ErgoNet v2.0 a-t-il été généré ?
> [!NOTE]
> **Concepts clés :** Limites cinématiques, génération bootstrap et validation Z-score.
* **Réponse :** « Les jeux de données ergonomiques annotés cliniquement sont pratiquement inexistants. Nous avons construit un **Générateur Bootstrap Synthétique Haute-Fidélité** (`ai/synthetic_gen.py`). Le simulateur échantillonne des angles articulaires dans toute la plage anatomique sous des **contraintes cinématiques strictes** (il ne peut pas générer de postures anatomiquement impossibles). Pour chaque posture échantillonnée, il calcule le squelette 3D, étiquette automatiquement via les règles mathématiques RULA/REBA et enrichit avec des **profils de contrainte asymétriques**. Cela génère **20 000+ échantillons propres** avec **97,14 % de précision d'entraînement** et **94,22 % de précision de validation**. »

### Q9 : Pourquoi ErgoNet v2.0 est-il implémenté en NumPy pur au lieu de TensorFlow, Keras ou PyTorch ?
* **Réponse :** « Déployer PyTorch ou TensorFlow sur un appareil ARM embarqué comme le Jetson Orin introduit des défis massifs :
  1. **Dépendances :** PyTorch/TF nécessitent des installations binaires lourdes, des correspondances de versions CUDA.
  2. **Empreinte mémoire :** PyTorch alloue ~1,5 à 2 Go de mémoire virtuelle. Notre MLP NumPy n'utilise que **15 Mo de RAM**.
  3. **Latence :** Une passe avant NumPy prend moins de **8 millisecondes**, le modèle charge en moins de **10 millisecondes**. La passe avant MLP est mathématiquement simple :
  $$Z^{[1]} = X \cdot W_1 + b_1, \quad A^{[1]} = \max(0, Z^{[1]}), \quad Y = A^{[1]} \cdot W_2 + b_2$$ »

---

## 🌐 Catégorie E : Systèmes Web & Pipeline de télémétrie dynamique

### Q10 : Comment votre système réalise-t-il la synchronisation en temps réel entre le backend Python et le tableau de bord web ?
> [!TIP]
> **Concepts clés :** WebSockets, streaming MJPEG à faible surcharge et tampons roulants.
* **Réponse :** « Le polling HTTP traditionnel crée une énorme surcharge réseau et de la latence. Au lieu de cela, nous utilisons des **WebSockets** via **Flask-SocketIO** (serveur) et **Socket.IO Client** (navigateur).
* Le gestionnaire de caméra fonctionne dans un thread d'arrière-plan à 8 Hz, emballant la télémétrie dans une charge utile JSON et émettant un événement `pose_update` sur le canal WebSocket avec **moins de 15 ms de latence de transport**. »

### Q11 : Expliquez comment fonctionne la visionneuse de squelette 3D Three.js sur la page `/3d`.
* **Réponse :** « La visionneuse Three.js rend une scène WebGL 3D interactive avec un squelette basé sur des maillages. Le backend diffuse un événement WebSocket `skeleton_3d` dédié contenant les landmarks 3D bruts de MediaPipe. Côté client, nous mappons ces landmarks à des cylindres et sphères représentant les os et articulations. »

---

## ⚙️ Catégorie F : Systèmes embarqués & Optimisations Jetson Orin

### Q12 : Quelles optimisations OS et matérielles spécifiques avez-vous implémentées pour assurer un fonctionnement fluide sur le Jetson Orin Nano ?
* **Réponse :** « Pour maximiser le débit et éliminer la limitation au niveau noyau :
  1. **Mode MAXN :** Activé via `sudo nvpmodel -m 0`. Active tous les cœurs CPU et lève les plafonds de consommation.
  2. **Jetson Clocks verrouillés :** `sudo jetson_clocks` verrouille les fréquences CPU/GPU à leur maximum absolu.
  3. **Assignation d'affinité CPU (`taskset`) :** La boucle temps réel est liée aux cœurs CPU 0–3 :
     ```bash
     taskset -c 0-3 python3 app.py
     ```
  4. **Gestion mémoire (`MALLOC_ARENA_MAX=2`) :** Empêche la fragmentation des arènes mémoire GLIBC sur ARM64. »

---

## ⚠️ Catégorie G : Limitations du projet & Travaux futurs

### Q13 : Quelles sont les principales limitations de votre implémentation actuelle et comment les adresseriez-vous en production ?
* **Réponse :** « Trois limitations principales ont été identifiées :
  1. **Suivi mono-sujet :** MediaPipe Pose est optimisé pour un sujet dominant. *Solution :* Implémenter **YOLOv8-Pose** ou le suivi deep SORT.
  2. **Occultation (blocage visuel) :** Si un travailleur passe derrière une machine, le suivi tombe. *Solution :* Implémentation d'une **matrice de calibration extrinsèque multi-caméra** pour fusionner les points clés de 2-3 caméras OAK-D.
  3. **Accélération TensorRT :** *Solution :* Nous avons écrit un script d'export ONNX (`ai/operation/export_onnx.py`). Dans le futur, nous pouvons compiler le modèle ONNX en un **moteur NVIDIA TensorRT** pour réduire la latence d'inférence à **< 1 ms**. »

---

## 🏆 Fiche récapitulative pour votre présentation de soutenance

| Sujet | Solution de votre projet | Impact / Métrique |
|---|---|---|
| **Vision par ordinateur** | MediaPipe + profondeur alignée OAK-D | Squelette 3D invariant à la distance et à l'échelle |
| **Scoring biomécanique** | Tables exactes RULA & REBA | Transparence totale du score clinique |
| **Stabilité du signal** | Filtre EMA ($\alpha=0,15$) + Hystérésis 5 trames | Zéro scintillement, graphiques de tendance fluides |
| **Apprentissage automatique** | ErgoNet v2.0 MLP Multi-Tâches | **97,14 % de précision**, prédiction de risque continue |
| **Surcharge des frameworks** | Moteur opérationnel NumPy pur | **~8 ms de latence**, **15 Mo de RAM** |
| **Optimisation embarquée** | `taskset` + `nvpmodel` MAXN | Pipeline temps réel 8+ FPS |
| **Interface utilisateur** | Flask + Socket.IO + Three.js | UI glassmorphique temps réel + vue 3D interactive |
| **Reporting clinique** | Compilateur de rapports PDF automatique | PDF instantané et téléchargeable avec graphiques de tendance |

---

*Préparé par l'Équipe de Développement ErgoVision pour soutenir votre succès ! Bonne chance pour votre soutenance ! 🚀*
