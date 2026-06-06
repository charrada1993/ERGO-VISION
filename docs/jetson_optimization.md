# Optimisation Jetson Orin : Guide d'ajustement matériel

*Dernière mise à jour : 2026-05-11*

Le système ERGO-VISION est optimisé pour le **NVIDIA Jetson Orin Nano (reComputer J3011, 8 Go)**. Pour atteindre des performances en temps réel (estimation de pose + inférence IA + streaming vidéo simultanément), plusieurs optimisations matérielles et logicielles sont appliquées.

---

## 1. Environnement système

| Composant | Détails |
|---|---|
| **Appareil** | NVIDIA Jetson Orin Nano (reComputer J3011) |
| **RAM** | 8 Go LPDDR5 |
| **OS** | Ubuntu 22.04 (JetPack 6.x) |
| **Python** | 3.10 (environnement virtuel `oak_env`) |
| **Caméra** | OAK-D (USB 3.1 Gen 2) |

---

## 2. Gestion de l'alimentation & des horloges

```bash
sudo nvpmodel -m 0   # Mode performances maximales 15W
sudo jetson_clocks   # Verrouiller CPU/GPU à la fréquence maximale
```

| Paramètre | Effet |
|---|---|
| **NVPModel 0** | Tous les cœurs CPU actifs, fréquence maximale |
| **Jetson Clocks** | Élimine les pics de latence causés par la mise à l'échelle dynamique des fréquences |

---

## 3. Efficacité mémoire

```bash
export MALLOC_ARENA_MAX=2
```

| Optimisation | Détails |
|---|---|
| `MALLOC_ARENA_MAX=2` | Empêche la fragmentation des arènes mémoire GLIBC sur ARM |
| Swap ZRAM 4 Go | Gère les pics de RAM MediaPipe avec élégance |
| IA NumPy uniquement | ErgoNet v2.0 utilise ~15 Mo de RAM vs. ~2 Go pour PyTorch |

---

## 4. Vitesse d'inférence IA

ErgoNet v2.0 atteint **< 8 ms** de latence d'inférence via :

- **OpenBLAS** — multiplication matricielle vectorisée ARM v8.2 NEON
- **Couche cachée unique (512)** — tailles de matrices minimales
- **Statistiques de normalisation pré-chargées** — zéro surcharge par trame à l'inférence
- **Pas de rétropropagation** — le modèle opérationnel effectue uniquement la passe avant

Pipeline d'inférence complet par trame :
```
Angles → Normalisation Z-score → Passe avant → Dénormalisation → Émission Socket.IO  (~8–12 ms total)
```

---

## 5. Paramètres du pipeline caméra

| Paramètre | Valeur | Raison |
|---|---|---|
| `setBlocking(False)` + `setQueueSize(1)` | Non bloquant, taille 1 | Empêche le blocage du pipeline ; élimine les trames périmées |
| `tryGetAll()[-1]` | Côté hôte | Traite toujours la trame la plus fraîche |
| Résolution RGB | 1280×720 @ 30 FPS | Équilibre précision et bande passante USB |

---

## 6. Lancement du système

```bash
source ~/oak_env/bin/activate
cd ~/ERGO-VISION
python3 app.py
```

Pour l'isolation des cœurs CPU (optionnel) :
```bash
taskset -c 0-3 python3 app.py
```

---

*Documenté par l'Équipe IA ErgoVision · 2026*
