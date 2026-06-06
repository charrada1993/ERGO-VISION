# Guide d'export ONNX pour ErgoNet v2.0

*Dernière mise à jour : 2026-05-11*

> **Note :** ErgoNet v2.0 est actuellement déployé comme un **modèle `.pkl` NumPy pur** pour une inférence sans dépendance sur le Jetson Orin. Ce guide documente la voie ONNX pour une accélération TensorRT future.

---

## Qu'est-ce qu'ONNX ?

**ONNX (Open Neural Network Exchange)** est un format open-source standardisé pour représenter les modèles ML — un « traducteur universel » pour l'IA. Il permet à un modèle entraîné dans un framework (PyTorch, TensorFlow) d'être exécuté dans un autre (ONNX Runtime, TensorRT).

---

## Pourquoi ONNX pour ERGO-VISION ?

| Raison | Détails |
|---|---|
| **Indépendance du framework** | Entraîner en PyTorch sur un poste de travail, déployer sur Jetson sans PyTorch |
| **Pont TensorRT** | TensorRT ne peut pas lire directement les fichiers `.pt` ou `.pth` — ONNX est le format intermédiaire requis |
| **Accélération matérielle** | ONNX → TensorRT mappe les opérations du modèle directement sur les cœurs CUDA et DLA Jetson |
| **Portabilité** | Les fichiers `.onnx` fonctionnent sur tout matériel supporté par ONNX Runtime |

---

## Modèle actuel : NumPy `.pkl` (Actif)

ErgoNet v2.0 fonctionne actuellement comme un modèle NumPy figé :

```
ai/models/ergo_net_v2.pkl
```

| Propriété | Valeur |
|---|---|
| Format | Python pickle (`.pkl`) |
| Inférence | Multiplication matricielle NumPy pure |
| Latence | ~8 ms (CPU Jetson) |
| Dépendances | `numpy` uniquement |

---

## Voie d'export ONNX (Futur)

Pour exporter ErgoNet v2.0 vers ONNX pour l'accélération TensorRT :

### Étape 1 : Reconstruire en PyTorch
```python
import torch
import torch.nn as nn

class ErgoNetTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

### Étape 2 : Charger les poids depuis `.pkl`
```python
import pickle, numpy as np, torch

with open('ai/models/ergo_net_v2.pkl', 'rb') as f:
    state = pickle.load(f)

model = ErgoNetTorch()
model.fc1.weight.data = torch.tensor(state['W1'].T, dtype=torch.float32)
model.fc1.bias.data   = torch.tensor(state['b1'].squeeze(), dtype=torch.float32)
model.fc2.weight.data = torch.tensor(state['W2'].T, dtype=torch.float32)
model.fc2.bias.data   = torch.tensor(state['b2'].squeeze(), dtype=torch.float32)
```

### Étape 3 : Exporter vers ONNX
```python
dummy_input = torch.randn(1, 12)
torch.onnx.export(
    model, dummy_input,
    'ai/models/ergo_net_v2.onnx',
    input_names=['joint_angles'],
    output_names=['risk_score', 'severity_code', 'location_code', 'condition_code'],
    opset_version=17
)
```

### Étape 4 : Compiler avec TensorRT
```bash
trtexec --onnx=ai/models/ergo_net_v2.onnx \
        --saveEngine=ai/models/ergo_net_v2.engine \
        --fp16
```

### Étape 5 : Exécuter via le script opérationnel
```bash
python3 ai/operation/export_onnx.py
```

Cela produira `ai/models/ergo_net_v2.onnx`, prêt pour la compilation TensorRT sur le Jetson.

---

## Paramètres clés ONNX

| Paramètre | Valeur | Description |
|---|---|---|
| `input_names` | `['joint_angles']` | Vecteur d'angles normalisés à 12 éléments |
| `output_names` | 4 têtes diagnostiques | risque, sévérité, localisation, condition |
| `opset_version` | 17 | Supporte toutes les opérations dans ErgoNet v2.0 |

---

*Documenté par l'Équipe IA ErgoVision · 2026*
