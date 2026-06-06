# Prévision temporelle : Prédiction de risque sur 10 jours

*Dernière mise à jour : 2026-05-11*

L'une des fonctionnalités les plus avancées de l'IA ERGO-VISION est sa capacité à **prédire les risques ergonomiques futurs** basés sur les données de posture historiques d'un travailleur. Ce module fonctionne sur la sortie `risk_score` en direct d'ErgoNet v2.0 et l'étend dans le temps à l'aide d'une analyse de patterns temporels.

---

## 1. Aperçu

| Propriété | Valeur |
|---|---|
| **Méthode** | Fenêtre Mobile Temporelle (TMW) + projection LSTM |
| **Fenêtre d'entrée** | 7 derniers jours de journaux de session |
| **Horizon de prévision** | 10 jours |
| **Source de données** | `ai/data/training_log.json` + flux `risk_score` en direct |

---

## 2. La logique des séries temporelles

Le prévisionneur ne regarde pas un seul instantané. Il analyse la **trajectoire de tendance** de l'accumulation de risques sur plusieurs sessions de travail.

### Extraction de caractéristiques (Fenêtre de 7 jours)

Le modèle traite trois signaux clés de l'historique des sessions :

| Signal | Description |
|---|---|
| **Charge cumulée** | Temps total passé en zones à haut risque (RULA 5+, risk_score > 7,0) par jour |
| **Indice de fatigue statique** | Durée pendant laquelle une seule articulation reste en état de haute flexion sans mouvement (un facteur de risque TMS primaire) |
| **Variance diurne** | Distribution du risque selon l'heure de la journée — ex., la posture du travailleur se dégrade-t-elle systématiquement après 15h00 ? |

---

## 3. Projection sur 10 jours (LSTM)

Le moteur de prévision ajuste une **courbe de croissance/décroissance** sur l'historique de 7 jours et la projette 10 jours en avant.

### Phase de croissance
Si l'IA détecte une tendance à la hausse (ex., risque `Neck_Flexion` augmentant de 5 % par jour) :
- **Prédiction jours 3–4** : La sévérité passe à `ÉLEVÉ` (severity_code = 3).
- **Prédiction jour 7** : Seuil `CRITIQUE` atteint (severity_code = 4) sans intervention.

### Phase de décroissance
Si le travailleur a pris des mesures correctives (posture améliorée, lectures de risque plus faibles cette semaine) :
- Le modèle prédit une **réduction du risque** et marque la tendance comme s'améliorant.
- Un badge « Trajectoire de récupération » est affiché sur le tableau de bord.

### Prévision d'anomalies
Au-delà de la projection de tendance, le modèle identifie les **types d'anomalies à venir** basés sur les patterns répétitifs :
- Déviation extrême répétée du poignet → prédit un `condition_code` élevé pour Canal Carpien / Tendinite.
- Élévation unilatérale prolongée de l'épaule → prédit un risque de Coiffe des Rotateurs du côté dominant.

---

## 4. Cadre d'action préventive

L'objectif de la prévision sur 10 jours est de **changer l'avenir**. Le tableau de bord présente :

| Indicateur de prévision | Action |
|---|---|
| 🟢 **En amélioration** | Aucune action requise — continuer les habitudes actuelles |
| 🟡 **Stable / Surveillance** | Vérifier les patterns de charge articulaire cette semaine |
| 🔶 **Risque croissant** | Ajustement ergonomique du poste de travail recommandé |
| 🔴 **Trajectoire à haut risque** | Intervention immédiate : référence en physiothérapie suggérée |

En voyant une prédiction « Haut risque » pour mardi prochain, un travailleur ou un ergonome peut ajuster la hauteur du moniteur, les réglages de la chaise ou la rotation des tâches **avant que la blessure ne se développe**.

---

## 5. Flux de données

```
Session en direct (Socket.IO)
        │
        ▼
ErgoNet v2.0 → risk_score (0,0–10,0)
        │
        ▼
Journaliseur de session (CSV) → agrégation quotidienne
        │
        ▼
Extracteur de fenêtre 7 jours → vecteur de caractéristiques
        │
        ▼
Projecteur LSTM / TMW → prévision 10 jours
        │
        ▼
Tableau de bord page /ai → graphique de prévision visuel
```

---

*Documenté par l'Équipe IA ErgoVision · 2026*
