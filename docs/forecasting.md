# Temporal Forecasting: 10-Day Risk Prediction

*Last updated: 2026-05-11*

One of the most advanced features of the ERGO-VISION AI is its ability to **predict future ergonomic risks** based on a worker's historical posture data. This module operates on top of ErgoNet v2.0's live `risk_score` output and extends it forward in time using temporal pattern analysis.

---

## 1. Overview

| Property | Value |
|---|---|
| **Method** | Temporal Moving Window (TMW) + LSTM projection |
| **Input Window** | Last 7 days of session logs |
| **Forecast Horizon** | 10 days |
| **Data Source** | `ai/data/training_log.json` + live `risk_score` stream |

---

## 2. The Time-Series Logic

The forecaster does not look at a single snapshot. It analyzes the **trend trajectory** of risk accumulation across multiple work sessions.

### Feature Extraction (7-Day Window)

The model processes three key signals from session history:

| Signal | Description |
|---|---|
| **Cumulative Load** | Total time spent in high-risk zones (RULA 5+, risk_score > 7.0) per day |
| **Static Fatigue Index** | Duration any single joint stays in a high-flexion state without movement (a primary TMS risk factor) |
| **Diurnal Variance** | Time-of-day risk distribution — e.g., does the worker's posture consistently worsen after 3:00 PM? |

---

## 3. 10-Day Projection (LSTM)

The forecasting engine fits a **growth/decay curve** to the 7-day history and projects it 10 days forward.

### Growth Phase
If the AI detects an upward trend (e.g., `Neck_Flexion` risk increasing by 5% per day), it extrapolates:
- **Day 3–4 prediction**: Severity escalates to `HIGH` (severity_code = 3).
- **Day 7 prediction**: `CRITICAL` threshold reached (severity_code = 4) unless intervention occurs.

### Decay Phase
If the worker has taken corrective action (posture improved, lower risk readings this week):
- The model predicts **risk reduction** and marks the trend as improving.
- A "Recovery Trajectory" badge is shown on the dashboard.

### Anomaly Forecasting
Beyond trend projection, the model identifies **upcoming anomaly types** based on repetitive patterns:
- Repeated extreme wrist deviation → predicts high `condition_code` for Carpal Tunnel / Tendinitis.
- Prolonged unilateral shoulder elevation → predicts Rotator Cuff risk on the dominant side.

---

## 4. Preventive Action Framework

The goal of 10-day forecasting is to **change the future**. The dashboard presents:

| Forecast Indicator | Action |
|---|---|
| 🟢 **Improving** | No action required — continue current habits |
| 🟡 **Stable / Monitoring** | Check joint loading patterns this week |
| 🔶 **Increasing Risk** | Ergonomic workstation adjustment recommended |
| 🔴 **High-Risk Trajectory** | Immediate intervention: physiotherapy referral suggested |

By seeing a "High Risk" prediction for next Tuesday, a worker or ergonomist can adjust monitor height, chair settings, or task rotation **before the injury develops**.

---

## 5. Data Flow

```
Live Session (Socket.IO)
        │
        ▼
ErgoNet v2.0 → risk_score (0.0–10.0)
        │
        ▼
Session Logger (CSV) → daily aggregation
        │
        ▼
7-Day Window Extractor → feature vector
        │
        ▼
LSTM / TMW Projector → 10-day forecast
        │
        ▼
Dashboard /ai page → visual forecast chart
```

---

*Documented by ErgoVision AI Team · 2026*
