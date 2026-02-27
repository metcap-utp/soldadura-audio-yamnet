# Métricas de Clasificación SMAW - 10seg

**Fecha de evaluación:** 2026-02-26 19:43:17

**Configuración:**
- Duración de segmento: 10.0s
- Número de muestras (blind): 447
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.6868 | 0.6954 |
| Electrode Type | 0.8345 | 0.8130 |
| Current Type | 0.9329 | 0.9283 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.6868
- **Macro F1-Score:** 0.6954

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 93 | 3 | 33 |
| **Placa_3mm** | 2 | 103 | 9 |
| **Placa_6mm** | 54 | 39 | 111 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6242 | 0.7209 | 0.6691 | 129 |
| Placa_3mm | 0.7103 | 0.9035 | 0.7954 | 114 |
| Placa_6mm | 0.7255 | 0.5441 | 0.6218 | 204 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8345
- **Macro F1-Score:** 0.8130

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 49 | 4 | 0 | 2 |
| **E6011** | 5 | 140 | 1 | 0 |
| **E6013** | 7 | 3 | 96 | 5 |
| **E7018** | 25 | 3 | 19 | 88 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.5698 | 0.8909 | 0.6950 | 55 |
| E6011 | 0.9333 | 0.9589 | 0.9459 | 146 |
| E6013 | 0.8276 | 0.8649 | 0.8458 | 111 |
| E7018 | 0.9263 | 0.6519 | 0.7652 | 135 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9329
- **Macro F1-Score:** 0.9283

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 152 | 6 |
| **DC** | 24 | 265 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8636 | 0.9620 | 0.9102 | 158 |
| DC | 0.9779 | 0.9170 | 0.9464 | 289 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
