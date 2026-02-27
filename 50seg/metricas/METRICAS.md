# Métricas de Clasificación SMAW - 50seg

**Fecha de evaluación:** 2026-02-26 21:17:05

**Configuración:**
- Duración de segmento: 50.0s
- Número de muestras (blind): 59
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.8305 | 0.8286 |
| Electrode Type | 0.8644 | 0.8772 |
| Current Type | 0.9661 | 0.9612 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.8305
- **Macro F1-Score:** 0.8286

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 14 | 1 | 1 |
| **Placa_3mm** | 1 | 13 | 1 |
| **Placa_6mm** | 1 | 5 | 22 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.8750 | 0.8750 | 0.8750 | 16 |
| Placa_3mm | 0.6842 | 0.8667 | 0.7647 | 15 |
| Placa_6mm | 0.9167 | 0.7857 | 0.8462 | 28 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8644
- **Macro F1-Score:** 0.8772

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 7 | 0 | 0 | 0 |
| **E6011** | 1 | 16 | 0 | 0 |
| **E6013** | 0 | 0 | 14 | 0 |
| **E7018** | 0 | 5 | 2 | 14 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.8750 | 1.0000 | 0.9333 | 7 |
| E6011 | 0.7619 | 0.9412 | 0.8421 | 17 |
| E6013 | 0.8750 | 1.0000 | 0.9333 | 14 |
| E7018 | 1.0000 | 0.6667 | 0.8000 | 21 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9661
- **Macro F1-Score:** 0.9612

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 18 | 0 |
| **DC** | 2 | 39 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9000 | 1.0000 | 0.9474 | 18 |
| DC | 1.0000 | 0.9512 | 0.9750 | 41 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
