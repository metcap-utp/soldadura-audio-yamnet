# Métricas de Clasificación SMAW - 30seg

**Fecha de evaluación:** 2026-02-26 20:53:05

**Configuración:**
- Duración de segmento: 30.0s
- Número de muestras (blind): 113
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7168 | 0.7214 |
| Electrode Type | 0.8938 | 0.8852 |
| Current Type | 0.9381 | 0.9340 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7168
- **Macro F1-Score:** 0.7214

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 23 | 2 | 5 |
| **Placa_3mm** | 0 | 26 | 3 |
| **Placa_6mm** | 11 | 11 | 32 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6765 | 0.7667 | 0.7188 | 30 |
| Placa_3mm | 0.6667 | 0.8966 | 0.7647 | 29 |
| Placa_6mm | 0.8000 | 0.5926 | 0.6809 | 54 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8938
- **Macro F1-Score:** 0.8852

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 12 | 1 | 0 | 1 |
| **E6011** | 0 | 36 | 0 | 1 |
| **E6013** | 1 | 0 | 22 | 1 |
| **E7018** | 1 | 1 | 5 | 31 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.8571 | 0.8571 | 0.8571 | 14 |
| E6011 | 0.9474 | 0.9730 | 0.9600 | 37 |
| E6013 | 0.8148 | 0.9167 | 0.8627 | 24 |
| E7018 | 0.9118 | 0.8158 | 0.8611 | 38 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9381
- **Macro F1-Score:** 0.9340

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 39 | 1 |
| **DC** | 6 | 67 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8667 | 0.9750 | 0.9176 | 40 |
| DC | 0.9853 | 0.9178 | 0.9504 | 73 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
