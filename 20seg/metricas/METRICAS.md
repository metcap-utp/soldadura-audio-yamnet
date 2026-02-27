# Métricas de Clasificación SMAW - 20seg

**Fecha de evaluación:** 2026-02-26 20:21:36

**Configuración:**
- Duración de segmento: 20.0s
- Número de muestras (blind): 199
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7588 | 0.7631 |
| Electrode Type | 0.8794 | 0.8647 |
| Current Type | 0.9497 | 0.9462 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7588
- **Macro F1-Score:** 0.7631

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 41 | 2 | 13 |
| **Placa_3mm** | 1 | 45 | 3 |
| **Placa_6mm** | 17 | 12 | 65 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6949 | 0.7321 | 0.7130 | 56 |
| Placa_3mm | 0.7627 | 0.9184 | 0.8333 | 49 |
| Placa_6mm | 0.8025 | 0.6915 | 0.7429 | 94 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8794
- **Macro F1-Score:** 0.8647

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 23 | 2 | 0 | 0 |
| **E6011** | 2 | 64 | 0 | 0 |
| **E6013** | 2 | 2 | 40 | 1 |
| **E7018** | 6 | 1 | 8 | 48 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6970 | 0.9200 | 0.7931 | 25 |
| E6011 | 0.9275 | 0.9697 | 0.9481 | 66 |
| E6013 | 0.8333 | 0.8889 | 0.8602 | 45 |
| E7018 | 0.9796 | 0.7619 | 0.8571 | 63 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9497
- **Macro F1-Score:** 0.9462

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 69 | 1 |
| **DC** | 9 | 120 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8846 | 0.9857 | 0.9324 | 70 |
| DC | 0.9917 | 0.9302 | 0.9600 | 129 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
