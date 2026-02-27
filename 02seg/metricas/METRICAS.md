# Métricas de Clasificación SMAW - 2seg

**Fecha de evaluación:** 2026-02-26 17:38:38

**Configuración:**
- Duración de segmento: 2.0s
- Número de muestras (blind): 2465
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.6219 | 0.6295 |
| Electrode Type | 0.7071 | 0.6831 |
| Current Type | 0.8365 | 0.8314 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.6219
- **Macro F1-Score:** 0.6295

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 520 | 35 | 160 |
| **Placa_3mm** | 53 | 516 | 78 |
| **Placa_6mm** | 377 | 229 | 497 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.5474 | 0.7273 | 0.6246 | 715 |
| Placa_3mm | 0.6615 | 0.7975 | 0.7232 | 647 |
| Placa_6mm | 0.6762 | 0.4506 | 0.5408 | 1103 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.7071
- **Macro F1-Score:** 0.6831

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 197 | 54 | 44 | 12 |
| **E6011** | 68 | 671 | 51 | 12 |
| **E6013** | 34 | 63 | 480 | 56 |
| **E7018** | 99 | 124 | 105 | 395 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.4950 | 0.6417 | 0.5589 | 307 |
| E6011 | 0.7357 | 0.8367 | 0.7830 | 802 |
| E6013 | 0.7059 | 0.7583 | 0.7312 | 633 |
| E7018 | 0.8316 | 0.5463 | 0.6594 | 723 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.8365
- **Macro F1-Score:** 0.8314

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 816 | 39 |
| **DC** | 364 | 1246 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.6915 | 0.9544 | 0.8020 | 855 |
| DC | 0.9696 | 0.7739 | 0.8608 | 1610 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
