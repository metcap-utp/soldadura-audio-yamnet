# Métricas de Clasificación SMAW - 1seg

**Fecha de evaluación:** 2026-02-26 15:54:59

**Configuración:**
- Duración de segmento: 1.0s
- Número de muestras (blind): 4988
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.5357 | 0.5404 |
| Electrode Type | 0.5764 | 0.5466 |
| Current Type | 0.7291 | 0.7280 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.5357
- **Macro F1-Score:** 0.5404

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 903 | 184 | 362 |
| **Placa_3mm** | 176 | 920 | 217 |
| **Placa_6mm** | 785 | 592 | 849 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.4844 | 0.6232 | 0.5451 | 1449 |
| Placa_3mm | 0.5425 | 0.7007 | 0.6115 | 1313 |
| Placa_6mm | 0.5945 | 0.3814 | 0.4647 | 2226 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.5764
- **Macro F1-Score:** 0.5466

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 291 | 165 | 139 | 28 |
| **E6011** | 177 | 1227 | 189 | 31 |
| **E6013** | 90 | 262 | 856 | 77 |
| **E7018** | 118 | 503 | 334 | 501 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.4305 | 0.4671 | 0.4480 | 623 |
| E6011 | 0.5688 | 0.7555 | 0.6490 | 1624 |
| E6013 | 0.5639 | 0.6661 | 0.6108 | 1285 |
| E7018 | 0.7865 | 0.3441 | 0.4787 | 1456 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.7291
- **Macro F1-Score:** 0.7280

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 1655 | 73 |
| **DC** | 1278 | 1982 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.5643 | 0.9578 | 0.7101 | 1728 |
| DC | 0.9645 | 0.6080 | 0.7458 | 3260 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
