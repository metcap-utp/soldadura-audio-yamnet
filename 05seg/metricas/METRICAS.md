# Métricas de Clasificación SMAW - 5seg

**Fecha de evaluación:** 2026-02-27 08:24:55

**Configuración:**
- Duración de segmento: 5.0s
- Número de muestras (blind): 951
- Número de modelos (ensemble): 20
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7014 | 0.7086 |
| Electrode Type | 0.8107 | 0.7884 |
| Current Type | 0.9117 | 0.9066 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7014
- **Macro F1-Score:** 0.7086

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 197 | 8 | 70 |
| **Placa_3mm** | 7 | 225 | 15 |
| **Placa_6mm** | 107 | 77 | 245 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6334 | 0.7164 | 0.6724 | 275 |
| Placa_3mm | 0.7258 | 0.9109 | 0.8079 | 247 |
| Placa_6mm | 0.7424 | 0.5711 | 0.6456 | 429 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8107
- **Macro F1-Score:** 0.7884

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 98 | 13 | 4 | 3 |
| **E6011** | 12 | 292 | 4 | 2 |
| **E6013** | 12 | 13 | 207 | 9 |
| **E7018** | 52 | 18 | 38 | 174 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.5632 | 0.8305 | 0.6712 | 118 |
| E6011 | 0.8690 | 0.9419 | 0.9040 | 310 |
| E6013 | 0.8182 | 0.8589 | 0.8381 | 241 |
| E7018 | 0.9255 | 0.6170 | 0.7404 | 282 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9117
- **Macro F1-Score:** 0.9066

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 323 | 7 |
| **DC** | 77 | 544 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8075 | 0.9788 | 0.8849 | 330 |
| DC | 0.9873 | 0.8760 | 0.9283 | 621 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
