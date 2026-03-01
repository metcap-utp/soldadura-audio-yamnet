# YAMNet - Resultados

## Configuración
- **Duración:** 5 segundos
- **K-folds:** 10
- **Overlap:** 0.5

---

## Métricas por Modelo (Blind Set)

| Modelo | Plate Acc | Electrode Acc | Current Acc | Exact Match | Hamming |
|--------|-----------|---------------|-------------|-------------|---------|
| **xvector** | 0.7203 | 0.8002 | 0.9211 | 0.6593 | 0.8139 |
| ecapa | 0.6761 | 0.8086 | 0.8780 | 0.6151 | 0.7876 |
| feedforward | 0.6898 | 0.8128 | 0.9222 | 0.6383 | 0.8083 |

---

## Mejor Modelo: xvector

| Métrica | Valor |
|---------|-------|
| Exact Match | **0.6593** |
| Hamming Accuracy | **0.8139** |
| Plate Accuracy | 0.7203 |
| Electrode Accuracy | 0.8002 |
| Current Accuracy | 0.9211 |

---

## Figuras

### Accuracy por Duración
![Accuracy por duración](graficas/accuracy_duracion_blind_set.png)

### Métricas Globales
![Métricas globales](graficas/metricas_globales_blind_set.png)

### Comparación de Modelos
![Backbones](graficas/backbones_blind_set.png)

### Matriz de Confusión - xvector (Electrode)
![Matriz xvector electrode](graficas/matriz_confusion_yamnet_xvector_electrode.png)

---

## Conclusiones

1. **xvector** es el mejor modelo con 65.93% exact match
2. **Current** es la tarea más fácil (>87% accuracy)
3. **Plate** es la tarea más difícil (~68-72% accuracy)
