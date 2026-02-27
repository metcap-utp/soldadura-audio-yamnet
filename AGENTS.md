# AGENTS.md - Guía para Agentes de Código

## Información del Proyecto

**Proyecto**: Clasificación de Audio SMAW (Soldadura por Arco Eléctrico)  
**Lenguaje**: Python 3.10+  
**Framework**: PyTorch + Librosa + Scikit-learn + TensorFlow Hub (YAMNet)  
**Idioma**: Español (código y documentación)

## Diferencias con VGGish Backbone

- **Embedding dimensions**: YAMNet produce 1024 dimensiones (vs 128 de VGGish)
- **Modelo**: YAMNet usa una arquitectura diferente, embeddings de la capa interna
- **Audio**: Mismo formato (16kHz, mono)

## Restricción Crítica: Ejecución Secuencial

**IMPORTANTE**: Las tareas de entrenamiento e inferencia consumen recursos significativos de GPU y CPU.
Solo se puede ejecutar **una tarea a la vez** - deben ser **secuenciales**, no paralelas.

Esto incluye:
- `entrenar_xvector.py`, `entrenar_ecapa.py`, `entrenar_feedforward.py`
- `inferir.py --evaluar`
- `generar_splits.py`

No lanzar múltiples procesos de entrenamiento/inferencia simultáneamente.

## Comandos de Desarrollo

### Generación de Datos

```bash
python generar_splits.py --duration 5 --overlap 0.5
python generar_splits.py --duration 10 --overlap 0.0
```

### Entrenamiento

```bash
# X-Vector
python entrenar_xvector.py --duration 5 --overlap 0.5 --k-folds 10

# ECAPA-TDNN
python entrenar_ecapa.py --duration 5 --overlap 0.5 --k-folds 10

# FeedForward
python entrenar_feedforward.py --duration 5 --overlap 0.5 --k-folds 10
```

### Inferencia/Evaluación

```bash
# Evaluar ensemble en conjunto blind
python inferir.py --duration 5 --overlap 0.5 --k-folds 10 --model xvector --evaluar

# Predicción de un archivo específico
python inferir.py --duration 5 --overlap 0.5 --audio ruta/archivo.wav

# Predicciones aleatorias
python inferir.py --duration 5 --overlap 0.5 --n 10
```

### Scripts Batch

```bash
# Entrenar y evaluar todos los modelos
./entrenar_todos.sh                              # Todo (k=10, overlap=0.5)
./entrenar_todos.sh --duration 5 --model xvector # Solo xvector, 5seg
./entrenar_todos.sh --dry-run                    # Solo mostrar qué se haría
./entrenar_todos.sh --skip-train                 # Solo evaluación
./entrenar_todos.sh --skip-eval                  # Solo entrenamiento
```

### Linting y Formato

```bash
# Instalar herramientas
pip install black ruff mypy

# Formatear código (línea: 100 caracteres)
black --line-length 100 *.py scripts/ utils/

# Linting
ruff check *.py scripts/ utils/

# Type checking
mypy modelo_ecapa.py modelo_xvector.py utils/
```

## Estructura de Salida

- `{N}seg/modelos/{arquitectura}/k{K}_overlap_{ratio}/` - Modelos `.pth`
- `{N}seg/resultados.json` - Métricas de entrenamiento (acumulativo)
- `{N}seg/inferencia.json` - Métricas de evaluación (acumulativo)
- `{N}seg/metricas/METRICAS.md` - Documento Markdown con matrices de confusión
- `{N}seg/embeddings_cache/yamnet_embeddings_*.pkl` - Cache de embeddings

## Notas Importantes

1. **No versionar**: Archivos `.pth`, `.keras`, `.pkl`, `.wav`, `.mp3`
2. **YAMNet**: Modelo pre-entrenado de Google se descarga automáticamente desde TensorFlow Hub
3. **Argumentos CLI**: `--duration` (1,2,5,10,20,30,50), `--overlap` (0.0-0.75), `--k-folds` (3-20)
4. **Arquitecturas**: xvector, ecapa_tdnn, feedforward
5. **Audio**: Usa el directorio de audio de `vggish-backbone/` para evitar duplicar archivos
