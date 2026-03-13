# AGENTS.md - Guía para Agentes de Código

## Información del Proyecto

**Proyecto**: Clasificación de Audio SMAW (Soldadura por Arco Eléctrico)  
**Lenguaje**: Python 3.10+  
**Framework**: PyTorch + Librosa + Scikit-learn + TensorFlow Hub (YAMNet)  
**Idioma**: Español (código y documentación)

## Diferencias con VGGish Approach

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

## Archivos de Log

Todos los scripts de entrenamiento e inferencia generan archivos de log automáticamente en la carpeta `logs/`.

### Localización de Logs

- **Entrenamiento**: `logs/entrenar_[arquitectura]_[duracion]seg_[timestamp].log`
  - Ejemplo: `logs/entrenar_ecapa_05seg_20250228_143000.log`
- **Inferencia**: `logs/inferir_[duracion]seg_[modelo]_[timestamp].log`
  - Ejemplo: `logs/inferir_05seg_xvector_20250228_150000.log`

### Formato de Timestamp

Los archivos de log usan el formato `YYYYMMDD_HHMMSS`:

- `YYYY`: Año (2025)
- `MM`: Mes (01-12)
- `DD`: Día (01-31)
- `HH`: Hora (00-23)
- `MM`: Minuto (00-59)
- `SS`: Segundo (00-59)

### Contenido de Logs

Los archivos de log contienen:

- Todos los prints de la ejecución del script
- Métricas de entrenamiento (loss, accuracy por fold)
- Tiempos de ejecución (total, por fold, extracción de YAMNet)
- Información de enfoque utilizado (vggish, yamnet, spectral-mfcc)
- Cualquier error o warning durante la ejecución

### Gestión de Logs

Los archivos .log están excluidos de control de versión (`.gitignore`). Para limpiar logs antiguos:

```bash
# Eliminar todos los logs
rm logs/*.log

# Eliminar logs más viejos que 30 días
find logs/ -name "*.log" -mtime +30 -delete
```

### Ejemplo de Lectura de Logs

```bash
# Ver último log de entrenamiento (últimas 50 líneas)
tail -n 50 logs/entrenar_ecapa_05seg_*.log

# Buscar errores en los logs
grep -i "error\|exception" logs/*.log

# Ver el log de una ejecución específica
cat logs/entrenar_xvector_10seg_20250228_143000.log
```

## Estructura de Salida

- `{N}seg/modelos/{arquitectura}/k{K}_overlap_{ratio}/` - Modelos `.pth`
- `{N}seg/resultados.json` - Métricas de entrenamiento (acumulativo)
- `{N}seg/inferencia.json` - Métricas de evaluación (acumulativo)
- `{N}seg/metricas/METRICAS.md` - Documento Markdown con matrices de confusión
- `{N}seg/embeddings_cache/yamnet_embeddings_*.pkl` - Cache de embeddings

## Estructura del Proyecto

- `models/` - Definiciones de modelos (modelo_xvector.py, modelo_ecapa.py, modelo_feedforward.py)
- `logs/` - Archivos de log de entrenamiento e inferencia
- `utils/` - Utilidades (audio_utils.py, timing.py, logging_utils.py)
- `{N}seg/` - Datos y resultados por duración de segmento

### Imports de Proyecto

```python
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from models.modelo_xvector import SMAWXVectorModel
from utils.audio_utils import load_audio_segment
```

## Notas Importantes

1. **No versionar**: Archivos `.pth`, `.keras`, `.pkl`, `.wav`, `.mp3`
2. **YAMNet**: Modelo pre-entrenado de Google se descarga automáticamente desde TensorFlow Hub
3. **Argumentos CLI**: `--duration` (1,2,5,10,20,30,50), `--overlap` (0.0-0.75), `--k-folds` (3-20)
4. **Arquitecturas**: xvector, ecapa_tdnn, feedforward
5. **Audio**: Usa el directorio de audio de `vggish-backbone/` para evitar duplicar archivos
