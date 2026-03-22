# YAMNet Approach - Clasificación de Audio SMAW

Este proyecto utiliza **YAMNet** como approach para extraer embeddings de audio de soldadura SMAW.

## Diferencias con VGGish Approach

| Característica       | VGGish                    | YAMNet                    |
| -------------------- | ------------------------- | ------------------------- |
| Dimensión embeddings | 128                       | 1024                      |
| URL modelo           | tfhub.dev/google/vggish/1 | tfhub.dev/google/yamnet/1 |

## Estructura del Proyecto

```
yamnet-backbone/
├── modelo_xvector.py       # Modelo X-Vector (feat_dim=1024)
├── modelo_ecapa.py         # Modelo ECAPA-TDNN (feat_dim=1024)
├── modelo_feedforward.py   # Modelo FeedForward (input=2048)
├── entrenar_xvector.py     # Script entrenamiento X-Vector
├── entrenar_ecapa.py       # Script entrenamiento ECAPA
├── entrenar_feedforward.py # Script entrenamiento FeedForward
├── inferir.py              # Script inferencia y evaluación
├── entrenar_todos.sh       # Script batch para entrenar todos
├── utils/                  # Utilidades (apunta a audio de vggish-backbone)
├── 01seg/...50seg/         # Datos por duración
```

## Uso

```bash
# Entrenar X-Vector
python entrenar_xvector.py --duration 5 --overlap 0.5 --k-folds 10

# Entrenar todos los modelos
./entrenar_todos.sh

# Evaluar
python inferir.py --duration 5 --k-folds 10 --model xvector --evaluar
```

## Audio

El proyecto utiliza el directorio de audio de `vggish-backbone/audio` para evitar duplicar archivos.
