#!/usr/bin/env python3
"""
Script de predicción usando Soft Voting.

Carga los K modelos entrenados con K-Fold y combina sus predicciones
promediando logits antes de aplicar argmax (soft voting).

Los audios se segmentan ON-THE-FLY según la duración especificada
(--duration) - NO hay archivos segmentados en disco.

Utiliza YAMNet como backbone para extraer embeddings de audio.

Uso:
    python inferir.py --duration 5                    # Predicciones aleatorias (05seg, overlap 0.5)
    python inferir.py --duration 10 --overlap 0.0     # Sin solapamiento
    python inferir.py --duration 5 --k-folds 10       # Usa modelos de 10-fold
    python inferir.py --duration 5 --evaluar           # Evalúa en conjunto blind
    python inferir.py --duration 5 --audio ruta.wav    # Predice un archivo específico
    python inferir.py --duration 30 --train-duration 5 # Cross-duration: modelo 05seg, test 30seg
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

# Añadir carpeta raíz al path para importar modelo.py
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from models.modelo_ecapa import ECAPAMultiTask
from models.modelo_feedforward import FeedForwardMultiTask
from models.modelo_xvector import SMAWXVectorModel
from utils.audio_utils import PROJECT_ROOT, load_audio_segment
from utils.logging_utils import setup_log_file
from utils.timing import timer

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


# =============================================================================
# Parseo de argumentos
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predicción de soldadura SMAW usando Ensemble con Soft Voting"
    )
    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        choices=[1, 2, 5, 10, 20, 30, 50],
        help="Duración de segmento en segundos (1, 2, 5, 10, 20, 30, 50)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap entre segmentos como ratio (0.0 a 0.75, default: 0.5)",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Número de folds del ensemble a usar (default: 5)",
    )
    parser.add_argument(
        "--audio", type=str, help="Ruta a un archivo de audio para predecir"
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="Evaluar ensemble en conjunto blind (vida real)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Número de muestras aleatorias a mostrar (default: 10)",
    )
    parser.add_argument(
        "--train-duration",
        type=int,
        default=None,
        help="Duración (seg) usada para entrenar el modelo a cargar. Default: mismo que --duration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xvector",
        choices=["xvector", "ecapa", "feedforward"],
        help="Arquitectura del modelo a cargar (default: xvector)",
    )
    return parser.parse_args()


# =============================================================================
# Extracción de embeddings YAMNet
# =============================================================================


def extract_yamnet_embeddings_from_segment(
    yamnet_model,
    audio_path: str,
    segment_idx: int,
    segment_duration: float,
    overlap_seconds: float,
) -> np.ndarray:
    """Extrae embeddings YAMNet de un segmento específico de audio."""
    full_path = PROJECT_ROOT / audio_path

    segment = load_audio_segment(
        full_path,
        segment_duration=segment_duration,
        segment_index=segment_idx,
        sr=16000,
        overlap_seconds=overlap_seconds,
    )

    if segment is None:
        raise ValueError(f"No se pudo cargar segmento {segment_idx} de {audio_path}")

    # YAMNet espera ventanas de 0.96 segundos con hop de 0.48 segundos
    window_size = int(0.96 * 16000)
    hop_size = int(0.48 * 16000)
    embeddings_list = []

    for start in range(0, len(segment), hop_size):
        end = start + window_size
        if end > len(segment):
            window = np.zeros(window_size, dtype=np.float32)
            window[: len(segment) - start] = segment[start:]
        else:
            window = segment[start:end]

        result = yamnet_model(window)
        # YAMNet devuelve una lista: [embeddings_class (521), embeddings_deep (1024), spectrogram]
        # Usar result[1] que tiene los embeddings profundos de 1024 dimensiones
        if isinstance(result, tuple):
            embedding = result[1].numpy()
        elif isinstance(result, list):
            embedding = np.array(result[1])
        else:
            embedding = result.numpy()

        embeddings_list.append(embedding[0])

        if end >= len(segment):
            break

    return np.stack(embeddings_list, axis=0)


def extract_yamnet_embeddings(yamnet_model, audio_path):
    """Extrae embeddings YAMNet de un archivo de audio completo."""
    import librosa

    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    window_size = int(0.96 * 16000)
    hop_size = int(0.48 * 16000)
    embeddings_list = []

    for start in range(0, len(y), hop_size):
        end = start + window_size
        if end > len(y):
            segment = np.zeros(window_size, dtype=np.float32)
            segment[: len(y) - start] = y[start:]
        else:
            segment = y[start:end]

        result = yamnet_model(segment)
        # YAMNet devuelve una lista: [embeddings_class (521), embeddings_deep (1024), spectrogram]
        # Usar result[1] que tiene los embeddings profundos de 1024 dimensiones
        if isinstance(result, tuple):
            embedding = result[1].numpy()
        elif isinstance(result, list):
            embedding = np.array(result[1])
        else:
            embedding = result.numpy()

        embeddings_list.append(embedding[0])

        if end >= len(y):
            break

    return np.stack(embeddings_list, axis=0)


# =============================================================================
# Modelo y ensemble
# =============================================================================


def create_model(
    plate_encoder, electrode_encoder, current_type_encoder, device, model_type="xvector"
):
    """Crea una instancia del modelo según el tipo especificado."""
    if model_type == "xvector":
        return SMAWXVectorModel(
            feat_dim=1024,
            xvector_dim=512,
            emb_dim=256,
            num_classes_espesor=len(plate_encoder.classes_),
            num_classes_electrodo=len(electrode_encoder.classes_),
            num_classes_corriente=len(current_type_encoder.classes_),
        ).to(device)
    elif model_type == "ecapa":
        return ECAPAMultiTask(
            feat_dim=1024,
            ecapa_channels=1024,
            emb_dim=256,
            num_classes_espesor=len(plate_encoder.classes_),
            num_classes_electrodo=len(electrode_encoder.classes_),
            num_classes_corriente=len(current_type_encoder.classes_),
        ).to(device)
    elif model_type == "feedforward":
        return FeedForwardMultiTask(
            num_classes_espesor=len(plate_encoder.classes_),
            num_classes_electrodo=len(electrode_encoder.classes_),
            num_classes_corriente=len(current_type_encoder.classes_),
        ).to(device)
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")


def load_ensemble_models(
    models_dir,
    n_models,
    plate_encoder,
    electrode_encoder,
    current_type_encoder,
    device,
    model_type="xvector",
):
    """Carga los K modelos del ensemble."""
    if not models_dir.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta {models_dir}. "
            f"Ejecuta primero: python entrenar.py --duration X --k-folds {n_models}"
        )

    models = []
    for fold in range(n_models):
        model_path = models_dir / f"model_fold_{fold}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo {model_path}. "
                f"Ejecuta primero: python entrenar.py --duration X --k-folds {n_models}"
            )

        model = create_model(
            plate_encoder, electrode_encoder, current_type_encoder, device, model_type
        )
        state_dict = torch.load(model_path, map_location=device)
        # Map legacy Spanish key names to English if needed
        mapped_keys = {}
        legacy_map = {
            "classifier_espesor": "classifier_plate",
            "classifier_electrodo": "classifier_electrode",
            "classifier_corriente": "classifier_current",
            "fc_espesor": "fc_plate",
            "fc_electrodo": "fc_electrode",
            "fc_corriente": "fc_current",
        }
        for k, v in state_dict.items():
            new_key = k
            for old_name, new_name in legacy_map.items():
                if old_name in k:
                    new_key = k.replace(old_name, new_name)
            mapped_keys[new_key] = v
        model.load_state_dict(mapped_keys)
        model.eval()
        models.append(model)

    print(f"Cargados {len(models)} modelos ({n_models}-fold) desde {models_dir}")
    return models


def predict_ensemble(
    ensemble_models,
    embeddings_tensor,
    plate_encoder,
    electrode_encoder,
    current_type_encoder,
):
    """
    Realiza predicción con ensemble usando soft voting.

    1. Obtiene logits de cada modelo
    2. Promedia los logits
    3. Aplica argmax para obtener la clase predicha
    """
    logits_plate_list = []
    logits_electrode_list = []
    logits_current_list = []

    with torch.no_grad():
        for model in ensemble_models:
            outputs = model(embeddings_tensor)
            logits_plate_list.append(outputs["plate"])
            logits_electrode_list.append(outputs["electrode"])
            logits_current_list.append(outputs["current"])

    # Promediar logits (soft voting)
    avg_logits_espesor = torch.stack(logits_plate_list).mean(dim=0)
    avg_logits_electrodo = torch.stack(logits_electrode_list).mean(dim=0)
    avg_logits_corriente = torch.stack(logits_current_list).mean(dim=0)

    # Obtener clases predichas
    pred_plate_idx = avg_logits_espesor.argmax(dim=1).item()
    pred_electrode_idx = avg_logits_electrodo.argmax(dim=1).item()
    pred_current_idx = avg_logits_corriente.argmax(dim=1).item()

    # Decodificar etiquetas
    pred_plate = plate_encoder.classes_[pred_plate_idx]
    pred_electrode = electrode_encoder.classes_[pred_electrode_idx]
    pred_current = current_type_encoder.classes_[pred_current_idx]

    # Probabilidades (softmax sobre logits promediados)
    probs_plate = torch.softmax(avg_logits_espesor, dim=1)[0].cpu().numpy()
    probs_electrode = torch.softmax(avg_logits_electrodo, dim=1)[0].cpu().numpy()
    probs_current = torch.softmax(avg_logits_corriente, dim=1)[0].cpu().numpy()

    return {
        "plate": pred_plate,
        "electrode": pred_electrode,
        "current": pred_current,
        "probs_plate": probs_plate,
        "probs_electrode": probs_electrode,
        "probs_current": probs_current,
        "plate_idx": pred_plate_idx,
        "electrode_idx": pred_electrode_idx,
        "current_idx": pred_current_idx,
    }


# =============================================================================
# Guardar resultados de inferencia
# =============================================================================


def save_inference_result(result_data, infer_json_path, config, elapsed_time=None):
    """
    Guarda los resultados de inferencia en inferencia.json.
    Conserva todas las corridas anteriores.
    """
    if infer_json_path.exists():
        with open(infer_json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    result_data["timestamp"] = datetime.now().isoformat()
    result_data["config"] = config

    if elapsed_time is not None:
        result_data["execution_time"] = {
            "seconds": round(elapsed_time, 2),
            "minutes": round(elapsed_time / 60, 2),
            "hours": round(elapsed_time / 3600, 4),
        }

    all_results.append(result_data)

    with open(infer_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados guardados en {infer_json_path}")


def format_confusion_matrix_markdown(cm, classes):
    """Formatea una matriz de confusión como tabla Markdown."""
    header = "| Pred \\ Real | " + " | ".join(classes) + " |"
    separator = "|" + "|".join(["---"] * (len(classes) + 1)) + "|"

    rows = [header, separator]
    for i, cls in enumerate(classes):
        row = (
            f"| **{cls}** | "
            + " | ".join(str(cm[i][j]) for j in range(len(classes)))
            + " |"
        )
        rows.append(row)

    return "\n".join(rows)


def generate_metrics_document(
    results,
    output_dir,
    segment_duration,
    plate_encoder,
    electrode_encoder,
    current_type_encoder,
):
    """
    Genera un documento Markdown con todas las métricas y matrices de confusión.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = f"""# Métricas de Clasificación SMAW - {int(segment_duration)}seg

**Fecha de evaluación:** {timestamp}

**Configuración:**
- Duración de segmento: {segment_duration}s
- Número de muestras (blind): {results["n_samples"]}
- Número de modelos (ensemble): {results["n_models"]}
- Método de votación: {results["voting_method"]}

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | {results["accuracy"]["plate_thickness"]:.4f} | {results["macro_f1"]["plate_thickness"]:.4f} |
| Electrode Type | {results["accuracy"]["electrode"]:.4f} | {results["macro_f1"]["electrode"]:.4f} |
| Current Type | {results["accuracy"]["current_type"]:.4f} | {results["macro_f1"]["current_type"]:.4f} |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** {results["accuracy"]["plate_thickness"]:.4f}
- **Macro F1-Score:** {results["macro_f1"]["plate_thickness"]:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results["confusion_matrices"]["plate_thickness"], results["classes"]["plate_thickness"])}

### Classification Report
"""
    cr_plate = results["classification_reports"]["plate_thickness"]
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results["classes"]["plate_thickness"]:
        metrics = cr_plate.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += f"""
---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** {results["accuracy"]["electrode"]:.4f}
- **Macro F1-Score:** {results["macro_f1"]["electrode"]:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results["confusion_matrices"]["electrode"], results["classes"]["electrode"])}

### Classification Report
"""
    cr_electrode = results["classification_reports"]["electrode"]
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results["classes"]["electrode"]:
        metrics = cr_electrode.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += f"""
---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** {results["accuracy"]["current_type"]:.4f}
- **Macro F1-Score:** {results["macro_f1"]["current_type"]:.4f}

### Confusion Matrix

{format_confusion_matrix_markdown(results["confusion_matrices"]["current_type"], results["classes"]["current_type"])}

### Classification Report
"""
    cr_current = results["classification_reports"]["current_type"]
    doc += "\n| Clase | Precision | Recall | F1-Score | Support |\n"
    doc += "|-------|-----------|--------|----------|--------|\n"
    for cls in results["classes"]["current_type"]:
        metrics = cr_current.get(cls, {})
        doc += f"| {cls} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('f1-score', 0):.4f} | {int(metrics.get('support', 0))} |\n"

    doc += """
---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
"""

    metrics_dir = output_dir / "metricas"
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / "METRICAS.md"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"Documento de métricas guardado en {metrics_file}")


# =============================================================================
# Funciones de Evaluación
# =============================================================================


def evaluate_blind_set(
    ctx,
):
    """Evalúa el ensemble en el conjunto blind (validación vida real)."""
    start_time = time.time()

    # Intentar blind.csv con nombre específico de overlap, fallback a genérico
    overlap_ratio = ctx["overlap_ratio"]
    blind_csv = ctx["test_dir"] / f"blind_overlap_{overlap_ratio}.csv"
    if not blind_csv.exists():
        blind_csv = ctx["test_dir"] / "blind.csv"
    if not blind_csv.exists():
        print(
            f"No se encontró {blind_csv}. Ejecuta generar_splits.py --duration {ctx['test_seconds']} --overlap {overlap_ratio} primero."
        )
        return None

    with timer("Cargar blind.csv"):
        blind_df = pd.read_csv(blind_csv)
    print(f"\nEvaluando ensemble en {len(blind_df)} segmentos de BLIND (vida real)...")
    print(f"Duración de segmento (test): {ctx['segment_duration']}s")

    y_true_plate, y_pred_plate = [], []
    y_true_electrode, y_pred_electrode = [], []
    y_true_current, y_pred_current = [], []

    with timer("Inferencia BLIND (segmentos)"):
        for idx, row in blind_df.iterrows():
            if idx % 100 == 0:
                print(f"  Procesando {idx}/{len(blind_df)}...")

            audio_path = row["Audio Path"]
            segment_idx = int(row["Segment Index"])

            y_true_plate.append(row["Plate Thickness"])
            y_true_electrode.append(row["Electrode"])
            y_true_current.append(row["Type of Current"])

            embeddings = extract_yamnet_embeddings_from_segment(
                ctx["yamnet_model"],
                audio_path,
                segment_idx,
                ctx["segment_duration"],
                ctx["overlap_seconds"],
            )
            embeddings_tensor = (
                torch.tensor(embeddings, dtype=torch.float32)
                .unsqueeze(0)
                .to(ctx["device"])
            )
            result = predict_ensemble(
                ctx["ensemble_models"],
                embeddings_tensor,
                ctx["plate_encoder"],
                ctx["electrode_encoder"],
                ctx["current_type_encoder"],
            )

            y_pred_plate.append(result["plate"])
            y_pred_electrode.append(result["electrode"])
            y_pred_current.append(result["current"])

    # Calcular métricas
    print("\n" + "=" * 70)
    print("RESULTADOS DEL ENSEMBLE EN CONJUNTO BLIND (VIDA REAL)")
    print("=" * 70)

    acc_plate = accuracy_score(y_true_plate, y_pred_plate)
    acc_electrode = accuracy_score(y_true_electrode, y_pred_electrode)
    acc_current = accuracy_score(y_true_current, y_pred_current)

    n_samples = len(y_true_plate)
    exact_matches = sum(
        1
        for i in range(n_samples)
        if (
            y_pred_plate[i] == y_true_plate[i]
            and y_pred_electrode[i] == y_true_electrode[i]
            and y_pred_current[i] == y_true_current[i]
        )
    )
    exact_match_accuracy = exact_matches / n_samples
    hamming_accuracy = (acc_plate + acc_electrode + acc_current) / 3

    print(f"\nMétricas Globales (Multi-tarea):")
    print(f"  Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"  Hamming Accuracy:     {hamming_accuracy:.4f}")

    print(f"\nAccuracy por Tarea:")
    print(f"  Plate Thickness:  {acc_plate:.4f}")
    print(f"  Electrode:        {acc_electrode:.4f}")
    print(f"  Type of Current:  {acc_current:.4f}")

    f1_plate = f1_score(y_true_plate, y_pred_plate, average="macro")
    f1_electrode = f1_score(y_true_electrode, y_pred_electrode, average="macro")
    f1_current = f1_score(y_true_current, y_pred_current, average="macro")

    print(f"\nMacro F1-Score:")
    print(f"  Plate Thickness:  {f1_plate:.4f}")
    print(f"  Electrode:        {f1_electrode:.4f}")
    print(f"  Type of Current:  {f1_current:.4f}")

    print("\n--- Plate Thickness ---")
    print(classification_report(y_true_plate, y_pred_plate))
    print("\n--- Electrode Type ---")
    print(classification_report(y_true_electrode, y_pred_electrode))
    print("\n--- Type of Current ---")
    print(classification_report(y_true_current, y_pred_current))

    cm_plate = confusion_matrix(
        y_true_plate, y_pred_plate, labels=ctx["plate_encoder"].classes_
    )
    cm_electrode = confusion_matrix(
        y_true_electrode, y_pred_electrode, labels=ctx["electrode_encoder"].classes_
    )
    cm_current = confusion_matrix(
        y_true_current, y_pred_current, labels=ctx["current_type_encoder"].classes_
    )

    print("\nMatrices de Confusión:")
    print("\nPlate Thickness:")
    print(cm_plate)
    print(f"Clases: {ctx['plate_encoder'].classes_}")
    print("\nElectrode Type:")
    print(cm_electrode)
    print(f"Clases: {ctx['electrode_encoder'].classes_}")
    print("\nType of Current:")
    print(cm_current)
    print(f"Clases: {ctx['current_type_encoder'].classes_}")

    results = {
        "model_type": ctx["config_dict"]["model_type"],
        "metrics": {
            "plate": {
                "accuracy": float(acc_plate),
                "f1_macro": float(f1_plate),
            },
            "electrode": {
                "accuracy": float(acc_electrode),
                "f1_macro": float(f1_electrode),
            },
            "current": {
                "accuracy": float(acc_current),
                "f1_macro": float(f1_current),
            },
            "global": {
                "exact_match": float(exact_match_accuracy),
                "hamming_accuracy": float(hamming_accuracy),
            },
        },
        "confusion_matrices": {
            "plate": cm_plate.tolist(),
            "electrode": cm_electrode.tolist(),
            "current": cm_current.tolist(),
        },
        "label_classes": {
            "plate": ctx["plate_encoder"].classes_.tolist(),
            "electrode": ctx["electrode_encoder"].classes_.tolist(),
            "current": ctx["current_type_encoder"].classes_.tolist(),
        },
    }

    elapsed_time = time.time() - start_time
    print(f"\nTiempo de ejecución: {elapsed_time:.2f}s ({elapsed_time / 60:.2f}min)")

    save_inference_result(
        results, ctx["infer_json"], ctx["config_dict"], elapsed_time=elapsed_time
    )

    return results


def show_random_predictions(ctx, n_samples=10):
    """Muestra predicciones aleatorias del conjunto blind."""
    start_time = time.time()

    # Intentar blind.csv con nombre específico de overlap, fallback a genérico
    overlap_ratio = ctx["overlap_ratio"]
    blind_csv = ctx["test_dir"] / f"blind_overlap_{overlap_ratio}.csv"
    if not blind_csv.exists():
        blind_csv = ctx["test_dir"] / "blind.csv"
    if not blind_csv.exists():
        print(
            f"No se encontró {blind_csv}. Ejecuta generar_splits.py --duration {ctx['test_seconds']} --overlap {overlap_ratio} primero."
        )
        return

    with timer("Cargar blind.csv"):
        blind_df = pd.read_csv(blind_csv)
    num_samples = min(n_samples, len(blind_df))
    samples = blind_df.sample(n=num_samples, random_state=None)

    print(f"\n{'=' * 80}")
    print(f"  PREDICCIONES ALEATORIAS ({num_samples} muestras)")
    print(f"{'=' * 80}")
    print(f"\n  {'Archivo':<25} {'Plate':<12} {'Electrode':<10} {'Current':<8}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 10} {'-' * 8}")

    correctas_plate = 0
    correctas_electrode = 0
    correctas_current = 0
    correctas_todas = 0

    with timer("Inferencia muestras aleatorias"):
        for _, row in samples.iterrows():
            audio_path = row["Audio Path"]
            segment_idx = int(row["Segment Index"])
            real_plate = row["Plate Thickness"]
            real_electrode = row["Electrode"]
            real_current = row["Type of Current"]

            embeddings = extract_yamnet_embeddings_from_segment(
                ctx["yamnet_model"],
                audio_path,
                segment_idx,
                ctx["segment_duration"],
                ctx["overlap_seconds"],
            )
            embeddings_tensor = (
                torch.tensor(embeddings, dtype=torch.float32)
                .unsqueeze(0)
                .to(ctx["device"])
            )
            result = predict_ensemble(
                ctx["ensemble_models"],
                embeddings_tensor,
                ctx["plate_encoder"],
                ctx["electrode_encoder"],
                ctx["current_type_encoder"],
            )

            # FIX: Este bloque estaba fuera del for-loop en la versión anterior
            correct_plate = result["plate"] == real_plate
            correct_electrode = result["electrode"] == real_electrode
            correct_current = result["current"] == real_current
            correct_all = correct_plate and correct_electrode and correct_current

            if correct_plate:
                correctas_plate += 1
            if correct_electrode:
                correctas_electrode += 1
            if correct_current:
                correctas_current += 1
            if correct_all:
                correctas_todas += 1

            sym_plate = "✓" if correct_plate else "✗"
            sym_electrode = "✓" if correct_electrode else "✗"
            sym_current = "✓" if correct_current else "✗"

            audio_name = Path(audio_path).name
            print(f"\n  {audio_name:<25}")
            print(
                f"    Real:      {real_plate:<12} {real_electrode:<10} {real_current:<8}"
            )
            print(
                f"    Pred:      {result['plate']:<12} {result['electrode']:<10} {result['current']:<8}"
            )
            print(
                f"    Status:    {sym_plate:<12} {sym_electrode:<10} {sym_current:<8}"
            )
            print(
                f"    Conf:      {result['probs_plate'].max():.2f}         {result['probs_electrode'].max():.2f}       {result['probs_current'].max():.2f}"
            )

    print(f"\n{'=' * 80}")
    print(f"  RESUMEN ({ctx['n_models']} modelos con Soft Voting)")
    print(f"{'=' * 80}")
    print(
        f"\n  Plate Thickness:  {correctas_plate:>2}/{num_samples} = {correctas_plate / num_samples:.4f}"
    )
    print(
        f"  Electrode:        {correctas_electrode:>2}/{num_samples} = {correctas_electrode / num_samples:.4f}"
    )
    print(
        f"  Type of Current:  {correctas_current:>2}/{num_samples} = {correctas_current / num_samples:.4f}"
    )
    print(
        f"  Todas correctas:  {correctas_todas:>2}/{num_samples} = {correctas_todas / num_samples:.4f}"
    )
    print()

    elapsed_time = time.time() - start_time
    print(f"Tiempo de ejecución: {elapsed_time:.2f}s ({elapsed_time / 60:.2f}min)")

    inference_result = {
        "model_type": ctx["config_dict"]["model_type"],
        "metrics": {
            "plate": {
                "accuracy": correctas_plate / num_samples,
                "f1_macro": 0.0,
            },
            "electrode": {
                "accuracy": correctas_electrode / num_samples,
                "f1_macro": 0.0,
            },
            "current": {
                "accuracy": correctas_current / num_samples,
                "f1_macro": 0.0,
            },
            "global": {
                "exact_match": correctas_todas / num_samples,
                "hamming_accuracy": (
                    correctas_plate + correctas_electrode + correctas_current
                )
                / (num_samples * 3),
            },
        },
    }
    save_inference_result(
        inference_result,
        ctx["infer_json"],
        ctx["config_dict"],
        elapsed_time=elapsed_time,
    )


def predict_single_audio(ctx, audio_path):
    """Predice un archivo de audio específico."""
    start_time = time.time()

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: No se encontró el archivo {audio_path}")
        return

    print(f"\nPrediciendo: {audio_path}")
    print("=" * 70)

    with timer("Inferencia audio"):
        embeddings = extract_yamnet_embeddings(ctx["yamnet_model"], str(audio_path))
        embeddings_tensor = (
            torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(ctx["device"])
        )
        result = predict_ensemble(
            ctx["ensemble_models"],
            embeddings_tensor,
            ctx["plate_encoder"],
            ctx["electrode_encoder"],
            ctx["current_type_encoder"],
        )

    print(f"\nResultados (Ensemble de {ctx['n_models']} modelos con Soft Voting):")
    print(f"\n  Plate Thickness: {result['plate']}")
    print(
        f"    Probabilidades: {dict(zip(ctx['plate_encoder'].classes_, result['probs_plate'].round(3)))}"
    )
    print(f"\n  Electrode: {result['electrode']}")
    print(
        f"    Probabilidades: {dict(zip(ctx['electrode_encoder'].classes_, result['probs_electrode'].round(3)))}"
    )
    print(f"\n  Type of Current: {result['current']}")
    print(
        f"    Probabilidades: {dict(zip(ctx['current_type_encoder'].classes_, result['probs_current'].round(3)))}"
    )

    inference_result = {
        "model_type": ctx["config_dict"]["model_type"],
        "mode": "single_audio",
        "audio_path": str(audio_path),
        "predictions": {
            "plate": result["plate"],
            "electrode": result["electrode"],
            "current": result["current"],
        },
        "probabilities": {
            "plate": dict(
                zip(
                    ctx["plate_encoder"].classes_.tolist(),
                    result["probs_plate"].round(4).tolist(),
                )
            ),
            "electrode": dict(
                zip(
                    ctx["electrode_encoder"].classes_.tolist(),
                    result["probs_electrode"].round(4).tolist(),
                )
            ),
            "current": dict(
                zip(
                    ctx["current_type_encoder"].classes_.tolist(),
                    result["probs_current"].round(4).tolist(),
                )
            ),
        },
    }

    elapsed_time = time.time() - start_time
    print(f"\nTiempo de ejecución: {elapsed_time:.2f}s")
    save_inference_result(
        inference_result,
        ctx["infer_json"],
        ctx["config_dict"],
        elapsed_time=elapsed_time,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # Set up logging
    log_file, log_path = setup_log_file(
        ROOT_DIR / "logs",
        "inferir",
        suffix=f"_{int(args.duration):02d}seg_{args.model}",
    )
    sys.stdout = log_file

    # Configuración derivada de argumentos
    SEGMENT_DURATION = float(args.duration)
    OVERLAP_RATIO = args.overlap
    OVERLAP_SECONDS = SEGMENT_DURATION * OVERLAP_RATIO
    N_MODELS = args.k_folds

    TEST_SECONDS = args.duration
    TRAIN_SECONDS = (
        args.train_duration if args.train_duration is not None else args.duration
    )

    TRAIN_DIR = ROOT_DIR / f"{TRAIN_SECONDS:02d}seg"
    TEST_DIR = ROOT_DIR / f"{TEST_SECONDS:02d}seg"
    DURATION_DIR = ROOT_DIR / f"{TEST_SECONDS:02d}seg"
    INFER_JSON = DURATION_DIR / "inferencia.json"

    # Directorio de modelos con overlap y arquitectura específica
    MODELS_DIR = (
        TRAIN_DIR / "modelos" / args.model / f"k{N_MODELS:02d}_overlap_{OVERLAP_RATIO}"
    )

    if not MODELS_DIR.exists():
        print(f"[ERROR] No se encontró el directorio de modelos: {MODELS_DIR}")
        sys.exit(1)

    print(f"[INFO] Modelos:              {MODELS_DIR}")
    print(f"[INFO] Arquitectura:         {args.model}")
    print(f"[INFO] Duración (train):     {TRAIN_SECONDS}s")
    print(f"[INFO] Duración (test):      {SEGMENT_DURATION}s")
    print(f"[INFO] Overlap ratio:        {OVERLAP_RATIO}")
    print(f"[INFO] Overlap seconds:      {OVERLAP_SECONDS}s")

    # Cargar YAMNet
    print(f"Cargando modelo YAMNet desde TensorFlow Hub...")
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    print("Modelo YAMNet cargado correctamente.")

    # Cargar encoders desde train.csv (preferir overlap-específico)
    train_csv = TRAIN_DIR / f"train_overlap_{OVERLAP_RATIO}.csv"
    if not train_csv.exists():
        train_csv = TRAIN_DIR / "train.csv"
    train_data = pd.read_csv(train_csv)
    plate_encoder = LabelEncoder()
    electrode_encoder = LabelEncoder()
    current_type_encoder = LabelEncoder()
    plate_encoder.fit(train_data["Plate Thickness"])
    electrode_encoder.fit(train_data["Electrode"])
    current_type_encoder.fit(train_data["Type of Current"])

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar ensemble
    ensemble_models = load_ensemble_models(
        MODELS_DIR,
        N_MODELS,
        plate_encoder,
        electrode_encoder,
        current_type_encoder,
        device,
        args.model,
    )

    # Diccionario de configuración para guardar en JSON
    config_dict = {
        "duration": TEST_SECONDS,
        "overlap": OVERLAP_RATIO,
        "k_folds": N_MODELS,
        "n_models": N_MODELS,
        "model_type": args.model,
    }

    # Contexto compartido entre funciones
    ctx = {
        "yamnet_model": yamnet_model,
        "ensemble_models": ensemble_models,
        "plate_encoder": plate_encoder,
        "electrode_encoder": electrode_encoder,
        "current_type_encoder": current_type_encoder,
        "device": device,
        "segment_duration": SEGMENT_DURATION,
        "overlap_ratio": OVERLAP_RATIO,
        "overlap_seconds": OVERLAP_SECONDS,
        "n_models": N_MODELS,
        "train_seconds": TRAIN_SECONDS,
        "test_seconds": TEST_SECONDS,
        "train_dir": TRAIN_DIR,
        "test_dir": TEST_DIR,
        "duration_dir": DURATION_DIR,
        "infer_json": INFER_JSON,
        "config_dict": config_dict,
    }

    # Despachar acción
    try:
        if args.audio:
            predict_single_audio(ctx, args.audio)
        elif args.evaluar:
            evaluate_blind_set(ctx)
        else:
            show_random_predictions(ctx, n_samples=args.n)
    finally:
        print(f"\nResultados guardados en JSON")
        print(f"Logs guardados en: {log_path}")
        # Close log file
        log_file.close()


if __name__ == "__main__":
    main()
