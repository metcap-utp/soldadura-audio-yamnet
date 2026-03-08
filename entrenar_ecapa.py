#!/usr/bin/env python3
"""
Entrenamiento ECAPA-TDNN para clasificación SMAW con YAMNet embeddings.

Uso:
    python entrenar_ecapa.py --duration 10 --overlap 0.5 --k-folds 10
"""

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from models.modelo_ecapa import ECAPAMultiTask
from utils.audio_utils import PROJECT_ROOT, load_audio_segment
from utils.timing import timer
from utils.logging_utils import setup_log_file
from utils.checkpoint import TrainingCheckpoint, setup_pause_handler, pause_requested

warnings.filterwarnings("ignore")

# Hiperparámetros
BATCH_SIZE = 32
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SWA_START = 5
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# Cargar modelo YAMNet
print(f"Cargando modelo YAMNet desde TensorFlow Hub...")
yamnet_model = hub.load(YAMNET_MODEL_URL)
print("Modelo YAMNet cargado correctamente.")


# ============= Parseo de argumentos =============
def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento ECAPA-TDNN SMAW con K-Fold CV"
    )
    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        choices=[1, 2, 5, 10, 20, 30, 50],
        help="Duración de segmento en segundos",
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
        default=10,
        choices=[1, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        help="Número de folds para cross-validation (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="No usar cache de embeddings YAMNet (fuerza recalculo)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cuda/cpu, auto-detectado si no se especifica)",
    )
    return parser.parse_args()


def extract_session_from_path(audio_path: str) -> str:
    """Extrae el identificador de sesión del path del audio."""
    parts = Path(audio_path).parts
    for part in parts:
        if part.endswith("_Audio"):
            return part
    return Path(audio_path).parent.name


def extract_yamnet_embeddings_from_segment(
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

    window_size = 16000
    hop_size = 8000
    embeddings_list = []

    for start in range(0, len(segment), hop_size):
        end = start + window_size
        if end > len(segment):
            window = np.zeros(window_size, dtype=np.float32)
            window[: len(segment) - start] = segment[start:]
        else:
            window = segment[start:end]

        # YAMNet devuelve una lista: [embeddings_class (521), embeddings_deep (1024), spectrogram]
        # Usar result[1] que tiene los embeddings profundos de 1024 dimensiones
        result = yamnet_model(window)
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


class AudioDataset(Dataset):
    """Dataset para embeddings YAMNet."""

    def __init__(self, embeddings_list, labels_plate, labels_electrode, labels_current):
        self.embeddings_list = embeddings_list
        self.labels_plate = labels_plate
        self.labels_electrode = labels_electrode
        self.labels_current = labels_current

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings_list[idx], dtype=torch.float32),
            torch.tensor(self.labels_plate[idx], dtype=torch.long),
            torch.tensor(self.labels_electrode[idx], dtype=torch.long),
            torch.tensor(self.labels_current[idx], dtype=torch.long),
        )


def collate_fn_pad(batch):
    """Padding de secuencias a longitud máxima del batch."""
    embeddings, labels_plate, labels_electrode, labels_current = zip(*batch)
    max_len = max(emb.shape[0] for emb in embeddings)

    padded_embeddings = []
    for emb in embeddings:
        if emb.shape[0] < max_len:
            pad = torch.zeros(max_len - emb.shape[0], emb.shape[1])
            emb = torch.cat([emb, pad], dim=0)
        padded_embeddings.append(emb)

    return (
        torch.stack(padded_embeddings),
        torch.stack(list(labels_plate)),
        torch.stack(list(labels_electrode)),
        torch.stack(list(labels_current)),
    )


def train_one_fold(
    fold_idx,
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    class_weights,
    encoders,
    device,
    models_dir,
):
    """Entrena un fold y guarda el mejor modelo."""

    plate_encoder, electrode_encoder, current_type_encoder = encoders

    # Crear datasets
    train_dataset = AudioDataset(
        train_embeddings,
        train_labels["plate"],
        train_labels["electrode"],
        train_labels["current"],
    )
    val_dataset = AudioDataset(
        val_embeddings,
        val_labels["plate"],
        val_labels["electrode"],
        val_labels["current"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_pad,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad
    )

    # Crear modelo
    model = ECAPAMultiTask(
        feat_dim=1024,
        ecapa_channels=1024,
        emb_dim=256,
        num_classes_espesor=len(plate_encoder.classes_),
        num_classes_electrodo=len(electrode_encoder.classes_),
        num_classes_corriente=len(current_type_encoder.classes_),
    ).to(device)

    # Criterios con class weights
    criterion_plate = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights["plate"]).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )
    criterion_electrode = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights["electrode"]).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )
    criterion_current = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights["current"]).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_metrics = {
        "accuracy_plate": 0.0,
        "accuracy_electrode": 0.0,
        "accuracy_current": 0.0,
        "f1_plate": 0.0,
        "f1_electrode": 0.0,
        "f1_current": 0.0,
    }
    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    training_history = []

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0

        for embeddings, labels_p, labels_e, labels_c in train_loader:
            embeddings = embeddings.to(device)
            labels_p = labels_p.to(device)
            labels_e = labels_e.to(device)
            labels_c = labels_c.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)

            loss_p = criterion_plate(outputs["plate"], labels_p)
            loss_e = criterion_electrode(outputs["electrode"], labels_e)
            loss_c = criterion_current(outputs["current"], labels_c)
            loss = (loss_p + loss_e + loss_c) / 3

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # SWA update
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = {"plate": [], "electrode": [], "current": []}
        all_labels = {"plate": [], "electrode": [], "current": []}

        with torch.no_grad():
            for embeddings, labels_p, labels_e, labels_c in val_loader:
                embeddings = embeddings.to(device)
                labels_p = labels_p.to(device)
                labels_e = labels_e.to(device)
                labels_c = labels_c.to(device)

                outputs = model(embeddings)

                loss_p = criterion_plate(outputs["plate"], labels_p)
                loss_e = criterion_electrode(outputs["electrode"], labels_e)
                loss_c = criterion_current(outputs["current"], labels_c)
                val_loss += (loss_p + loss_e + loss_c).item()

                _, pred_p = outputs["plate"].max(1)
                _, pred_e = outputs["electrode"].max(1)
                _, pred_c = outputs["current"].max(1)

                all_preds["plate"].extend(pred_p.cpu().numpy())
                all_preds["electrode"].extend(pred_e.cpu().numpy())
                all_preds["current"].extend(pred_c.cpu().numpy())
                all_labels["plate"].extend(labels_p.cpu().numpy())
                all_labels["electrode"].extend(labels_e.cpu().numpy())
                all_labels["current"].extend(labels_c.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Calcular métricas
        acc_p = np.mean(np.array(all_preds["plate"]) == np.array(all_labels["plate"]))
        acc_e = np.mean(
            np.array(all_preds["electrode"]) == np.array(all_labels["electrode"])
        )
        acc_c = np.mean(
            np.array(all_preds["current"]) == np.array(all_labels["current"])
        )

        f1_p = f1_score(all_labels["plate"], all_preds["plate"], average="macro")
        f1_e = f1_score(
            all_labels["electrode"], all_preds["electrode"], average="macro"
        )
        f1_c = f1_score(all_labels["current"], all_preds["current"], average="macro")

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": avg_val_loss,
            "val_acc_plate": acc_p,
            "val_acc_electrode": acc_e,
            "val_acc_current": acc_c,
            "val_f1_plate": f1_p,
            "val_f1_electrode": f1_e,
            "val_f1_current": f1_c,
        })

        # Early stopping y guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_metrics = {
                "accuracy_plate": acc_p,
                "accuracy_electrode": acc_e,
                "accuracy_current": acc_c,
                "f1_plate": f1_p,
                "f1_electrode": f1_e,
                "f1_current": f1_c,
            }
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    # Guardar el mejor modelo de este fold
    model_path = models_dir / f"model_fold_{fold_idx}.pth"
    torch.save(best_state_dict, model_path)

    # Calcular matrices de confusión del mejor modelo
    model.load_state_dict(best_state_dict)
    model.eval()

    val_preds = {"plate": [], "electrode": [], "current": []}
    val_labels_all = {"plate": [], "electrode": [], "current": []}

    with torch.no_grad():
        for embeddings, labels_p, labels_e, labels_c in val_loader:
            embeddings = embeddings.to(device)
            outputs = model(embeddings)

            _, pred_p = outputs["plate"].max(1)
            _, pred_e = outputs["electrode"].max(1)
            _, pred_c = outputs["current"].max(1)

            val_preds["plate"].extend(pred_p.cpu().numpy())
            val_preds["electrode"].extend(pred_e.cpu().numpy())
            val_preds["current"].extend(pred_c.cpu().numpy())
            val_labels_all["plate"].extend(labels_p.numpy())
            val_labels_all["electrode"].extend(labels_e.numpy())
            val_labels_all["current"].extend(labels_c.numpy())

    cm_plate = confusion_matrix(val_labels_all["plate"], val_preds["plate"])
    cm_electrode = confusion_matrix(val_labels_all["electrode"], val_preds["electrode"])
    cm_current = confusion_matrix(val_labels_all["current"], val_preds["current"])

    best_metrics["confusion_matrix_plate"] = cm_plate.tolist()
    best_metrics["confusion_matrix_electrode"] = cm_electrode.tolist()
    best_metrics["confusion_matrix_current"] = cm_current.tolist()

    print(
        f"  Fold {fold_idx + 1}: Plate={best_metrics['accuracy_plate']:.4f} | "
        f"Electrode={best_metrics['accuracy_electrode']:.4f} | "
        f"Current={best_metrics['accuracy_current']:.4f} | "
        f"Best epoch={best_epoch}"
    )

    return best_metrics, best_epoch, training_history


def ensemble_predict(models, embeddings, device):
    """Realiza predicciones usando voting de múltiples modelos."""
    all_logits_plate = []
    all_logits_electrode = []
    all_logits_current = []

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(embeddings.to(device))
            all_logits_plate.append(outputs["plate"])
            all_logits_electrode.append(outputs["electrode"])
            all_logits_current.append(outputs["current"])

    avg_logits_plate = torch.stack(all_logits_plate).mean(dim=0)
    avg_logits_electrode = torch.stack(all_logits_electrode).mean(dim=0)
    avg_logits_current = torch.stack(all_logits_current).mean(dim=0)

    pred_plate = avg_logits_plate.argmax(dim=1)
    pred_electrode = avg_logits_electrode.argmax(dim=1)
    pred_current = avg_logits_current.argmax(dim=1)

    return pred_plate, pred_electrode, pred_current


def get_system_info(device):
    """Recolecta información del sistema y dependencias."""
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
    except Exception:
        pass
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    return info


def count_model_parameters(model):
    """Cuenta parámetros totales y entrenables del modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_embeddings_cache_dir(duration_dir: Path) -> Path:
    """Directorio de cache de embeddings."""
    cache_dir = duration_dir / "embeddings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_embeddings_cache_path(
    duration_dir: Path, segment_duration: float, overlap_ratio: float
) -> Path:
    """Obtiene la ruta del archivo de cache de embeddings."""
    cache_dir = get_embeddings_cache_dir(duration_dir)
    return (
        cache_dir / f"yamnet_embeddings_{segment_duration}s_overlap_{overlap_ratio}.pkl"
    )


def get_legacy_embeddings_cache_paths(
    duration_dir: Path, segment_duration: float
) -> list[Path]:
    """Rutas legacy posibles para reutilizar caches viejos sin borrarlos."""
    cache_dir = duration_dir / "embeddings_cache"
    cache_dir.mkdir(exist_ok=True)
    dur = float(segment_duration)
    return [
        cache_dir / f"yamnet_embeddings_{dur}s_overlap0.0s.pkl",
        cache_dir / f"yamnet_embeddings_{dur}s_overlap_0.0.pkl",
        cache_dir / f"yamnet_embeddings_{dur}s.pkl",
    ]


def compute_dataset_hash(
    paths: list, segment_indices: list, segment_duration: float, overlap_ratio: float
) -> str:
    """Calcula un hash del dataset para detectar cambios."""
    data_str = f"dur={segment_duration}|overlap={overlap_ratio}|" + "".join(
        [f"{p}:{s}" for p, s in zip(paths, segment_indices)]
    )
    return hashlib.md5(data_str.encode()).hexdigest()


def load_embeddings_cache(
    paths: list,
    segment_indices: list,
    duration_dir: Path,
    segment_duration: float,
    overlap_ratio: float,
    overlap_seconds: float,
) -> tuple:
    """Carga embeddings del cache si existe y es válido.

    Returns:
        tuple: (embeddings_list, success) donde success indica si se cargó del cache
    """
    new_cache_path = get_embeddings_cache_path(
        duration_dir, segment_duration, overlap_ratio
    )
    cache_paths = [
        new_cache_path,
        *get_legacy_embeddings_cache_paths(duration_dir, segment_duration),
    ]
    cache_path = next((p for p in cache_paths if p.exists()), None)

    if cache_path is None:
        return None, False

    try:
        with timer("Cargar cache embeddings YAMNet"):
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Verificar duración del segmento
            if cache_data.get("segment_duration") != segment_duration:
                print("  [CACHE] Duración no coincide, regenerando embeddings...")
                return None, False

            # Verificar hash
            current_hash = compute_dataset_hash(
                paths, segment_indices, segment_duration, overlap_ratio
            )
            if cache_data.get("hash") != current_hash:
                print("  [CACHE] Hash no coincide, regenerando embeddings...")
                return None, False

        print(
            f"  [CACHE] Cargando {len(cache_data['embeddings'])} embeddings desde cache ({cache_path})"
        )

        # Si es legacy, migrar al nuevo naming sin borrar el original
        if cache_path != new_cache_path:
            try:
                with timer("Migrar cache legacy -> nuevo naming"):
                    with open(new_cache_path, "wb") as f:
                        pickle.dump(cache_data, f)
                print(f"  [CACHE] Migrado a: {new_cache_path}")
            except Exception as e:
                print(f"  [CACHE] No se pudo migrar cache legacy: {e}")
        return cache_data["embeddings"], True

    except Exception as e:
        print(f"  [CACHE] Error leyendo cache: {e}")
        return None, False


def save_embeddings_cache(
    embeddings: list,
    paths: list,
    segment_indices: list,
    duration_dir: Path,
    segment_duration: float,
    overlap_ratio: float,
    overlap_seconds: float,
):
    """Guarda embeddings en cache."""
    cache_path = get_embeddings_cache_path(
        duration_dir, segment_duration, overlap_ratio
    )

    cache_data = {
        "hash": compute_dataset_hash(
            paths, segment_indices, segment_duration, overlap_ratio
        ),
        "embeddings": embeddings,
        "segment_duration": segment_duration,
        "overlap_ratio": overlap_ratio,
        "overlap_seconds": overlap_seconds,
        "created_at": datetime.now().isoformat(),
        "num_embeddings": len(embeddings),
    }

    with timer("Guardar cache embeddings YAMNet"):
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

    print(f"  [CACHE] Guardados {len(embeddings)} embeddings en cache ({cache_path})")


# ============= Main =============


def main():
    start_time = time.time()

    args = parse_args()
    SEGMENT_DURATION = float(args.duration)
    OVERLAP_RATIO = args.overlap
    OVERLAP_SECONDS = SEGMENT_DURATION * OVERLAP_RATIO
    N_FOLDS = args.k_folds
    RANDOM_SEED = args.seed

    # Set up logging
    ROOT_DIR = Path(__file__).parent
    log_file, log_path = setup_log_file(
        ROOT_DIR / "logs", "entrenar_ecapa", suffix=f"_{int(SEGMENT_DURATION):02d}seg"
    )
    sys.stdout = log_file

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Directorios
    ROOT_DIR = Path(__file__).parent
    DURATION_DIR = ROOT_DIR / f"{int(SEGMENT_DURATION):02d}seg"
    DURATION_DIR.mkdir(exist_ok=True)

    MODELS_BASE_DIR = DURATION_DIR / "modelos" / "ecapa"
    MODELS_BASE_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR = MODELS_BASE_DIR / f"k{N_FOLDS:02d}_overlap_{OVERLAP_RATIO}"
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"ENTRENAMIENTO ECAPA-TDNN")
    print(f"{'=' * 70}")
    print(f"Dispositivo: {device}")
    print(f"Duración de segmento: {SEGMENT_DURATION}s")
    print(f"Overlap: {OVERLAP_RATIO} ({OVERLAP_SECONDS}s)")
    print(f"K-Folds: {N_FOLDS}")
    print(f"Semilla: {RANDOM_SEED}")
    print(f"Directorio datos: {DURATION_DIR}/")
    print(f"Modelos se guardarán en: {MODELS_DIR}/")

    # Cargar CSVs
    with timer("Cargar CSVs (train/test)"):
        train_csv = DURATION_DIR / f"train_overlap_{OVERLAP_RATIO}.csv"
        test_csv = DURATION_DIR / f"test_overlap_{OVERLAP_RATIO}.csv"
        if not train_csv.exists():
            train_csv = DURATION_DIR / "train.csv"
        if not test_csv.exists():
            test_csv = DURATION_DIR / "test.csv"
        train_data = pd.read_csv(train_csv)
        test_data = pd.read_csv(test_csv)
        all_data = pd.concat([train_data, test_data], ignore_index=True)

    print(f"\nTotal de segmentos: {len(all_data)}")

    # Extraer sesión
    all_data["Session"] = all_data["Audio Path"].apply(extract_session_from_path)
    print(f"Sesiones únicas: {all_data['Session'].nunique()}")

    # Encoders
    plate_encoder = LabelEncoder()
    electrode_encoder = LabelEncoder()
    current_type_encoder = LabelEncoder()

    plate_encoder.fit(all_data["Plate Thickness"])
    electrode_encoder.fit(all_data["Electrode"])
    current_type_encoder.fit(all_data["Type of Current"])

    all_data["Plate Encoded"] = plate_encoder.transform(all_data["Plate Thickness"])
    all_data["Electrode Encoded"] = electrode_encoder.transform(all_data["Electrode"])
    all_data["Current Encoded"] = current_type_encoder.transform(
        all_data["Type of Current"]
    )

    # Extraer embeddings YAMNet
    print("\nExtrayendo embeddings YAMNet de todos los segmentos...")
    paths = all_data["Audio Path"].values
    segment_indices = all_data["Segment Index"].values

    # Intentar cargar desde cache
    all_embeddings = None
    if not args.no_cache:
        all_embeddings, cache_loaded = load_embeddings_cache(
            list(paths),
            list(segment_indices),
            DURATION_DIR,
            SEGMENT_DURATION,
            OVERLAP_RATIO,
            OVERLAP_SECONDS,
        )
    else:
        cache_loaded = False

    # Si no se cargó del cache, extraer
    if all_embeddings is None:
        with timer("Extracción YAMNet") as get_extraction_time:
            all_embeddings = []
            for i, (path, seg_idx) in enumerate(zip(paths, segment_indices)):
                if i % 100 == 0:
                    print(f"  Procesando {i}/{len(paths)}...")
                emb = extract_yamnet_embeddings_from_segment(
                    path, int(seg_idx), SEGMENT_DURATION, OVERLAP_SECONDS
                )
                all_embeddings.append(emb)

        yamnet_extraction_time = get_extraction_time().seconds

        # Guardar en cache si no se cargó de cache
        if not cache_loaded:
            save_embeddings_cache(
                all_embeddings,
                list(paths),
                list(segment_indices),
                DURATION_DIR,
                SEGMENT_DURATION,
                OVERLAP_RATIO,
                OVERLAP_SECONDS,
            )
    else:
        yamnet_extraction_time = 0  # Se cargó desde cache

    print(f"Embeddings extraídos: {len(all_embeddings)}")

    # Preparar arrays
    y_plate = all_data["Plate Encoded"].values
    y_electrode = all_data["Electrode Encoded"].values
    y_current = all_data["Current Encoded"].values
    sessions = all_data["Session"].values
    y_stratify = y_electrode

    # ============= FASE 1: Entrenar K modelos =============
    print(f"\n{'=' * 70}")
    print(f"FASE 1: ENTRENAMIENTO DE {N_FOLDS} MODELOS (StratifiedGroupKFold)")
    print(f"{'=' * 70}")

    training_start_time = time.time()
    
    # Store the boundary between train and test data for k=1 case
    n_train_rows = len(train_data)

    # Prepare folds: k=1 uses original train/test split, k>=2 uses StratifiedGroupKFold
    if N_FOLDS == 1:
        # Single train/test split without cross-validation
        train_idx = np.arange(n_train_rows)
        val_idx = np.arange(n_train_rows, len(all_data))
        fold_splits = [(train_idx, val_idx)]
    else:
        sgkf = StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )
        fold_splits = list(sgkf.split(all_embeddings, y_stratify, groups=sessions))

    # Checkpoint: soporte de pausa/reanudación por fold
    setup_pause_handler()
    ckpt = TrainingCheckpoint(MODELS_DIR)
    ckpt_state = ckpt.load()

    if ckpt_state:
        start_fold = len(ckpt_state["completed_folds"])
        fold_metrics = list(ckpt_state["fold_results"])
        fold_best_epochs = list(ckpt_state["fold_best_epochs"])
        fold_training_times = list(ckpt_state["fold_training_times"])
        all_fold_histories = list(ckpt_state["fold_histories"])
        _pause_count = ckpt_state.get("pause_count", 0)
        _resumed_from_fold = start_fold
        print(f"[RESUME] Resumiendo desde fold {start_fold + 1}/{N_FOLDS}")
    else:
        ckpt_state = ckpt.initialize()
        start_fold = 0
        fold_metrics = []
        fold_best_epochs = []
        fold_training_times = []
        all_fold_histories = []
        _pause_count = 0
        _resumed_from_fold = None

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        if fold_idx < start_fold:
            continue
        train_sessions = set(sessions[train_idx])
        val_sessions = set(sessions[val_idx])
        assert len(train_sessions & val_sessions) == 0, "ERROR: Sesiones mezcladas!"

        print(f"\nFold {fold_idx + 1}/{N_FOLDS}")
        print(f"  Train: {len(train_idx)} segmentos ({len(train_sessions)} sesiones)")
        print(f"  Val: {len(val_idx)} segmentos ({len(val_sessions)} sesiones)")

        train_embeddings = [all_embeddings[i] for i in train_idx]
        val_embeddings = [all_embeddings[i] for i in val_idx]

        train_labels = {
            "plate": y_plate[train_idx],
            "electrode": y_electrode[train_idx],
            "current": y_current[train_idx],
        }
        val_labels = {
            "plate": y_plate[val_idx],
            "electrode": y_electrode[val_idx],
            "current": y_current[val_idx],
        }

        class_weights = {
            "plate": compute_class_weight(
                "balanced",
                classes=np.unique(train_labels["plate"]),
                y=train_labels["plate"],
            ),
            "electrode": compute_class_weight(
                "balanced",
                classes=np.unique(train_labels["electrode"]),
                y=train_labels["electrode"],
            ),
            "current": compute_class_weight(
                "balanced",
                classes=np.unique(train_labels["current"]),
                y=train_labels["current"],
            ),
        }

        fold_start_time = time.time()
        with timer(f"Entrenar fold {fold_idx + 1}/{N_FOLDS}"):
            metrics, best_epoch, fold_history = train_one_fold(
                fold_idx,
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                class_weights,
                (plate_encoder, electrode_encoder, current_type_encoder),
                device,
                MODELS_DIR,
            )
        fold_time = time.time() - fold_start_time
        
        metrics["time_seconds"] = round(fold_time, 2)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)
        fold_best_epochs.append(best_epoch)
        fold_training_times.append(round(fold_time, 2))
        all_fold_histories.append(fold_history)
        ckpt.save_fold(ckpt_state, fold_idx, metrics, fold_time, best_epoch, fold_history)
        if pause_requested():
            ckpt.mark_paused(ckpt_state)
            print(f"[PAUSE] Pausado después del fold {fold_idx + 1}/{N_FOLDS}. Re-ejecuta el mismo comando para continuar.")
            sys.exit(0)

    training_end_time = time.time()
    training_time = sum(fold_training_times)
    training_time_minutes = training_time / 60
    print(
        f"\nTiempo de entrenamiento puro: {training_time:.2f}s ({training_time_minutes:.2f}min)"
    )

    # ============= FASE 2: Evaluar Ensemble =============
    print(f"\n{'=' * 70}")
    print("FASE 2: EVALUACIÓN DEL ENSEMBLE (Soft Voting)")
    print(f"{'=' * 70}")

    sample_model = ECAPAMultiTask(
        feat_dim=1024,
        ecapa_channels=1024,
        emb_dim=256,
        num_classes_espesor=len(plate_encoder.classes_),
        num_classes_electrodo=len(electrode_encoder.classes_),
        num_classes_corriente=len(current_type_encoder.classes_),
    )
    model_params = count_model_parameters(sample_model)
    del sample_model

    with timer("Cargar modelos del ensemble"):
        models = []
        for fold_idx in range(N_FOLDS):
            model = ECAPAMultiTask(
                feat_dim=1024,
                ecapa_channels=1024,
                emb_dim=256,
                num_classes_espesor=len(plate_encoder.classes_),
                num_classes_electrodo=len(electrode_encoder.classes_),
                num_classes_corriente=len(current_type_encoder.classes_),
            ).to(device)
            model.load_state_dict(torch.load(MODELS_DIR / f"model_fold_{fold_idx}.pth"))
            model.eval()
            models.append(model)
        print(f"Cargados {len(models)} modelos del ensemble")

    # Evaluar en todo el dataset
    all_preds = {"plate": [], "electrode": [], "current": []}
    all_labels = {"plate": [], "electrode": [], "current": []}

    full_dataset = AudioDataset(all_embeddings, y_plate, y_electrode, y_current)
    full_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad
    )

    with timer("Evaluación ensemble (Soft Voting)"):
        print("Evaluando ensemble en todo el dataset...")
        for embeddings, labels_p, labels_e, labels_c in full_loader:
            pred_p, pred_e, pred_c = ensemble_predict(models, embeddings, device)

            all_preds["plate"].extend(pred_p.cpu().numpy())
            all_preds["electrode"].extend(pred_e.cpu().numpy())
            all_preds["current"].extend(pred_c.cpu().numpy())
            all_labels["plate"].extend(labels_p.numpy())
            all_labels["electrode"].extend(labels_e.numpy())
            all_labels["current"].extend(labels_c.numpy())

    # Calcular métricas del ensemble
    acc_p = np.mean(np.array(all_preds["plate"]) == np.array(all_labels["plate"]))
    acc_e = np.mean(
        np.array(all_preds["electrode"]) == np.array(all_labels["electrode"])
    )
    acc_c = np.mean(np.array(all_preds["current"]) == np.array(all_labels["current"]))

    f1_p = f1_score(all_labels["plate"], all_preds["plate"], average="macro")
    f1_e = f1_score(all_labels["electrode"], all_preds["electrode"], average="macro")
    f1_c = f1_score(all_labels["current"], all_preds["current"], average="macro")

    prec_p = precision_score(
        all_labels["plate"], all_preds["plate"], average="macro"
    )
    prec_e = precision_score(
        all_labels["electrode"], all_preds["electrode"], average="macro"
    )
    prec_c = precision_score(
        all_labels["current"], all_preds["current"], average="macro"
    )

    rec_p = recall_score(all_labels["plate"], all_preds["plate"], average="macro")
    rec_e = recall_score(
        all_labels["electrode"], all_preds["electrode"], average="macro"
    )
    rec_c = recall_score(
        all_labels["current"], all_preds["current"], average="macro"
    )

    avg_acc_p = np.mean([m["accuracy_plate"] for m in fold_metrics])
    avg_acc_e = np.mean([m["accuracy_electrode"] for m in fold_metrics])
    avg_acc_c = np.mean([m["accuracy_current"] for m in fold_metrics])

    print(f"\n{'=' * 70}")
    print("RESULTADOS FINALES")
    print(f"{'=' * 70}")

    print("\nMétricas individuales por fold (promedio):")
    print(f"  Plate:     {avg_acc_p:.4f}")
    print(f"  Electrode: {avg_acc_e:.4f}")
    print(f"  Current:   {avg_acc_c:.4f}")

    print(f"\nMétricas del ENSEMBLE (Soft Voting, {N_FOLDS} modelos):")
    print(
        f"  Plate:     Acc={acc_p:.4f} | F1={f1_p:.4f} | Prec={prec_p:.4f} | Rec={rec_p:.4f}"
    )
    print(
        f"  Electrode: Acc={acc_e:.4f} | F1={f1_e:.4f} | Prec={prec_e:.4f} | Rec={rec_e:.4f}"
    )
    print(
        f"  Current:   Acc={acc_c:.4f} | F1={f1_c:.4f} | Prec={prec_c:.4f} | Rec={rec_c:.4f}"
    )

    print(f"\nMejora del Ensemble vs Promedio Individual:")
    print(f"  Plate:     {acc_p - avg_acc_p:+.4f}")
    print(f"  Electrode: {acc_e - avg_acc_e:+.4f}")
    print(f"  Current:   {acc_c - avg_acc_c:+.4f}")

    print(f"\n{'=' * 70}")
    print("REPORTES DE CLASIFICACIÓN")
    print(f"{'=' * 70}")

    print("\n--- Plate Thickness ---")
    print(
        classification_report(
            all_labels["plate"],
            all_preds["plate"],
            target_names=plate_encoder.classes_,
            zero_division=0,
        )
    )

    print("\n--- Electrode Type ---")
    print(
        classification_report(
            all_labels["electrode"],
            all_preds["electrode"],
            target_names=electrode_encoder.classes_,
            zero_division=0,
        )
    )

    print("\n--- Type of Current ---")
    print(
        classification_report(
            all_labels["current"],
            all_preds["current"],
            target_names=current_type_encoder.classes_,
            zero_division=0,
        )
    )

    print(f"\n{'=' * 70}")
    print("MATRICES DE CONFUSIÓN")
    print(f"{'=' * 70}")

    cm_plate = confusion_matrix(all_labels["plate"], all_preds["plate"])
    cm_electrode = confusion_matrix(all_labels["electrode"], all_preds["electrode"])
    cm_current = confusion_matrix(all_labels["current"], all_preds["current"])

    print("\nPlate Thickness:")
    print(cm_plate)
    print(f"Clases: {plate_encoder.classes_}")

    print("\nElectrode Type:")
    print(cm_electrode)
    print(f"Clases: {electrode_encoder.classes_}")

    print("\nType of Current:")
    print(cm_current)
    print(f"Clases: {current_type_encoder.classes_}")

    # Guardar resultados
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    elapsed_hours = elapsed_time / 3600

    segments_per_class = {
        "plate": all_data["Plate Thickness"].value_counts().to_dict(),
        "electrode": all_data["Electrode"].value_counts().to_dict(),
        "current": all_data["Type of Current"].value_counts().to_dict(),
    }

    new_entry = {
        "id": f"{int(SEGMENT_DURATION)}seg_{N_FOLDS}fold_overlap_{OVERLAP_RATIO}_ecapa_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "model_type": "ecapa",
        "backbone": "yamnet",
        "execution_time": {
            "seconds": round(elapsed_time, 2),
            "minutes": round(elapsed_minutes, 2),
            "hours": round(elapsed_hours, 4),
        },
        "training_time": {
            "seconds": round(training_time, 2),
            "minutes": round(training_time_minutes, 2),
        },
        "feature_extraction": {
            "from_cache": False,
            "extraction_time_seconds": round(yamnet_extraction_time, 2),
            "extraction_time_minutes": round(yamnet_extraction_time / 60, 2),
        },
        "config": {
            "duration": SEGMENT_DURATION,
            "overlap": OVERLAP_RATIO,
            "k_folds": N_FOLDS,
            "models_dir": str(MODELS_DIR.name),
            "seed": RANDOM_SEED,
            "voting_method": "soft",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "swa_start": SWA_START,
            "swa_lr": 1e-4,
        },
        "system_info": get_system_info(device),
        "model_parameters": model_params,
        "data": {
            "total_segments": len(all_data),
            "unique_sessions": int(all_data["Session"].nunique()),
            "segments_per_class": segments_per_class,
            "classes": {
                "plate": list(plate_encoder.classes_),
                "electrode": list(electrode_encoder.classes_),
                "current": list(current_type_encoder.classes_),
            },
        },
        "fold_results": fold_metrics,
        "fold_best_epochs": fold_best_epochs,
        "fold_training_times_seconds": fold_training_times,
        "ensemble_results": {
            "plate": {
                "accuracy": round(acc_p, 4),
                "f1": round(f1_p, 4),
                "precision": round(prec_p, 4),
                "recall": round(rec_p, 4),
            },
            "electrode": {
                "accuracy": round(acc_e, 4),
                "f1": round(f1_e, 4),
                "precision": round(prec_e, 4),
                "recall": round(rec_e, 4),
            },
            "current": {
                "accuracy": round(acc_c, 4),
                "f1": round(f1_c, 4),
                "precision": round(prec_c, 4),
                "recall": round(rec_c, 4),
            },
        },
        "individual_avg": {
            "plate": round(avg_acc_p, 4),
            "electrode": round(avg_acc_e, 4),
            "current": round(avg_acc_c, 4),
        },
        "improvement_vs_individual": {
            "plate": round(acc_p - avg_acc_p, 4),
            "electrode": round(acc_e - avg_acc_e, 4),
            "current": round(acc_c - avg_acc_c, 4),
        },
        "training_history": all_fold_histories,
        "pause_resume": {
            "was_paused": ckpt_state.get("was_paused", False),
            "pause_count": _pause_count,
            "resumed_from_fold": _resumed_from_fold,
        },
    }

    # Cargar historial existente o crear nuevo
    results_path = DURATION_DIR / "resultados.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = [history]
    else:
        history = []

    history.append(new_entry)

    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)

    ckpt.delete()

    print(f"\nResultados guardados en: {results_path} (entrada #{len(history)})")
    print(f"Modelos guardados en: {MODELS_DIR}/")
    print(
        f"\nTiempo de ejecución: {elapsed_time:.2f}s ({elapsed_minutes:.2f}min / {elapsed_hours:.4f}h)"
    )
    print(f"Logs guardados en: {log_path}")
    
    # Close log file
    log_file.close()


if __name__ == "__main__":
    main()
