#!/usr/bin/env python3
"""
Mide el tiempo de extracción de embeddings YAMNet para cada duración.
Genera embeddings_extraction_times.json con los tiempos medidos.

Uso:
    python medir_extraccion.py
    python medir_extraccion.py --duraciones 10 20
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub

sys.path.insert(0, str(Path(__file__).parent))
from utils.audio_utils import PROJECT_ROOT, load_audio_segment

warnings.filterwarnings("ignore")

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
OVERLAP_RATIO = 0.5
DURACIONES = [1, 2, 5, 10, 20, 30, 50]
PROJECT_DIR = Path(__file__).parent
OUTPUT_FILE = PROJECT_DIR / "embeddings_extraction_times.json"


def extract_yamnet_embeddings_aggregated(
    audio_path: str,
    segment_idx: int,
    segment_duration: float,
    overlap_seconds: float,
    yamnet_model,
) -> np.ndarray:
    """Extrae embeddings YAMNet agregados (mean+std) de un segmento."""
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

    embeddings = np.stack(embeddings_list, axis=0)
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    return np.concatenate([mean, std], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Medir tiempo de extracción YAMNet")
    parser.add_argument(
        "--duraciones",
        type=int,
        nargs="+",
        default=DURACIONES,
        help="Duraciones a medir (default: todas)",
    )
    args = parser.parse_args()

    print("Cargando modelo YAMNet desde TensorFlow Hub...")
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    print("Modelo YAMNet cargado.\n")

    # Cargar resultados existentes si hay
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
    else:
        results = []

    # Duraciones ya medidas
    measured = {r["segment_duration"] for r in results}

    for duration in args.duraciones:
        dur_key = f"{duration:02d}seg" if duration < 10 else f"{duration}seg"
        csv_path = PROJECT_DIR / dur_key / f"completo_overlap_{OVERLAP_RATIO}.csv"

        if not csv_path.exists():
            print(f"[SKIP] {dur_key}: No existe {csv_path}")
            continue

        if duration in measured:
            print(f"[SKIP] {dur_key}: Ya medido previamente")
            continue

        df = pd.read_csv(csv_path)
        paths = df["Audio Path"].values
        segment_indices = df["Segment Index"].values
        overlap_seconds = duration * OVERLAP_RATIO
        n_segments = len(df)

        print(f"{'='*60}")
        print(f"Duración: {duration}s ({n_segments} segmentos)")
        print(f"{'='*60}")

        start_time = time.perf_counter()

        for i, (path, seg_idx) in enumerate(zip(paths, segment_indices)):
            if i % 200 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Procesando {i}/{n_segments}... ({elapsed:.1f}s)")
            extract_yamnet_embeddings_aggregated(
                path, int(seg_idx), duration, overlap_seconds, yamnet_model
            )

        extraction_time = round(time.perf_counter() - start_time, 2)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "segment_duration": duration,
            "overlap_ratio": OVERLAP_RATIO,
            "overlap_seconds": overlap_seconds,
            "num_segments": n_segments,
            "num_embeddings": n_segments,
            "extraction_time_seconds": extraction_time,
            "extraction_time_minutes": round(extraction_time / 60, 2),
        }
        results.append(entry)

        # Guardar después de cada duración (por si interrumpen)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  -> {extraction_time}s ({extraction_time/60:.2f}min)")
        print(f"  Guardado en {OUTPUT_FILE}\n")

    print("\nResultados finales:")
    for r in sorted(results, key=lambda x: x["segment_duration"]):
        print(
            f"  {r['segment_duration']:2d}s: {r['extraction_time_seconds']:.2f}s "
            f"({r['extraction_time_minutes']:.2f}min) - {r['num_segments']} segmentos"
        )


if __name__ == "__main__":
    main()
