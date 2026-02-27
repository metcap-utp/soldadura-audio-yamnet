"""
Utilidades de audio para clasificación SMAW.

Este módulo centraliza:
- Carga de audio desde la carpeta base `audio/`
- Segmentación on-the-fly según duración especificada (05seg, 10seg, 30seg)
- Extracción de etiquetas desde paths
- Agrupación por sesión para evitar data leakage

Estructura esperada de archivos:
    audio/Placa_Xmm/EXXXX/{AC,DC}/YYMMDD-HHMMSS_Audio/*.wav

¿Qué es una SESIÓN?
    Una sesión es una grabación continua de soldadura, identificada por
    su carpeta con timestamp (ej: 240912-143741_Audio). Todos los segmentos
    de la misma sesión DEBEN ir al mismo split para evitar data leakage.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

# Ruta base del proyecto (apunta al directorio de audio de vggish-backbone para evitar duplicar archivos)
YAMNET_ROOT = Path(__file__).parent.parent
SIBLING_ROOT = YAMNET_ROOT.parent / "vggish-backbone"  # Directorio hermano con archivos de audio
PROJECT_ROOT = SIBLING_ROOT  # Resolver paths de audio desde el proyecto hermano
AUDIO_BASE_DIR = SIBLING_ROOT / "audio"


def get_audio_base_dir() -> Path:
    """Retorna la ruta base del directorio de audio."""
    return AUDIO_BASE_DIR


def extract_labels_from_session_path(session_path: Path) -> Optional[Dict]:
    """
    Extrae etiquetas del path de una carpeta de sesión.

    Ejemplo path relativo desde PROJECT_ROOT:
        audio/Placa_6mm/E6011/AC/240802-105935_Audio

    Returns:
        dict con Plate Thickness, Electrode, Type of Current, Session
        None si el path no tiene la estructura esperada
    """
    try:
        relative_path = session_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    parts = relative_path.parts

    # Esperamos: audio/Placa_Xmm/EXXXX/AC|DC/TIMESTAMP_Audio
    if len(parts) != 5:
        return None

    try:
        plate_thickness = parts[1]  # Placa_Xmm
        electrode = parts[2]  # EXXXX
        current_type = parts[3]  # AC/DC
        session = parts[4]  # YYMMDD-HHMMSS_Audio

        return {
            "Session Path": str(relative_path),
            "Plate Thickness": plate_thickness,
            "Electrode": electrode,
            "Type of Current": current_type,
            "Session": session,
        }
    except (ValueError, IndexError):
        return None


def discover_sessions() -> pd.DataFrame:
    """
    Descubre todas las sesiones de audio disponibles.

    Returns:
        DataFrame con columnas:
        - Session Path: ruta relativa de la carpeta de sesión
        - Plate Thickness, Electrode, Type of Current, Session
    """
    sessions_data = []

    # Buscar carpetas que terminen en _Audio
    for session_dir in AUDIO_BASE_DIR.rglob("*_Audio"):
        if not session_dir.is_dir():
            continue

        labels = extract_labels_from_session_path(session_dir)
        if labels:
            # Contar archivos WAV en la sesión
            wav_files = list(session_dir.glob("*.wav"))
            labels["Num Files"] = len(wav_files)
            if wav_files:
                sessions_data.append(labels)

    df = pd.DataFrame(sessions_data)
    return df


def get_session_audio_files(session_path: str) -> List[Path]:
    """
    Obtiene todos los archivos WAV de una sesión.

    Args:
        session_path: Ruta relativa de la sesión (ej: audio/Placa_6mm/E6011/AC/240802_Audio)

    Returns:
        Lista de Paths a archivos WAV
    """
    full_path = PROJECT_ROOT / session_path
    if not full_path.exists():
        return []
    return sorted(full_path.glob("*.wav"))


def load_audio_segment(
    audio_path: Path,
    segment_duration: float,
    segment_index: int,
    sr: int = 16000,
    hop_ratio: Optional[float] = None,
    overlap_seconds: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Carga un segmento específico de un archivo de audio.

    Args:
        audio_path: Ruta al archivo WAV
        segment_duration: Duración del segmento en segundos
        segment_index: Índice del segmento a extraer
        sr: Sample rate
        hop_ratio: Ratio de hop (0.5 = 50% overlap)

    Returns:
        Array numpy con el audio del segmento. Si el índice excede la duración,
        retorna un segmento con padding de ceros.
    """
    try:
        # Cargar audio completo
        y, _ = librosa.load(audio_path, sr=sr, mono=True)

        segment_samples = int(segment_duration * sr)

        if hop_ratio is None:
            overlap_samples = int(max(0.0, overlap_seconds) * sr)
            hop_samples = max(1, segment_samples - overlap_samples)
        else:
            hop_samples = max(1, int(segment_samples * hop_ratio))

        start = segment_index * hop_samples
        end = start + segment_samples

        if start >= len(y):
            return np.zeros(segment_samples, dtype=np.float32)

        # Si el segmento excede el final, hacer padding con zeros
        if end > len(y):
            segment = np.zeros(segment_samples, dtype=np.float32)
            segment[: len(y) - start] = y[start:]
        else:
            segment = y[start:end]

        return segment.astype(np.float32)

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def count_segments_in_file(
    audio_path: Path,
    segment_duration: float,
    sr: int = 16000,
    hop_ratio: Optional[float] = None,
    overlap_seconds: float = 0.0,
) -> int:
    """
    Cuenta cuántos segmentos se pueden extraer de un archivo.

    Args:
        audio_path: Ruta al archivo WAV
        segment_duration: Duración del segmento en segundos
        sr: Sample rate
        hop_ratio: Ratio de hop

    Returns:
        Número de segmentos
    """
    try:
        duration = librosa.get_duration(path=audio_path)

        if hop_ratio is None:
            hop_seconds = max(1e-6, segment_duration - max(0.0, overlap_seconds))
        else:
            hop_seconds = max(1e-6, segment_duration * hop_ratio)

        if duration < segment_duration:
            return 1  # Al menos un segmento con padding

        num_segments = int((duration - segment_duration) / hop_seconds) + 1
        return max(1, num_segments)

    except Exception:
        return 0


def count_segments_in_session(
    session_path: str,
    segment_duration: float,
    sr: int = 16000,
    hop_ratio: Optional[float] = None,
    overlap_seconds: float = 0.0,
) -> int:
    """
    Cuenta el total de segmentos en una sesión.

    Args:
        session_path: Ruta relativa de la sesión
        segment_duration: Duración del segmento en segundos
        sr: Sample rate
        hop_ratio: Ratio de hop

    Returns:
        Número total de segmentos en la sesión
    """
    audio_files = get_session_audio_files(session_path)
    total = 0
    for audio_file in audio_files:
        total += count_segments_in_file(
            audio_file,
            segment_duration,
            sr=sr,
            hop_ratio=hop_ratio,
            overlap_seconds=overlap_seconds,
        )
    return total


def get_all_segments_from_session(
    session_path: str,
    segment_duration: float,
    sr: int = 16000,
    hop_ratio: Optional[float] = None,
    overlap_seconds: float = 0.0,
) -> List[Tuple[Path, int]]:
    """
    Obtiene lista de (archivo, índice_segmento) para todos los segmentos de una sesión.

    Args:
        session_path: Ruta relativa de la sesión
        segment_duration: Duración del segmento en segundos
        sr: Sample rate
        hop_ratio: Ratio de hop

    Returns:
        Lista de tuplas (Path al archivo, índice del segmento)
    """
    audio_files = get_session_audio_files(session_path)
    segments = []

    for audio_file in audio_files:
        num_segs = count_segments_in_file(
            audio_file,
            segment_duration,
            sr=sr,
            hop_ratio=hop_ratio,
            overlap_seconds=overlap_seconds,
        )
        for seg_idx in range(num_segs):
            segments.append((audio_file, seg_idx))

    return segments


def create_stratification_label(row) -> str:
    """Crea etiqueta combinada para estratificación."""
    return f"{row['Plate Thickness']}_{row['Electrode']}_{row['Type of Current']}"


def parse_segment_duration_from_dir(dir_name: str) -> float:
    """
    Parsea la duración del segmento desde el nombre del directorio.

    Args:
        dir_name: Nombre del directorio (ej: "05seg", "10seg", "30seg")

    Returns:
        Duración en segundos
    """
    match = re.match(r"(\d+)seg", dir_name)
    if match:
        return float(match.group(1))
    return 5.0  # Default


def get_script_segment_duration(script_path: Path) -> float:
    """
    Obtiene la duración del segmento basándose en el directorio del script.

    Args:
        script_path: Path del script (ej: /path/to/05seg/entrenar.py)

    Returns:
        Duración en segundos
    """
    dir_name = script_path.parent.name
    return parse_segment_duration_from_dir(dir_name)
