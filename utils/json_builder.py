#!/usr/bin/env python3
"""
Utilidades compartidas para construcción de resultados JSON.

Proporciona funciones para:
- Formateo estándar de resultados entrenamiento
- Cálculo de estadísticas de ejecución
- Validación de esquema JSON
"""

import platform
import torch
from datetime import datetime


def get_system_info(device):
    """Recolecta información del sistema y dependencias."""
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": None,  # Importado dinámicamente donde se usa
        "platform": platform.platform(),
        "device": str(device),
    }
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        pass
    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
    except Exception:
        pass
    if hasattr(device, 'type') and device.type == "cuda":
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


def format_timing(elapsed_time):
    """Formatea tiempo de ejecución en múltiples unidades.
    
    Args:
        elapsed_time: Tiempo en segundos
        
    Returns:
        Diccionario con segundos, minutos y horas redondeados
    """
    return {
        "seconds": round(elapsed_time, 2),
        "minutes": round(elapsed_time / 60, 2),
        "hours": round(elapsed_time / 3600, 4),
    }


def create_result_entry(
    segment_duration,
    n_folds,
    overlap_ratio,
    model_type,
    backbone,
    config,
    system_info,
    model_params,
    data_info,
    fold_results,
    fold_best_epochs,
    fold_training_times,
    ensemble_results,
    individual_avg,
    improvement_vs_individual,
    training_history,
    execution_time,
    training_time,
):
    """Crea entrada de resultado con esquema canónico.
    
    Args:
        segment_duration: Duración de segmento en segundos
        n_folds: Número de folds
        overlap_ratio: Ratio de overlap
        model_type: Tipo de modelo (ecapa, xvector, feedforward)
        backbone: Tipo de embedding (vggish, yamnet, spectral-mfcc)
        config: Diccionario con configuración
        system_info: Información del sistema
        model_params: Parámetros del modelo
        data_info: Información de datos
        fold_results: Resultados por fold
        fold_best_epochs: Mejor época por fold
        fold_training_times: Tiempo de entrenamiento por fold
        ensemble_results: Resultados ensemble
        individual_avg: Promedio individual
        improvement_vs_individual: Mejora vs individual
        training_history: Historia de entrenamiento
        execution_time: Tiempo total de ejecución
        training_time: Tiempo acumulado de entrenamiento
        
    Returns:
        Diccionario con entrada formateada
    """
    return {
        "id": f"{int(segment_duration)}seg_{n_folds}fold_overlap_{overlap_ratio}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "backbone": backbone,
        "execution_time": execution_time,
        "training_time": training_time,
        "config": config,
        "system_info": system_info,
        "model_parameters": model_params,
        "data": data_info,
        "fold_results": fold_results,
        "fold_best_epochs": fold_best_epochs,
        "fold_training_times_seconds": fold_training_times,
        "ensemble_results": ensemble_results,
        "individual_avg": individual_avg,
        "improvement_vs_individual": improvement_vs_individual,
        "training_history": training_history,
    }
