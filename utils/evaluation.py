#!/usr/bin/env python3
"""
Funciones compartidas de evaluación y ensemble para clasificación SMAW.

Proporciona funciones reutilizables para:
- Predicción con soft voting de múltiples modelos
- Cálculo de métricas de evaluación
- Construcción de matrices de confusión
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def ensemble_predict(models, embeddings, device):
    """Realiza predicciones usando soft voting (averaging de logits).
    
    Args:
        models: Lista de modelos entrenados
        embeddings: Tensor de embeddings
        device: Dispositivo (cuda/cpu)
    
    Returns:
        Tupla de (predicciones_plate, predicciones_electrode, predicciones_current)
    """
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

    return pred_plate.cpu().numpy(), pred_electrode.cpu().numpy(), pred_current.cpu().numpy()


def calculate_metrics(y_true, y_pred, task_name=""):
    """Calcula accuracy, F1, precision y recall para una tarea.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        task_name: Nombre de la tarea (para logging)
        
    Returns:
        Diccionario con métricas
    """
    accuracy = float(np.mean(y_true == y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    precision = float(precision_score(y_true, y_pred, average="macro"))
    recall = float(recall_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm,
    }


def calculate_multi_task_metrics(y_true_dict, y_pred_dict):
    """Calcula métricas para múltiples tareas simultáneamente.
    
    Args:
        y_true_dict: Diccionario {task: labels}
        y_pred_dict: Diccionario {task: predictions}
        
    Returns:
        Diccionario con métricas por tarea
    """
    results = {}
    for task in y_true_dict.keys():
        results[task] = calculate_metrics(y_true_dict[task], y_pred_dict[task], task_name=task)
    
    # Calcular exact match (todas las tareas correctas)
    exact_match = np.mean(
        (y_pred_dict["plate"] == y_true_dict["plate"]) &
        (y_pred_dict["electrode"] == y_true_dict["electrode"]) &
        (y_pred_dict["current"] == y_true_dict["current"])
    )
    
    # Calcular hamming accuracy (promedio de aciertos por tarea)
    hamming_accuracy = np.mean(
        (y_pred_dict["plate"] == y_true_dict["plate"]).astype(int) +
        (y_pred_dict["electrode"] == y_true_dict["electrode"]).astype(int) +
        (y_pred_dict["current"] == y_true_dict["current"]).astype(int)
    ) / 3
    
    results["global"] = {
        "exact_match": float(round(exact_match, 4)),
        "hamming_accuracy": float(round(hamming_accuracy, 4)),
    }
    
    return results
