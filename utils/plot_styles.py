"""
Módulo de estilos para gráficas.

Contiene la configuración de matplotlib y constantes de colores,
marcadores y etiquetas para mantener consistencia visual en todas
las gráficas del proyecto.

Basado en los estilos del proyecto graficas_tesis.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from matplotlib.figure import Figure

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.markersize": 6,
    "lines.linewidth": 1.5,
})

COLORS = {
    "plate": "#60a5fa",
    "electrode": "#3b82f6",
    "current": "#1d4ed8",
    "exact_match": "#8e44ad",
    "hamming": "#16a085",
}

MARKERS = {
    "accuracy": "o",
    "f1": "s",
    "plate": "o",
    "electrode": "s",
    "current": "^",
}

LINESTYLES = {
    "accuracy": "-",
    "f1": "--",
}

TASKS = ["plate", "electrode", "current"]

TASK_LABELS = {
    "es": {
        "plate": "Espesor de Placa",
        "electrode": "Tipo de Electrodo",
        "current": "Tipo de Corriente",
    },
    "en": {
        "plate": "Plate Thickness",
        "electrode": "Electrode Type",
        "current": "Current Type",
    },
}

TASK_LABELS_SHORT = {
    "plate": "Placa",
    "electrode": "Electrodo",
    "current": "Corriente",
}

DPI = 300


def get_smart_ylim(values: List[float]) -> tuple:
    """Calcula límites Y inteligentes basados en los datos."""
    if not values:
        return 0.0, 1.0

    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    if not valid_values:
        return 0.0, 1.0

    min_val = min(valid_values)
    max_val = max(valid_values)

    if max_val > 0.95:
        upper = 1.0
    else:
        upper = min(1.0, max_val + 0.05)

    if min_val > 0.9:
        lower = max(0, min_val - 0.1)
    elif min_val > 0.7:
        lower = max(0, min_val - 0.15)
    elif min_val > 0.5:
        lower = max(0, min_val - 0.2)
    else:
        lower = max(0, min_val - 0.1)

    return lower, upper


def setup_axis(
    ax: plt.Axes,
    x_values: List[int],
    y_values: Optional[List[float]] = None,
    smart_ylim: bool = True,
) -> None:
    """Configura el eje X e Y de manera consistente."""
    ax.set_xticks(x_values)
    ax.set_xlim(min(x_values) - 2, max(x_values) + 2)

    if y_values and smart_ylim:
        y_lower, y_upper = get_smart_ylim(y_values)
        ax.set_ylim(y_lower, y_upper)
    elif smart_ylim:
        ax.set_ylim(0.0, 1.0)

    ax.set_yticks(np.arange(0, 1.1, 0.1))


def add_config_annotation(
    ax: plt.Axes,
    n_folds: int = 10,
    overlap: float = 0.5,
    duration: Optional[str] = None,
) -> None:
    """Agrega anotación de configuración en la esquina superior izquierda."""
    text = f"k={n_folds}"
    if overlap is not None:
        text += f", overlap={overlap}"
    if duration:
        text += f", {duration}"

    ax.annotate(
        text,
        xy=(0.01, 0.98),
        xycoords="axes fraction",
        fontsize=8,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def save_fig(fig: Figure, output_path: Path, filename: str, dpi: int = DPI) -> Path:
    """Guarda la figura con configuración consistente."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    fig.savefig(
        filepath,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close(fig)
    return filepath


def adjust_ylim(ax: plt.Axes, values: List[float]) -> None:
    """Ajusta los límites del eje Y según los datos."""
    if values:
        valid = [v for v in values if v is not None and not np.isnan(v)]
        if valid:
            y_min = min(valid)
            y_max = max(valid)
            span = y_max - y_min
            pad = max(0.02, span * 0.15)
            ax.set_ylim(max(0.0, y_min - pad), min(1.05, y_max + pad))
        else:
            ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(0, 1.05)

    if ax.get_ylim()[1] >= 1.0:
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.5)
