#!/usr/bin/env python3
"""
Generador de splits para yamnet-backbone.

Este script es un wrapper que utiliza generar_splits.py del proyecto vggish-backbone,
ya que ambos proyectos utilizan los mismos archivos de audio.
"""

import sys
from pathlib import Path

# Agregar vggish-backbone al path para importar su generar_splits
vggish_path = Path(__file__).parent.parent / "vggish-backbone"
sys.path.insert(0, str(vggish_path))

# Cambiar el directorio de trabajo por si acaso
import os
original_cwd = os.getcwd()

try:
    # Importar y ejecutar el módulo de vggish
    from generar_splits import main
    
    if __name__ == "__main__":
        main()
finally:
    os.chdir(original_cwd)
