#!/usr/bin/env python3
"""
Script de migración: Convierte JSON de resultados antiguos al esquema canónico.

Uso:
    python migrar_resultados.py --all          # Migrar todos los JSONs
    python migrar_resultados.py --duration 10  # Migrar solo 10seg
    python migrar_resultados.py --backup       # Crear respaldos sin migrar
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


def migrate_yamnet_result(old_entry):
    """Convierte entrada antigua de yamnet al esquema canónico."""
    
    new_entry = {
        "timestamp": old_entry.get("timestamp", datetime.now().isoformat()),
        "model_type": old_entry.get("model_type", "unknown"),
        "backbone": "yamnet",
    }
    
    # Mapear configuración
    if "config" in old_entry:
        new_entry["config"] = old_entry["config"]
    
    # Extraer resultados
    if "results" in old_entry:
        results = old_entry["results"]
        if isinstance(results, dict) and "ensemble_results" in results:
            new_entry["ensemble_results"] = results["ensemble_results"]
        elif isinstance(results, dict) and "plate" in results:
            # Convertir formato antiguo directo
            new_entry["ensemble_results"] = results
    
    # Copiar fold_results si existe, renombrar claves de accuracy
    if "fold_results" in old_entry:
        fold_results = old_entry["fold_results"]
        # Renombrar claves antiguas si es necesario
        for fold in fold_results:
            if "acc_plate" in fold:
                fold["accuracy_plate"] = fold.pop("acc_plate")
            if "acc_electrode" in fold:
                fold["accuracy_electrode"] = fold.pop("acc_electrode")
            if "acc_current" in fold:
                fold["accuracy_current"] = fold.pop("acc_current")
        new_entry["fold_results"] = fold_results
    
    # Copiar campos opcionales
    for field in ["fold_best_epochs", "fold_training_times_seconds", 
                  "improvement_vs_individual", "individual_avg", 
                  "system_info", "model_parameters", "data", "training_history"]:
        if field in old_entry:
            new_entry[field] = old_entry[field]
    
    # Generar ID si no existe
    if "id" not in new_entry:
        config = new_entry.get("config", {})
        duration = config.get("segment_duration", config.get("duration", "?"))
        model = new_entry.get("model_type", "unknown")
        fold_count = config.get("n_folds", config.get("k_folds", "?"))
        overlap = config.get("overlap_ratio", config.get("overlap", "?"))
        new_entry["id"] = f"{duration}seg_{fold_count}fold_overlap_{overlap}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return new_entry


def migrate_json_file(json_path, project_type):
    """Migra un archivo JSON completo.
    
    Args:
        json_path: Ruta al archivo resultados.json
        project_type: 'yamnet'
    
    Returns:
        Lista con entradas migradas
    """
    print(f"Leyendo {json_path}...")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    migrated = []
    for entry in data:
        try:
            new_entry = migrate_yamnet_result(entry)
            migrated.append(new_entry)
        except Exception as e:
            print(f"  ⚠ Error al migrar entrada: {e}")
            migrated.append(entry)  # Mantener entrada original si hay error
    
    return migrated


def create_backup(json_path):
    """Crea respaldo automático del archivo original."""
    backup_path = json_path.parent / f"{json_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy(json_path, backup_path)
    print(f"  ✓ Respaldo creado: {backup_path}")
    return backup_path


def process_duration_dir(duration_dir, create_backup_only=False):
    """Procesa directorio de duración específica."""
    json_path = duration_dir / "resultados.json"
    
    if not json_path.exists():
        return None
    
    print(f"\nProcesando {duration_dir.name}...")
    
    # Crear respaldo
    create_backup(json_path)
    
    if create_backup_only:
        print(f"  (modo respaldo solamente)")
        return None
    
    # Migrar
    migrated = migrate_json_file(json_path, "yamnet")
    
    # Guardar
    with open(json_path, "w") as f:
        json.dump(migrated, f, indent=2)
    
    print(f"  ✓ Migrado: {len(migrated)} entradas")
    return len(migrated)


def main():
    parser = argparse.ArgumentParser(
        description="Migrar JSON de resultados al esquema canónico"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrar todos los archivos JSON"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Migrar solo duración específica (ej: 10)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Solo crear respaldos, no migrar"
    )
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    
    print(f"\n{'='*60}")
    print(f"Migración de JSON - Proyecto: yamnet-backbone")
    print(f"{'='*60}")
    
    durations = None
    if args.duration:
        durations = [f"{args.duration:02d}seg"]
        print(f"Duración específica: {args.duration}s")
    else:
        # Detectar duraciones disponibles
        durations = sorted([d.name for d in base_dir.iterdir() 
                          if d.is_dir() and d.name.endswith("seg")])
        if args.all:
            print(f"Migrando TODAS las duraciones encontradas: {durations}")
        else:
            print(f"Duraciones disponibles (usa --all para migrar todas): {durations}")
            print("Uso: python migrar_resultados.py --all")
            return
    
    if args.backup:
        print("Modo: Crear respaldos solamente (sin migrar)\n")
    else:
        print("Modo: Migrar esquema\n")
    
    total_entries = 0
    for duration in durations:
        duration_dir = base_dir / duration
        if duration_dir.exists():
            count = process_duration_dir(
                duration_dir,
                create_backup_only=args.backup
            )
            if count is not None:
                total_entries += count
    
    print(f"\n{'='*60}")
    if args.backup:
        print(f"Respaldos creados exitosamente")
    else:
        print(f"Migración completada: {total_entries} entradas procesadas")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
