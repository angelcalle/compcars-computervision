#!/usr/bin/env python3
# 01_prepare_splits.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 1: PREPARACIÓN DE DATOS (+ VERIFICACIÓN)
# ==============================================================================
#
# OBJETIVO:
#   1. Escanear todas las imágenes del dataset y preparar las listas de
#      entrenamiento (train_all.txt) y prueba (test_all.txt).
#   2. Verificar automáticamente que los archivos generados son válidos y
#      las imágenes existen en el disco.
#
# LOGICA PRINCIPAL:
#   - Busca imágenes .jpg recursivamente.
#   - Extrae la "clase" (marca) de la estructura de carpetas (CompCars/image/78/...)
#   - Divide aleatoriamente (Split) en Train/Test.
#   - Comprueba que las rutas escritas son accesibles.
#
# USO:
#   Ejecución completa:
#     python 01_prepare_splits.py --compcars ./CompCars --outdir .
#
#   Ejecución rápida (Demo clase):
#     python 01_prepare_splits.py --compcars ./CompCars --max-makes 5 --per-make 100 --top-makes

import os
import glob
import argparse
import random
from collections import defaultdict, Counter

def check_splits(image_root, split_file):
    """
    Verifica que las rutas listadas en el archivo split_file existen.
    """
    print(f"\n--> Verificando integridad de: {split_file}")
    if not os.path.exists(split_file):
        print("ERROR: El archivo no existe.")
        return False

    total = 0
    missing = 0
    examples = []

    with open(split_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rel = line.strip()
            if not rel: continue
            total += 1

            # Construir ruta absoluta
            parts = rel.replace("\\", "/").split("/")
            full_path = os.path.join(image_root, *parts)

            if not os.path.isfile(full_path):
                missing += 1
                if len(examples) < 5:
                    examples.append(full_path)

    if missing == 0:
        print(f"   OK. {total} imágenes verificadas correctamente.")
        return True
    else:
        print(f"   ERROR. Faltan {missing} de {total} imágenes.")
        print("   Ejemplos faltantes:", examples)
        return False

def main():
    ap = argparse.ArgumentParser("Preparación y Verificación de Splits")
    ap.add_argument("--compcars", required=True, help="Ruta a carpeta CompCars")
    ap.add_argument("--outdir", default=".", help="Donde guardar la carpeta splits/")

    # Filtros para demos rápidas
    ap.add_argument("--max-makes", type=int, default=0, help="Máx marcas (0=todas)")
    ap.add_argument("--per-make", type=int, default=0, help="Máx fotos por marca (0=todas)")
    ap.add_argument("--top-makes", action="store_true", help="Si limitas marcas, coge las más frecuentes")

    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    ap.add_argument("--test-ratio", type=float, default=0.2, help="20% para test")

    ap.add_argument("--skip-check", action="store_true", help="Saltar verificación final")

    args = ap.parse_args()

    image_root = os.path.join(args.compcars, "image")
    if not os.path.isdir(image_root):
        print(f"Error: No encuentro 'image' dentro de {args.compcars}")
        return

    print(f"--> Escaneando imágenes en {image_root} (puede tardar un poco)...")
    # Búsqueda recursiva de archivos .jpg en todas las subcarpetas
    search_pattern = os.path.join(image_root, "**", "*.jpg")
    all_files = glob.glob(search_pattern, recursive=True)

    print(f"--> Encontrados {len(all_files)} archivos .jpg")
    if len(all_files) == 0:
        print("Error: No se encontraron imágenes.")
        return

    # Estructura: make_id -> lista de rutas relativas
    by_make = defaultdict(list)

    for fpath in all_files:
        # Obtener ruta relativa para guardar en el .txt
        # os.path.relpath calcula el camino desde image_root hasta el archivo
        rel = os.path.relpath(fpath, start=image_root)

        # Normalizar a barras unix para compatibilidad
        rel = rel.replace("\\", "/")

        # Extraer marca (primer directorio)
        parts = rel.split("/")
        if len(parts) >= 1:
            make_id = parts[0]
            by_make[make_id].append(rel)

    makes = list(by_make.keys())
    print(f"--> Marcas encontradas: {len(makes)}")

    # FILTRADO (para demos)
    if args.max_makes > 0:
        if args.top_makes:
            # Ordenar por cantidad de fotos
            makes.sort(key=lambda m: len(by_make[m]), reverse=True)
        else:
            # Ordenar numéricamente si es posible
            makes.sort(key=lambda x: int(x) if x.isdigit() else x)

        makes = makes[:args.max_makes]
        print(f"--> Filtrado a Top-{args.max_makes} marcas.")

    # DIVISIÓN TRAIN / TEST
    train_list = []
    test_list = []

    rng = random.Random(args.seed)

    for mk in makes:
        images = by_make[mk]

        # Mezclar aleatoriamente
        rng.shuffle(images)

        # Limitar número de imágenes por marca
        if args.per_make > 0:
            images = images[:args.per_make]

        # Calcular punto de corte
        n_test = int(len(images) * args.test_ratio)
        if len(images) > 1 and n_test == 0: n_test = 1 # Asegurar al menos 1 de test si hay pocas

        test_imgs = images[:n_test]
        train_imgs = images[n_test:]

        test_list.extend(test_imgs)
        train_list.extend(train_imgs)

    print(f"--> Total Final: {len(train_list)} Train, {len(test_list)} Test")

    # GUARDAR
    out_splits = os.path.join(args.outdir, "01_splits")
    os.makedirs(out_splits, exist_ok=True)

    train_file = os.path.join(out_splits, "train_all.txt")
    test_file = os.path.join(out_splits, "test_all.txt")

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_list))

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("\n".join(test_list))

    print(f"--> Listas guardadas en {os.path.abspath(out_splits)}")

    # VERIFICACIÓN AUTOMÁTICA
    if not args.skip_check:
        print("\n=== AUTO-VERIFICACIÓN DE INTEGRIDAD ===")
        ok1 = check_splits(image_root, train_file)
        ok2 = check_splits(image_root, test_file)

        if ok1 and ok2:
            print("\nTODO LISTO: Los archivos generados son 100% válidos.")
        else:
            print("\nADVERTENCIA: Hubo errores en la verificación.")

if __name__ == "__main__":
    main()
