#!/usr/bin/env python3
# 03_predict_forest.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 3b: INFERENCIA (Usar el modelo para predecir)
# ==============================================================================
#
# OBJETIVO:
#   Cargar el modelo RandomForest entrenado y probarlo con una imagen suelta.
#   Muestra cómo usar el modelo en "producción".
#
# USO:
#   python 03_predict_forest.py --image ./CompCars/image/78/2012/xxxxx.jpg --compcars ./CompCars

import os
import json
import argparse
import numpy as np
import cv2
import joblib

def extract_features(img_bgr, size=64, hist_bins=16):
    """
    IMPORTANTE: Debemos aplicar EXACTAMENTE el mismo preprocesado que usamos
    en el entrenamiento. Si cambiamos algo, el modelo no entenderá nada.
    """
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [hist_bins], [0, 256]).flatten()
        feats.append(hist)
    x = np.concatenate(feats).astype(np.float32)
    s = x.sum()
    if s > 0: x /= s
    return x

def load_make_map(compcars_root):
    """Carga nombres de marcas (Opcional)"""
    try:
        from scipy.io import loadmat
        p = os.path.join(compcars_root, "misc", "make_model_name.mat")
        if not os.path.isfile(p): return {}
        data = loadmat(p)
        if "make_names" in data:
            names = data["make_names"].squeeze()
            return {str(i): str(n[0]) if n.size>0 else "" for i,n in enumerate(names, 1)}
    except:
        pass
    return {}

def main():
    ap = argparse.ArgumentParser("Predecir imagen con RandomForest")
    ap.add_argument("--image", required=True, help="Ruta de la imagen a probar")
    ap.add_argument("--modeldir", default="03_models_forest", help="Carpeta donde está el modelo (.joblib)")
    ap.add_argument("--compcars", default="./CompCars", help="Ruta a CompCars (solo para sacar nombres de marcas)")
    args = ap.parse_args()

    # 1. Cargar artefactos del modelo
    path_model = os.path.join(args.modeldir, "forest_model.joblib")
    path_enc = os.path.join(args.modeldir, "label_encoder.joblib")
    path_cfg = os.path.join(args.modeldir, "feature_config.joblib")
    path_names = os.path.join(args.modeldir, "label_names.json")

    if not os.path.exists(path_model):
        print(f"Error: No encuentro el modelo en {path_model}. ¿Has ejecutado 03_fit_forest.py?")
        return

    print("--> Cargando modelo Random Forest...")
    clf = joblib.load(path_model)
    le = joblib.load(path_enc)
    try:
        cfg = joblib.load(path_cfg)
    except:
        cfg = {"size": 64, "hist_bins": 16} # Fallback por si acaso

    # Cargar nombres de marcas si existe el archivo
    label_names = None
    if os.path.exists(path_names):
        with open(path_names, "r", encoding='utf-8') as f:
            label_names = json.load(f).get("idx_to_label", {})

    # Si no existe label_names.json, usar el método antiguo
    if not label_names:
        make_map = load_make_map(args.compcars)

    n_classes = len(le.classes_)
    print(f"--> Cargando Random Forest (Clases={n_classes})...")

    # 2. Cargar imagen
    img = cv2.imread(args.image)
    if img is None:
        print("Error: No se pudo leer la imagen.")
        return

    # 3. Extraer features
    x = extract_features(img, size=cfg["size"], hist_bins=cfg["hist_bins"])
    # Convertir a matriz 1 fila (sklearn espera batch)
    x = x.reshape(1, -1)

    # 4. Predecir
    pred_idx = clf.predict(x)[0]

    # Predecir probabilidades (confianza)
    probs = clf.predict_proba(x)[0]

    # 5. Mostrar resultados en formato consistente
    print("\nPREDICCIONES RANDOM FOREST")
    print("="*40)

    # Top 5 predicciones
    top5_idx = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(top5_idx, 1):
        # Usar label_names si está disponible, sino usar make_map
        if label_names:
            name = label_names.get(str(idx), str(idx))
        else:
            L = le.inverse_transform([idx])[0]
            name = make_map.get(L, "Desconocido")

        print(f" {rank}. Desconocido (ID: {name}) -> {probs[idx]:.2%}")

    print("="*40)

if __name__ == "__main__":
    main()
