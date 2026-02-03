#!/usr/bin/env python3
# 08_predict_local.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# UTILIDAD: PREDECIR MIS PROPIAS FOTOS - TODOS LOS MODELOS
# ==============================================================================
#
# OBJETIVO:
#   Escanear la carpeta "images/" y predecir qué coche es usando cualquier
#   modelo entrenado (Random Forest, CNN o ViT).
#
# USO:
#   python 08_predict_local.py --modeldir 04_models_cnn_demo
#   python 08_predict_local.py --modeldir 03_models_forest_demo
#   python 08_predict_local.py --modeldir 05_models_vit_demo

import os
import argparse
import json
import torch
import joblib
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, vit_b_16

# ==============================================================================
# FUNCIONES DE CARGA DE MODELOS
# ==============================================================================

def detect_model_type(model_dir):
    """Auto-detecta el tipo de modelo basándose en los archivos presentes"""
    if os.path.exists(os.path.join(model_dir, "forest_model.joblib")):
        return "forest"
    elif os.path.exists(os.path.join(model_dir, "cnn_best.pt")):
        return "cnn"
    elif os.path.exists(os.path.join(model_dir, "vit_best.pt")):
        return "vit"
    else:
        return None

def load_forest_model(model_dir):
    """Carga modelo Random Forest"""
    model_path = os.path.join(model_dir, "forest_model.joblib")
    enc_path = os.path.join(model_dir, "label_encoder.joblib")
    config_path = os.path.join(model_dir, "feature_config.joblib")
    label_names_path = os.path.join(model_dir, "label_names.json")

    model = joblib.load(model_path)
    le = joblib.load(enc_path) if os.path.exists(enc_path) else None
    config = joblib.load(config_path) if os.path.exists(config_path) else {"size": 64, "hist_bins": 16}

    label_names = {}
    if os.path.exists(label_names_path):
        with open(label_names_path, "r") as f:
            label_names = json.load(f)

    return model, le, config, label_names

def load_cnn_model(model_dir):
    """Carga modelo CNN/MobileNetV3"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(model_dir, "cnn_best.pt")
    map_path = os.path.join(model_dir, "label_map.json")

    with open(map_path, "r") as f:
        data = json.load(f)
        idx_to_label = data["idx_to_label"]

    num_classes = len(idx_to_label)
    model = mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model, idx_to_label, device

def load_vit_model(model_dir):
    """Carga modelo Vision Transformer"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(model_dir, "vit_best.pt")
    map_path = os.path.join(model_dir, "label_map.json")

    with open(map_path, "r") as f:
        data = json.load(f)
        idx_to_label = data["idx_to_label"]

    num_classes = len(idx_to_label)
    model = vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model, idx_to_label, device

# ==============================================================================
# FUNCIONES DE PREDICCIÓN
# ==============================================================================

def extract_rf_features(img_path, size=64, bins=16):
    """Extrae características para Random Forest"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (size, size))
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [bins], [0, 256]).flatten()
        feats.append(hist)
    x = np.concatenate(feats).astype(np.float32)
    x /= (x.sum() + 1e-6)
    return x.reshape(1, -1)

def predict_forest(model, le, config, label_names, img_path):
    """Predice con Random Forest"""
    feats = extract_rf_features(img_path, size=config["size"], bins=config["hist_bins"])
    if feats is None:
        return None, None

    pred_proba = model.predict_proba(feats)[0]
    best_idx = np.argmax(pred_proba)
    conf = pred_proba[best_idx]

    # Obtener nombre legible
    if le:
        pred_id = le.inverse_transform([best_idx])[0]
        pred_name = label_names.get(str(pred_id), str(pred_id))
    else:
        pred_name = str(best_idx)

    return pred_name, conf

def predict_torch(model, idx_to_label, device, img_path):
    """Predice con CNN o ViT"""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]

    best_idx = torch.argmax(probs).item()
    conf = probs[best_idx].item()
    label = idx_to_label[str(best_idx)]

    return label, conf

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser("Predicción Masiva - Todos los Modelos")
    ap.add_argument("--images-dir", default="images", help="Carpeta con tus fotos")
    ap.add_argument("--modeldir", required=True, help="Carpeta del modelo entrenado")
    args = ap.parse_args()

    # Verificar carpeta de imágenes
    if not os.path.isdir(args.images_dir):
        print(f"Creando carpeta {args.images_dir}...")
        os.makedirs(args.images_dir, exist_ok=True)
        print(f"La carpeta estaba vacía. Pon tus imágenes en '{args.images_dir}' y vuelve a ejecutar.")
        return

    # Buscar imágenes
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp")
    files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(valid_exts)]

    if not files:
        print(f"No hay imágenes en '{args.images_dir}'. Añade alguna foto de un coche.")
        return

    # Detectar tipo de modelo
    model_type = detect_model_type(args.modeldir)
    if model_type is None:
        print(f"Error: No se pudo detectar el tipo de modelo en {args.modeldir}")
        print("Asegúrate de que la carpeta contiene un modelo entrenado válido.")
        return

    # Cargar modelo según tipo
    print(f"--> Detectado modelo: {model_type.upper()}")
    print(f"--> Cargando desde {args.modeldir} ...")

    if model_type == "forest":
        model, le, config, label_names = load_forest_model(args.modeldir)
        model_name = "Random Forest"
    elif model_type == "cnn":
        model, idx_to_label, device = load_cnn_model(args.modeldir)
        model_name = "MobileNetV3 (CNN)"
    elif model_type == "vit":
        model, idx_to_label, device = load_vit_model(args.modeldir)
        model_name = "Vision Transformer"

    # Realizar predicciones
    print(f"\n{'='*80}")
    print(f"Analizando {len(files)} imágenes con {model_name}")
    print(f"{'='*80}\n")
    print(f"{'ARCHIVO':<35} | {'MARCA PREDICHA':<30} | {'CONFIANZA'}")
    print("-" * 80)

    for fname in files:
        fpath = os.path.join(args.images_dir, fname)
        try:
            if model_type == "forest":
                pred_name, conf = predict_forest(model, le, config, label_names, fpath)
            else:  # cnn o vit
                pred_name, conf = predict_torch(model, idx_to_label, device, fpath)

            if pred_name is None:
                print(f"{fname:<35} | Error al leer imagen")
            else:
                print(f"{fname:<35} | {pred_name:<30} | {conf:.1%}")

        except Exception as e:
            print(f"{fname:<35} | Error: {str(e)[:20]}")

    print("-" * 80)
    print(f"\n✅ Predicción completada con {model_name}")

if __name__ == "__main__":
    main()
