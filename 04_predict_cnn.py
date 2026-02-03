#!/usr/bin/env python3
# 04_predict_cnn.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 4b: INFERENCIA CON CNN
# ==============================================================================
#
# OBJETIVO:
#   Cargar el modelo MobileNet entrenado y usarlo para predecir una nueva imagen.
#
# USO:
#   python 04_predict_cnn.py --image ./CompCars/image/78/2012/x.jpg --modeldir ./models_cnn_mobilenet --compcars ./CompCars

import os
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

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
    ap = argparse.ArgumentParser("Predecir con CNN")
    ap.add_argument("--image", required=True, help="Ruta de la imagen a clasificar")
    ap.add_argument("--modeldir", required=True, help="Carpeta con el modelo entrenado y metadatos")
    ap.add_argument("--compcars", default="./CompCars", help="Ruta raíz para cargar nombres de marcas (opcional)")
    ap.add_argument("--topk", type=int, default=5, help="Mostrar top K predicciones")
    args = ap.parse_args()

    # 1. Cargar Configuración
    label_map_path = os.path.join(args.modeldir, "label_map.json")
    model_path = os.path.join(args.modeldir, "cnn_best.pt")

    # Fallback si no existe el best, probamos el last
    if not os.path.exists(model_path):
        model_path = os.path.join(args.modeldir, "cnn_last.pt")

    if not os.path.exists(label_map_path) or not os.path.exists(model_path):
        print("Error: No encuentro el modelo o el mapa de etiquetas.")
        return

    with open(label_map_path, "r") as f:
        d = json.load(f)
        idx_to_label = {int(k): v for k,v in d["idx_to_label"].items()}

    num_classes = len(idx_to_label)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Reconstruir la Arquitectura
    # Debemos crear el modelo VACÍO con la MISMA estructura que el entrenado
    print(f"--> Cargando arquitectura MobileNetV3 (Clases={num_classes})...")
    model = mobilenet_v3_large(weights=None) # No necesitamos pesos de internet, cargaremos los nuestros
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # Cargar los "conocimientos" (pesos)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Preprocesar Imagen
    # Mismas transformaciones que en 'test_tf' del entrenamiento
    weights = MobileNet_V3_Large_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device) # Añadir dimensión batch: [1, 3, 224, 224]

    # 4. Inferencia
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # 5. Resultados
    make_map = load_make_map(args.compcars)

    top_indices = np.argsort(probs)[::-1][:args.topk]

    print("\nPREDICCIONES DE LA RED NEURONAL")
    print("="*40)
    for rank, idx in enumerate(top_indices, 1):
        mk_id = idx_to_label[idx]
        mk_name = make_map.get(mk_id, "Desconocido")
        score = probs[idx]
        print(f" {rank}. {mk_name} (ID: {mk_id}) -> {score:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
