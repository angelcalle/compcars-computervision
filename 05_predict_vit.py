#!/usr/bin/env python3
# 05_predict_vit.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 5b: INFERENCIA CON TRANSFORMERS
# ==============================================================================
#
# OBJETIVO:
#   Usar el ViT entrenado para predecir.
#
# USO:
#   python 05_predict_vit.py --image ./X.jpg --modeldir ./models_vit_compcars --compcars ./CompCars

import os
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

def load_make_map(compcars_root):
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
    ap = argparse.ArgumentParser("Predecir con ViT")
    ap.add_argument("--image", required=True, help="Ruta de la imagen a clasificar")
    ap.add_argument("--modeldir", required=True, help="Carpeta con el modelo ViT entrenado")
    ap.add_argument("--compcars", default="./CompCars", help="Ruta raíz para cargar nombres de marcas (opcional)")
    ap.add_argument("--topk", type=int, default=5, help="Mostrar top K predicciones")
    args = ap.parse_args()

    # 1. Cargar metadatos
    path_map = os.path.join(args.modeldir, "label_map.json")
    path_model = os.path.join(args.modeldir, "vit_best.pt")
    if not os.path.exists(path_model): # Fallback
        path_model = os.path.join(args.modeldir, "vit_model.pt")

    if not os.path.exists(path_map) or not os.path.exists(path_model):
        print("Error: No encuentro el modelo o el mapa de etiquetas.")
        return

    with open(path_map, "r") as f:
        d = json.load(f)
        idx_to_label = {int(k): v for k,v in d["idx_to_label"].items()}

    num_classes = len(idx_to_label)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Reconstruir Arquitectura ViT
    print(f"--> Cargando ViT-B/16 (Clases={num_classes})...")
    model = vit_b_16(weights=None)
    in_feat = model.heads.head.in_features
    model.heads.head = nn.Linear(in_feat, num_classes)

    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device)
    model.eval()

    # 3. Preprocesar Imagen
    weights = ViT_B_16_Weights.DEFAULT
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(weights.transforms().mean, weights.transforms().std),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    # 4. Inferencia
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # 5. Resultados
    make_map = load_make_map(args.compcars)
    top_idx = np.argsort(probs)[::-1][:args.topk]

    print("\nPREDICCIONES ViT")
    print("="*40)
    for rank, idx in enumerate(top_idx, 1):
        mk_id = idx_to_label[idx]
        name = make_map.get(mk_id, "Desconocido")
        print(f" {rank}. {name} (ID: {mk_id}) -> {probs[idx]:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
