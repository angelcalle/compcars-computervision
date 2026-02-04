#!/usr/bin/env python3
# 05_train_vit.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 5: VISION TRANSFORMERS (ViT)
# ==============================================================================
#
# OBJETIVO:
#   Usar la arquitectura más moderna (Transformers), que originalmente se inventó
#   para texto (NLP) pero que ahora reina en visión por computador.
#
# CONCEPTO CLAVE: PATCHES & ATTENTION
#   - En vez de mirar píxeles vecinos (CNN), dividimos la imagen en "cuadraditos" (chunks/patches).
#   - La red aprende a prestar "atención" (Attention Mechanism) a qué parches
#     son importantes (ej: el logo, o la forma del faro) sin importar dónde estén.
#
# USO:
#   python 05_fit_vit.py --compcars ./CompCars --splits-dir ./01_splits --epochs 5 --outdir 05_models_vit

import os
import json
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image

# PyTorch & Torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights # El modelo ViT Base 16x16

from sklearn.metrics import accuracy_score, classification_report

# Importar función para obtener nombres de marcas
def load_make_map_from_mat(compcars_root):
    """
    Carga nombres reales de las marcas desde el archivo .mat original de CompCars.
    Requiere scipy.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        return {}

    path_mat = os.path.join(compcars_root, "misc", "make_model_name.mat")
    if not os.path.isfile(path_mat):
        return {}

    try:
        data = loadmat(path_mat)
        if "make_names" in data:
            names = data["make_names"].squeeze()
            return {str(i+1): str(n[0]) if n.size > 0 else f"Make-{i+1}" for i, n in enumerate(names)}
    except Exception:
        pass

    return {}

# ==============================================================================
# FUNCIONES DE CARGA Y PREPROCESADO
# ==============================================================================

def read_list(path):
    items = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip(): items.append(line.strip())
    return items

def make_id_from_rel(rel):
    return rel.replace("\\", "/").split("/")[0]

def filter_by_limits(rel_paths, max_makes=0, per_make=0, seed=42, top_makes=True):
    """
    Filtro idéntico al usado en pasos anteriores para garantizar comparaciones justas.
    """
    rng = np.random.default_rng(seed)
    by_make = defaultdict(list)
    for r in rel_paths:
        by_make[make_id_from_rel(r)].append(r)

    makes = list(by_make.keys())

    if max_makes and max_makes > 0:
        if top_makes:
            makes = sorted(makes, key=lambda m: len(by_make[m]), reverse=True)[:max_makes]
        else:
            makes = sorted(makes)[:max_makes]
    else:
        makes = sorted(makes)

    out = []
    for m in makes:
        items = by_make[m]
        rng.shuffle(items)
        if per_make and per_make > 0:
            items = items[:per_make]
        out.extend(items)

    rng.shuffle(out)
    return out

class CompCarsDataset(Dataset):
    def __init__(self, compcars_root, rel_paths, label_to_idx, transform=None):
        self.image_root = os.path.join(compcars_root, "image")
        self.rel_paths = rel_paths
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, i):
        rel = self.rel_paths[i].replace("\\", "/")
        mk = make_id_from_rel(rel)
        label = self.label_to_idx[mk]

        img_path = os.path.join(self.image_root, *rel.split("/"))
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)
        return img, label

def evaluate(model, loader, device):
    """Evalúa el modelo en un dataset completo y devuelve etiquetas reales vs predichas."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.cpu().numpy())
            ps.append(pred.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ==============================================================================
# FUNCIONES DE ALTO NIVEL (Lógica del Pipeline)
# ==============================================================================

def load_and_prepare_data(splits_dir, max_makes=0, per_make=0, seed=42):
    """Carga listas, filtra y genera mapeos de etiquetas"""
    train_txt = os.path.join(splits_dir, "train_all.txt")
    test_txt = os.path.join(splits_dir, "test_all.txt")

    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"No se encuentra {train_txt}")

    train_rels = read_list(train_txt)
    test_rels = read_list(test_txt)

    print("--> Filtrando datos...")
    train_rels = filter_by_limits(train_rels, max_makes, per_make, seed=seed)
    test_rels = filter_by_limits(test_rels, max_makes, per_make, seed=seed + 1)

    classes = sorted(list(set(make_id_from_rel(r) for r in train_rels)), key=lambda x: int(x) if x.isdigit() else x)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    test_rels = [r for r in test_rels if make_id_from_rel(r) in label_to_idx]

    print(f"Train: {len(train_rels)} | Test: {len(test_rels)} | Clases: {len(classes)}")

    return train_rels, test_rels, label_to_idx, idx_to_label, classes

def create_dataloaders(compcars_dir, train_rels, test_rels, label_to_idx, img_size=224, batch_size=16):
    """Crea los DataLoaders de PyTorch con las transformaciones para ViT"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Dispositivo: {device}")

    # Transformaciones (Preprocesado específico de ViT)
    weights = ViT_B_16_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2), # Jitter ayuda a que no memorice colores exactos
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = CompCarsDataset(compcars_dir, train_rels, label_to_idx, transform=train_tf)
    test_ds = CompCarsDataset(compcars_dir, test_rels, label_to_idx, transform=test_tf)

    # Configurar pin_memory según disponibilidad de CUDA para evitar warnings
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    return train_loader, test_loader, device

def build_model(num_classes, freeze_backbone=False, unfreeze_last_n_enc=0, lr=2e-4, device="cpu"):
    """Descarga y configura el modelo ViT"""
    print("--> Cargando ViT-B/16 (Preentrenado)...")
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Adaptar cabeza (Classifier Head)
    # En ViT, la 'cabeza' está en model.heads.head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    # Congelado selectivo (Fine-Tuning tactics)
    if freeze_backbone:
        print("--> Modo Congelado Activado")
        # Congelamos TODO primero
        for p in model.parameters():
            p.requires_grad = False

        # Descongelamos solo la cabeza nueva
        for p in model.heads.head.parameters():
            p.requires_grad = True

        # Opcional: Descongelar capas finales del encoder (Hybrid approach)
        if unfreeze_last_n_enc > 0:
            print(f"    Descongelando últimas {unfreeze_last_n_enc} capas del encoder.")
            # model.encoder.layers es una lista de bloques transformer
            blocks = list(model.encoder.layers)
            for blk in blocks[-unfreeze_last_n_enc:]:
                for p in blk.parameters():
                    p.requires_grad = True

    model = model.to(device)

    # Optimizador y Loss
    # Weight Decay ayuda a evitar overfitting (regularización L2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

def train_loop(model, train_loader, test_loader, optimizer, criterion, epochs, device, outdir):
    """Ejecuta el bucle de entrenamiento y guarda el mejor modelo"""
    os.makedirs(outdir, exist_ok=True)
    best_acc = 0.0

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, desc=""): return x

    print(f"\nEntrenando ViT por {epochs} épocas...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"loss": f"{total_loss/n_samples:.4f}"})

        avg_loss = total_loss / max(1, n_samples)

        # Eval
        y_true, y_pred = evaluate(model, test_loader, device)
        acc = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2%}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(outdir, "vit_best.pt"))

    return best_acc

def save_artifacts(outdir, model, best_acc, epochs, idx_to_label, compcars_dir, device):
    """Guarda modelo final y metadatos"""
    # Guardar métricas y modelo final
    torch.save(model.state_dict(), os.path.join(outdir, "vit_model.pt"))

    # Guardar mapeo de etiquetas con nombres legibles
    make_names = load_make_map_from_mat(compcars_dir)
    idx_to_label_names = {str(i): make_names.get(c, c) for i, c in idx_to_label.items()}

    with open(os.path.join(outdir, "label_map.json"), "w", encoding='utf-8') as f:
        json.dump({"idx_to_label": idx_to_label_names}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({
            "model": "vit_b_16",
            "best_test_accuracy": best_acc,
            "device": device,
            "epochs": epochs
        }, f, indent=2)

    print(f"\nGuardando resultados en {os.path.abspath(outdir)} ...")
    print(f"✅ Modelo guardado: vit_model.pt")
    print(f"✅ {len(idx_to_label)} clases detectadas")
    print(f"✅ Accuracy final (test): {best_acc:.2%}")


# ==============================================================================
# MAIN (Pipeline simplificado)
# ==============================================================================

def main():
    ap = argparse.ArgumentParser("Entrenamiento Vision Transformer (ViT)")
    ap.add_argument("--compcars", required=True, help="Ruta de la carpeta raíz de CompCars")
    ap.add_argument("--splits-dir", default="./01_splits", help="Directorio con listas train/test")
    ap.add_argument("--outdir", default="05_models_vit", help="Carpeta de salida para el modelo")

    # Params de ViT
    # ViT es muy pesado, usa batch-size bajo si tienes poca VRAM (ej. 16 o 8)
    ap.add_argument("--batch-size", type=int, default=16, help="Tamaño del batch (reducir si falta VRAM)")
    ap.add_argument("--img-size", type=int, default=224, help="ViT requiere 224x224")
    ap.add_argument("--epochs", type=int, default=3, help="Número de épocas de entrenamiento")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate bajo suele ir mejor")

    # Transfer Learning
    ap.add_argument("--freeze-backbone", action="store_true", help="Congelar encoder")
    # A veces es util descongelar las últimas capas del encoder para que se adapte mejor
    ap.add_argument("--unfreeze-last-n-enc", type=int, default=0, help="Descongelar últimos N bloques")

    # Demo
    ap.add_argument("--max-makes", type=int, default=0, help="Limitar número de marcas (0=todas)")
    ap.add_argument("--per-make", type=int, default=0, help="Limitar imágenes por marca (0=todas)")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")

    args = ap.parse_args()

    try:
        # 1. Cargar Datos
        train_rels, test_rels, label_to_idx, idx_to_label, classes = load_and_prepare_data(
            args.splits_dir, args.max_makes, args.per_make, args.seed
        )

        # 2. Crear DataLoaders
        train_loader, test_loader, device = create_dataloaders(
            args.compcars, train_rels, test_rels, label_to_idx, args.img_size, args.batch_size
        )

        # 3. Construir Modelo
        model, optimizer, criterion = build_model(
            len(classes), args.freeze_backbone, args.unfreeze_last_n_enc, args.lr, device
        )

        # 4. Entrenar
        best_acc = train_loop(
            model, train_loader, test_loader, optimizer, criterion,
            args.epochs, device, args.outdir
        )

        # 5. Guardar Resultados
        save_artifacts(args.outdir, model, best_acc, args.epochs, idx_to_label, args.compcars, device)

        print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE.")

    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()
