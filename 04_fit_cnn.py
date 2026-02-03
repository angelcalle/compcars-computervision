#!/usr/bin/env python3
# 04_train_cnn.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 4: DEEP LEARNING CON CNN (Redes Neuronales Convolucionales)
# ==============================================================================
#
# OBJETIVO:
#   Usar una red neuronal moderna (MobileNetV3) preentrenada en ImageNet
#   y ajustarla (Fine-Tuning) para distinguir nuestras marcas de coches.
#
# CONCEPTO CLAVE: TRANSFER LEARNING
#   En lugar de entrenar desde cero (lo cual requiere millones de fotos y semanas),
#   tomamos un "cerebro" que ya sabe ver (entrenado con ImageNet) y solo le
#   enseñamos los nombres nuevos de los coches.
#
# USO:
#   python 04_train_cnn.py --compcars ./CompCars --splits-dir ./splits --epochs 5

import os
import json
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from sklearn.metrics import accuracy_score, classification_report

# Importar función para obtener nombres de marcas
try:
    from generate_labels import get_make_labels
except ImportError:
    # Fallback si no existe
    def get_make_labels(d): return {}

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
    Filtra el dataset para hacerlo más pequeño si queremos una demo rápida.
    """
    rng = np.random.default_rng(seed)
    by_make = defaultdict(list)
    for r in rel_paths:
        by_make[make_id_from_rel(r)].append(r)

    makes = list(by_make.keys())

    # 1. Seleccionar marcas
    if max_makes and max_makes > 0:
        if top_makes:
            # Las que tienen más fotos primero
            makes = sorted(makes, key=lambda m: len(by_make[m]), reverse=True)[:max_makes]
        else:
            makes = sorted(makes)[:max_makes]
    else:
        makes = sorted(makes)

    out = []
    for m in makes:
        items = by_make[m]
        rng.shuffle(items)
        # 2. Seleccionar imágenes por marca
        if per_make and per_make > 0:
            items = items[:per_make]
        out.extend(items)

    rng.shuffle(out)
    return out, makes

# Definición del Dataset para PyTorch
class CompCarsDataset(Dataset):
    def __init__(self, compcars_root, rel_paths, label_to_idx, transform=None):
        self.image_root = os.path.join(compcars_root, "image")
        self.rel_paths = rel_paths
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, i):
        # 1. Identificar ruta y etiqueta
        rel = self.rel_paths[i].replace("\\", "/")
        mk = make_id_from_rel(rel)
        label = self.label_to_idx[mk]

        # 2. Cargar imagen
        img_path = os.path.join(self.image_root, *rel.split("/"))

        # Ojo: PyTorch espera PIL Images para las transformaciones
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            # Fallback a imagen negra si falla la carga (robusted)
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        # 3. Aplicar transformaciones (Data Augmentation)
        if self.transform:
            img = self.transform(img)

        return img, label

def evaluate(model, loader, device):
    """Evalúa el modelo en un dataset completo y devuelve etiquetas reales vs predichas."""
    model.eval() # Modo evaluación (apaga Dropout, BatchNorm fijo)
    ys, ps = [], []
    with torch.no_grad(): # No calcular gradientes (ahorra memoria)
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

    print("--> Aplicando filtros y cargando listas...")
    train_rels, _ = filter_by_limits(train_rels, max_makes, per_make, seed=seed)
    # Importante: Test debe tener las mismas clases que train (o menos), no más.
    # Aquí simplificamos filtrando test con las mismas reglas
    test_rels, _ = filter_by_limits(test_rels, max_makes, per_make, seed=seed+1)

    # Mapa de etiquetas (Clases)
    # Extraemos todas las marcas únicas del TRAIN
    classes_train = sorted(list(set(make_id_from_rel(r) for r in train_rels)), key=lambda x: int(x) if x.isdigit() else x)
    label_to_idx = {c: i for i, c in enumerate(classes_train)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    # Filtrar Test para quitar clases que no estén en Train (por si el split aleatorio dejó fuera alguna marca rara)
    test_rels = [r for r in test_rels if make_id_from_rel(r) in label_to_idx]

    print(f"Datos finales -> Train: {len(train_rels)} | Test: {len(test_rels)} | Clases: {len(classes_train)}")

    return train_rels, test_rels, label_to_idx, idx_to_label, classes_train

def create_dataloaders(compcars_dir, train_rels, test_rels, label_to_idx, batch_size=32):
    """Crea los DataLoaders de PyTorch con las transformaciones adecuadas"""
    # Selección automática de GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Usando dispositivo: {device.upper()}")

    # MobileNetV3 espera cierta normalización (Media y Desviación estándar de ImageNet)
    weights = MobileNet_V3_Large_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Data Augmentation: Zoom aleatorio
        transforms.RandomHorizontalFlip(),                    # Data Augmentation: Espejo
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # DataLoaders
    train_ds = CompCarsDataset(compcars_dir, train_rels, label_to_idx, transform=train_tf)
    test_ds = CompCarsDataset(compcars_dir, test_rels, label_to_idx, transform=test_tf)

    # Configurar pin_memory según disponibilidad de CUDA para evitar warnings
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    return train_loader, test_loader, device

def build_model(num_classes, freeze_backbone=False, lr=3e-4, device="cpu"):
    """Descarga y configura el modelo MobileNetV3"""
    print("--> Descargando/Cargando MobileNetV3 Preentrenada...")
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)

    # Reemplazar la última capa ("cabeza") para que coincida con nuestro número de clases
    # (Originalmente ImageNet tiene 1000 clases, nosotros tenemos N)
    num_ft = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ft, num_classes)

    # Congelar backbone si se pide
    if freeze_backbone:
        print("--> CONGELANDO Backbone (Solo se entrena la capa final).")
        for param in model.features.parameters():
            param.requires_grad = False

    model = model.to(device)

    # Optimizador y Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW es el estándar moderno
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

def train_loop(model, train_loader, test_loader, optimizer, criterion, epochs, device, outdir):
    """Ejecuta el bucle de entrenamiento y guarda el mejor modelo"""
    os.makedirs(outdir, exist_ok=True)
    best_acc = 0.0

    # Import local para evitar error si no está instalada (aunque aquí ya asumimos que sí)
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, desc=""): return x

    print(f"\nIniciando entrenamiento por {epochs} épocas...")

    for epoch in range(1, epochs + 1):
        model.train() # Modo entrenamiento
        total_loss = 0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()       # Reset gradientes
            outputs = model(xb)         # Predicción
            loss = criterion(outputs, yb) # Calcular error
            loss.backward()             # Calcular gradientes (Backprop)
            optimizer.step()            # Actualizar pesos

            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"loss": f"{total_loss/n_samples:.4f}"})

        avg_loss = total_loss / max(1, n_samples)

        # Evaluación al final de la época
        y_true, y_pred = evaluate(model, test_loader, device)
        acc = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2%}")

        # Guardar el mejor modelo
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(outdir, "cnn_best.pt"))

    print(f"\nMejor Accuracy en Test: {best_acc:.2%}")
    return best_acc

def save_artifacts(outdir, model, best_acc, epochs, idx_to_label, compcars_dir):
    """Guarda modelo final, metadatos y mapeo de etiquetas"""
    final_path = os.path.join(outdir, "cnn_last.pt")
    torch.save(model.state_dict(), final_path)

    # Guardar mapeo de etiquetas (imprescindible para predecir luego)
    # Convertir IDs de marca a nombres legibles
    make_names = get_make_labels(compcars_dir)
    idx_to_label_names = {str(i): make_names.get(c, c) for i, c in idx_to_label.items()}

    with open(os.path.join(outdir, "label_map.json"), "w", encoding='utf-8') as f:
        json.dump({"idx_to_label": idx_to_label_names}, f, ensure_ascii=False, indent=2)

    # Métricas
    metrics = {
        "best_accuracy": best_acc,
        "epochs": epochs,
        "model": "mobilenet_v3_large",
        "n_classes": len(idx_to_label)
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nGuardando resultados en {os.path.abspath(outdir)} ...")
    print(f"✅ Modelo guardado: cnn_best.pt")
    print(f"✅ {len(idx_to_label)} clases detectadas")
    print(f"✅ Accuracy final (test): {best_acc:.2%}")


# ==============================================================================
# MAIN (Pipeline simplificado)
# ==============================================================================

def main():
    ap = argparse.ArgumentParser("Entrenamiento CNN MobileNetV3")
    ap.add_argument("--compcars", required=True, help="Ruta de CompCars")
    ap.add_argument("--splits-dir", default="./01_splits", help="Donde están los .txt")
    ap.add_argument("--outdir", default="04_models_cnn", help="Carpeta de salida")

    # Hiperparámetros
    ap.add_argument("--epochs", type=int, default=5, help="Vueltas completas al dataset")
    ap.add_argument("--batch-size", type=int, default=32, help="Imágenes procesadas a la vez")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning Rate: velocidad de aprendizaje")

    # Config Transfer Learning
    ap.add_argument("--freeze-backbone", action="store_true", help="Congela el cerebro principal, entrena solo la salida")

    # Datos Demo
    ap.add_argument("--max-makes", type=int, default=0, help="Limitar clases (0=todas)")
    ap.add_argument("--per-make", type=int, default=0, help="Limitar imagenes por clase")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")

    args = ap.parse_args()

    try:
        # 1. Cargar Datos
        train_rels, test_rels, label_to_idx, idx_to_label, classes_train = load_and_prepare_data(
            args.splits_dir, args.max_makes, args.per_make, args.seed
        )

        # 2. Crear DataLoader
        train_loader, test_loader, device = create_dataloaders(
            args.compcars, train_rels, test_rels, label_to_idx, args.batch_size
        )

        # 3. Construir Modelo
        model, optimizer, criterion = build_model(
            len(classes_train), args.freeze_backbone, args.lr, device
        )

        # 4. Entrenar
        best_acc = train_loop(
            model, train_loader, test_loader, optimizer, criterion,
            args.epochs, device, args.outdir
        )

        # 5. Guardar Resultados
        save_artifacts(args.outdir, model, best_acc, args.epochs, idx_to_label, args.compcars)

        print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE.")

    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()
