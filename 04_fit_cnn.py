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

        # Ojo: PyTorch espera PIL (img es objeto PIL) Images para las transformaciones
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
    # -------------------------------------------------------------------------
    # TODO: Crear DataLoaders (Pipeline de Datos)
    #
    # TAREA: Configurar cómo se cargan y transforman las imágenes.
    # 1. Detecta si hay GPU ("cuda") o usa "cpu".
    # 2. Define las transformaciones (transforms.Compose):
    #    - Resize a 224x224 (estándar para MobileNet/ResNet).
    #    - Aumentación de datos en Train (Flip, Crop...).
    #    - Convertir a Tensor y Normalizar (con medias/std de ImageNet).
    # 3. Instancia el dataset CompCarsDataset para train y test.
    # 4. Crea los DataLoader:
    #    - Train: con shuffle=True (barajar siempre).
    #    - Test: con shuffle=False.
    #
    # RETORNO ESPERADO:
    #   return train_loader, test_loader, device
    # -------------------------------------------------------------------------

def build_model(num_classes, freeze_backbone=False, lr=3e-4, device="cpu"):
    """Descarga y configura el modelo MobileNetV3"""
    # -------------------------------------------------------------------------
    # TODO: Construir el Modelo (Transfer Learning)
    #
    # TAREA: Cargar una red pre-entrenada y adaptarla a nuestras clases.
    # 1. Carga MobileNetV3 (o ResNet) con pesos "DEFAULT" (ImageNet).
    # 2. Modifica la última capa ("head" o "classifier") para tener
    #    tantas neuronas de salida como clases tengamos (num_classes).
    # 3. (Opcional) Si freeze_backbone es True, congela los pesos del extractor
    #    de características para entrenar solo la nueva capa final.
    # 4. Mueve el modelo a la GPU (device).
    # 5. Configura el Optimizador (AdamW) y la Función de Pérdida (CrossEntropy).
    #
    # RETORNO ESPERADO:
    #   return model, optimizer, criterion
    # -------------------------------------------------------------------------


def train_loop(model, train_loader, test_loader, optimizer, criterion, epochs, device, outdir):
    """Ejecuta el bucle de entrenamiento y guarda el mejor modelo"""
    # -------------------------------------------------------------------------
    # TODO: Bucle de Entrenamiento (Training Loop)
    #
    # TAREA: Iterar sobre las épocas y los batches para entrenar la red.
    #
    # BUCLE PRINCIPAL (por cada época):
    #   1. Pon el modelo en modo train: model.train().
    #   2. Itera sobre el train_loader:
    #      - Mueve datos a GPU: xb.to(device), yb.to(device).
    #      - Limpia gradientes: optimizer.zero_grad().
    #      - Predice: outputs = model(xb).
    #      - Calcula error: loss = criterion(outputs, yb).
    #      - Backpropagation: loss.backward().
    #      - Actualiza pesos: optimizer.step().
    #   3. Al final de la época, evalúa en Test (model.eval).
    #   4. Si la accuracy mejora, guarda el modelo como "mejor del momento".
    #
    # RETORNO ESPERADO:
    #   return best_acc
    # -------------------------------------------------------------------------

def save_artifacts(outdir, model, best_acc, epochs, idx_to_label, compcars_dir):
    """Guarda modelo final, metadatos y mapeo de etiquetas"""
    final_path = os.path.join(outdir, "cnn_last.pt")
    torch.save(model.state_dict(), final_path)

    # Guardar mapeo de etiquetas (imprescindible para predecir luego)
    # Convertir IDs de marca a nombres legibles
    make_names = load_make_map_from_mat(compcars_dir)
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
    # -------------------------------------------------------------------------
    # TODO: Configurar Argumentos de Línea de Comandos
    #
    # TAREA: Definir parámetros para ejecutar el script desde terminal.
    # 1. Rutas:
    #    - --compcars (dataset raíz)
    #    - --splits-dir (listas train/test)
    #    - --outdir (dónde guardar el modelo, default "04_models_cnn")
    #
    # 2. Hiperparámetros de Deep Learning:
    #    - --epochs (cuántas pasadas, ej: 5)
    #    - --batch-size (imágenes por bloque, ej: 32)
    #    - --lr (velocidad de aprendizaje, ej: 3e-4)
    #
    # 3. Opciones Avanzadas:
    #    - --freeze-backbone (para no dañar los pesos preentrenados al inicio)
    #    - Filtros de demo (--max-makes, --per-make)
    # -------------------------------------------------------------------------
    # ap = ...
    # args = ap.parse_args()

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
