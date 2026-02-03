#!/usr/bin/env python3
# 02_eda_main.py (FINAL & OPTIMIZADO - ESTÉTICA ORANGE)
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA) UNIFICADO
# ==============================================================================
#
# OBJETIVO:
#   Centralizar TODO el análisis de datos, desde lo básico hasta lo avanzado.
#   - Módulo 1 (Stats): Números rápidos.
#   - Módulo 2 (Visual): Mosaicos de imágenes (Grids).
#   - Módulo 3 (Plots Clásicos): Histogramas simples.
#   - Módulo 4 (Avanzado):
#       * Años (Corregido)
#       * Dimensiones & Aspect Ratio
#       * Color (RGB) -> AHORA EN 3D
#       * Calidad de Imagen (Brillo, Contraste, Nitidez)
#
# USO:
#   python 02_eda_main.py --compcars ./CompCars --splits-dir ./01_splits --mode full

import os
import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageStat
import cv2

# ===== CONFIGURACIÓN DE ESTILOS =====
# Paleta de colores del proyecto
COLORS = {
    'forest': '#E67E22',      # Naranja para Random Forest
    'cnn': '#3498DB',         # Azul para CNN
    'transformer': '#9B59B6', # Morado para Transformers
}
PROJECT_PALETTE = ['#3498DB', '#E67E22', '#9B59B6', '#2ECC71', '#E74C3C']

def setup_plot_style():
    """Configura el estilo global de matplotlib"""
    import matplotlib as mpl
    plt.style.use('seaborn-v0_8-darkgrid')
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = '#F8F9FA'
    mpl.rcParams['axes.edgecolor'] = '#BDC3C7'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['grid.color'] = '#FFFFFF'
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['grid.alpha'] = 0.8
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['text.color'] = '#2C3E50'
# ===== FIN CONFIGURACIÓN =====

# Intentar cargar librerías avanzadas
try:
    import pandas as pd
    import seaborn as sns
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("Aviso: Faltan pandas/seaborn. El análisis avanzado será limitado.")

def read_list(path):
    items = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip(): items.append(line.strip().replace("\\", "/"))
    return items

def make_id(rel):
    return rel.split("/")[0]

def load_make_map(compcars_root):
    if not compcars_root: return {}
    mat_path = os.path.join(compcars_root, "misc", "make_model_name.mat")
    if not os.path.isfile(mat_path): return {}

    try:
        from scipy.io import loadmat
        data = loadmat(mat_path)
        if "make_names" in data:
            names = data["make_names"].squeeze()
            return {str(i): (str(n[0]) if n.size>0 else str(n)) for i,n in enumerate(names, 1)}
    except:
        pass
    return {}

def pretty_make(mid, make_map):
    return f"{mid} ({make_map[mid]})" if mid in make_map else mid

# ==============================================================================
# MÓDULO 1: ESTADÍSTICAS BÁSICAS
# ==============================================================================
def run_stats(train_rels, test_rels, make_map, outdir):
    print("\n=== MODO: ESTADÍSTICAS BÁSICAS ===")
    c_train = Counter([make_id(r) for r in train_rels])

    print(f"Imágenes Train: {len(train_rels)}")
    print(f"Imágenes Test : {len(test_rels)}")
    print(f"Marcas Train  : {len(c_train)}")

    counts = list(c_train.values())
    if counts:
        p = np.percentile(counts, [0, 25, 50, 75, 100])
        print("\nDistribución de imágenes por marca (Train):")
        print(f"  Min: {int(p[0])} | Mediana: {int(p[2])} | Max: {int(p[4])}")

    print("\nTop 5 Marcas más frecuentes:")
    for mk, ct in c_train.most_common(5):
        print(f"  - {pretty_make(mk, make_map)}: {ct} imgs")

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "stats_counts.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["make_id", "count_train"])
        for mk, ct in c_train.most_common():
            w.writerow([mk, ct])
    print(f"--> CSV guardado en {os.path.abspath(csv_path)}")

# ==============================================================================
# MÓDULO 2: VISUALIZACIÓN (GRIDS)
# ==============================================================================
def run_visualize(compcars_root, train_rels, make_map, outdir, topn=5, seed=42):
    print("\n=== MODO: VISUALIZACIÓN (GRIDS) ===")

    c = Counter([make_id(r) for r in train_rels])
    top_makes = [m for m, _ in c.most_common(topn)]

    by_make = defaultdict(list)
    for r in train_rels:
        by_make[make_id(r)].append(r)

    rng = np.random.default_rng(seed)
    rows, cols = 4, 4
    k = rows * cols

    os.makedirs(outdir, exist_ok=True)

    print(f"Generando {len(top_makes)} grids de mosaicos...")
    for mk in tqdm(top_makes, desc="Creando Mosaicos"):
        files = by_make[mk]
        if not files: continue

        selected = files
        if len(files) > k:
            selected = rng.choice(files, size=k, replace=False)

        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        fig.suptitle(f"Ejemplos: {pretty_make(mk, make_map)}", fontsize=16)

        for i, ax in enumerate(axes.flatten()):
            ax.axis("off")
            if i < len(selected):
                path = os.path.join(compcars_root, "image", *selected[i].split("/"))
                try:
                    img = Image.open(path).convert("RGB")
                    ax.imshow(img)
                except:
                    ax.text(0.5, 0.5, "Error", ha="center")
            else:
                ax.text(0.5, 0.5, "-", ha="center")

        out_path = os.path.join(outdir, f"grid_{mk}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    print(f"--> Mosaicos guardados en {os.path.abspath(outdir)}")

# ==============================================================================
# MÓDULO 3: GRÁFICOS CLÁSICOS
# ==============================================================================
def run_plots_basic(train_rels, make_map, outdir):
    print("\n=== MODO: GRÁFICOS CLÁSICOS ===")

    setup_plot_style()  # Aplicar estilo consistente

    c = Counter([make_id(r) for r in train_rels])
    counts = sorted(c.values(), reverse=True)

    os.makedirs(outdir, exist_ok=True)

    # 1. Histograma Simple (usando color project forest)
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=30, color=COLORS['forest'], edgecolor='white', alpha=0.85)
    plt.title("Distribución: Cantidad de imágenes por Marca", fontsize=16, fontweight='bold')
    plt.xlabel("Número de fotos")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_distribution_hist_basic.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"--> Histograma básico guardado en {os.path.abspath(outdir)}")

# ==============================================================================
# MÓDULO 4: ANÁLISIS AVANZADO & CALIDAD DE IMAGEN
# ==============================================================================

def extract_year(rel):
    parts = rel.split("/")
    if len(parts) >= 2:
        candidate = parts[-2]
        if candidate.isdigit() and len(candidate) == 4:
            val = int(candidate)
            if 1900 <= val <= 2025:
                return val
    return None

def get_image_metrics(compcars_root, rel_path):
    full_path = os.path.join(compcars_root, "image", *rel_path.split("/"))
    try:
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            stat = ImageStat.Stat(img)
            r, g, b = stat.mean

            brightness = (0.299*r + 0.587*g + 0.114*b)

            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            contrast = gray.std()
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            return w, h, r, g, b, brightness, contrast, blur_score
    except:
        return None, None, None, None, None, None, None, None

def plot_3d_color_space(df, outdir, sample_size=None):
    """
    Genera un scatter plot 3D de los valores RGB.
    Si sample_size es None, usa todos los puntos disponibles.
    """
    # Si no se especifica sample_size, usar todos los datos
    if sample_size is None:
        df_sample = df
    elif len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalizar colores para visualización (0..1)
    colors = df_sample[['mean_r', 'mean_g', 'mean_b']].values / 255.0

    # Ajustar tamaño y alpha según cantidad de puntos
    if len(df_sample) > 5000:
        point_size = 10
        alpha = 0.3
    elif len(df_sample) > 2000:
        point_size = 20
        alpha = 0.5
    else:
        point_size = 30
        alpha = 0.6

    ax.scatter(
        df_sample['mean_r'],
        df_sample['mean_g'],
        df_sample['mean_b'],
        c=colors,
        marker='o',
        alpha=alpha,
        s=point_size
    )

    ax.set_xlabel('Rojo (Red)')
    ax.set_ylabel('Verde (Green)')
    ax.set_zlabel('Azul (Blue)')
    ax.set_title(f'Espacio de Color 3D ({len(df_sample):,} imágenes)', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "analysis_color_3d.png"), dpi=120)
    plt.close()


def run_advanced_analysis(compcars_root, train_rels, outdir):
    print("\n=== MODO: ANÁLISIS AVANZADO (Metadata + Calidad) ===")

    if not HAS_LIBS:
        print("Saltando análisis avanzado por falta de librerías.")
        return

    data = []
    print("Extrayendo metadatos INTENSOS (Años, Color, Brillo, Contraste, Nitidez)...")

    for rel in tqdm(train_rels, desc="Analizando"):
        mk = make_id(rel)
        yr = extract_year(rel)
        metrics = get_image_metrics(compcars_root, rel)

        if metrics[0] is not None:
            w, h, r, g, b, brit, cont, blur = metrics
            data.append({
                "make_id": mk,
                "year": yr,
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 0,
                "mean_r": r,
                "mean_g": g,
                "mean_b": b,
                "brightness": brit,
                "contrast": cont,
                "blurriness": blur
            })

    df = pd.DataFrame(data)
    if df.empty:
        print("Error: No se obtuvieron datos.")
        return

    os.makedirs(outdir, exist_ok=True)
    setup_plot_style()  # Aplicar estilo consistente

    # 1. Años (PALETA NARANJA)
    plt.figure(figsize=(12, 6))
    if df['year'].notna().sum() > 0:
        years_df = df.dropna(subset=['year'])
        years_df = years_df.sort_values('year')

        # Usamos YlOrBr para progresión de amarillo a naranja/marrón
        sns.countplot(data=years_df, x='year', hue='year', palette="YlOrBr", legend=False)
        plt.title("Distribución Temporal (Años)", fontsize=16, fontweight='bold', color="#d35400")
        plt.xlabel("Año", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "analysis_years_dist.png"), dpi=120)
        plt.close()

    # 2. Dimensiones (ORANGES Fill)
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x="width", y="height", fill=True, cmap="Oranges", thresh=0.05)
    plt.title("Mapa de Densidad: Dimensiones", fontsize=16, fontweight='bold', color="#d35400")
    plt.xlabel("Ancho (px)")
    plt.ylabel("Alto (px)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "analysis_dimensions_kde.png"), dpi=120)
    plt.close()

    # 3. Aspect Ratio (Violeta -> Naranja oscuro)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="aspect_ratio", bins=30, kde=True, color="#e67e22") # Pumpkin Color
    plt.axvline(x=1.33, color='black', linestyle='--', label='4:3')
    plt.axvline(x=1.77, color='brown', linestyle='--', label='16:9')
    plt.legend()
    plt.title("Aspect Ratio (Formato)", fontsize=16, fontweight='bold', color="#d35400")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "analysis_aspect_ratio.png"), dpi=120)
    plt.close()

    # 4. Color RGB (Clásico + 3D)

    # 4a. 2D Distribution (Mantener colores reales RGB por didáctica, pero mejorar estilo)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="mean_r", color="red",  label="R", fill=True, alpha=0.1)
    sns.kdeplot(data=df, x="mean_g", color="green",label="G", fill=True, alpha=0.1)
    sns.kdeplot(data=df, x="mean_b", color="blue", label="B", fill=True, alpha=0.1)
    plt.title("Distribución de Canales RGB (Intensidad)", fontsize=16, fontweight='bold')
    plt.xlim(0, 255)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "analysis_color_dist.png"), dpi=120)
    plt.close()

    # 4b. 3D COLOR SPACE (NUEVO)
    plot_3d_color_space(df, outdir)

    # 5. Top Marcas (Oranges_r: De oscuro a claro)
    top20 = df['make_id'].value_counts().head(20).index
    df_top20 = df[df['make_id'].isin(top20)]
    plt.figure(figsize=(14, 8))
    sns.countplot(
        data=df_top20,
        x='make_id',
        order=df_top20['make_id'].value_counts().index,
        hue='make_id',
        palette="Oranges_r", # Invertido para que el top 1 sea el más fuerte
        legend=False
    )
    plt.title("Top 20 Marcas con más Imágenes", fontsize=18, fontweight='bold', color="#d35400")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "analysis_top20_makes.png"), dpi=120)
    plt.close()

    # 6. MÉTRICAS DE CALIDAD (Tonos cálidos)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Brillo (Gold)
    sns.histplot(data=df, x="brightness", ax=axes[0], bins=30, color="gold", kde=True)
    axes[0].set_title("Brillo", color="#d35400")
    axes[0].set_xlim(0, 255)

    # Contraste (Chocolate)
    sns.histplot(data=df, x="contrast", ax=axes[1], bins=30, color="chocolate", kde=True)
    axes[1].set_title("Contraste", color="#d35400")

    # Blur (SaddleBrown) - El teal desentonaba
    sns.histplot(data=df, x="blurriness", ax=axes[2], bins=30, color="saddlebrown", kde=True, log_scale=True)
    axes[2].set_title("Nitidez (Log Scale)", color="#d35400")

    plt.suptitle("Análisis de Calidad de Imagen", fontsize=20, fontweight="bold", color="#d35400")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "analysis_image_quality.png"), dpi=120)
    plt.close()

    # Guardar resumen
    stats_path = os.path.join(outdir, "dataset_summary.csv")
    df.describe().to_csv(stats_path)
    print(f"--> Gráficos avanzados (Naranjas + 3D) y resumen guardados en {os.path.abspath(outdir)}")


def main():
    ap = argparse.ArgumentParser("EDA Unificado (Básico + Avanzado)")
    ap.add_argument("--compcars", required=True, help="Ruta de la carpeta raíz de CompCars")
    ap.add_argument("--splits-dir", default="./01_splits", help="Directorio con listas train/test (default: ./01_splits)")
    ap.add_argument("--outdir", default="02_eda_out", help="Carpeta de salida para gràficos (default: 02_eda_out)")
    ap.add_argument("--mode", default="full", choices=["full", "basic", "advanced"], help="Modo de ejecución: basic, advanced o full")
    ap.add_argument("--topn", type=int, default=5, help="Nº de mosaicos a generar")

    args = ap.parse_args()

    train_path = os.path.join(args.splits_dir, "train_all.txt")
    test_path = os.path.join(args.splits_dir, "test_all.txt")

    if not os.path.exists(train_path):
        print("Error: No encuentro train_all.txt")
        return

    train_rels = read_list(train_path)
    test_rels = read_list(test_path)
    make_map = load_make_map(args.compcars)

    # Lógica de ejecución
    run_all = (args.mode == "full")

    if run_all or args.mode == "basic":
        # 1. Stats
        run_stats(train_rels, test_rels, make_map, args.outdir)
        # 2. Basic Plots
        run_plots_basic(train_rels, make_map, args.outdir)
        # 3. Mosaicos
        run_visualize(args.compcars, train_rels, make_map, os.path.join(args.outdir, "grids"), topn=args.topn)

    if run_all or args.mode == "advanced":
        # 4. Advanced
        run_advanced_analysis(args.compcars, train_rels, args.outdir)

if __name__ == "__main__":
    main()
