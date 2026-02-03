#!/usr/bin/env python3
# 06_analyze_metrics.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 6: ANÁLISIS COMPARATIVO COMPLETO
# ==============================================================================

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# ===== CONFIGURACIÓN DE ESTILOS =====
# Paleta de colores del proyecto
COLORS = {
    'forest': '#E67E22',
    'cnn': '#3498DB',
    'transformer': '#9B59B6',
}
DATASET_COLORS = {
    'full': '#2ECC71',
    'demo': '#E74C3C',
}

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

def save_figure(fig, filepath, dpi=150):
    """Guarda una figura con configuración consistente"""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Gráfica guardada: {filepath}")
# ===== FIN CONFIGURACIÓN =====

def find_all_model_dirs(base_dir="."):
    """Auto-detecta todas las carpetas de modelos"""
    model_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith(("03_models", "04_models", "05_models")):
            path = os.path.join(base_dir, item)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "metrics.json")):
                model_dirs.append(path)
    return sorted(model_dirs)

def load_metrics(model_dir):
    """Carga el archivo metrics.json"""
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def extract_accuracy(metrics):
    """Extrae accuracy de diferentes formatos de metrics.json"""
    if metrics is None:
        return 0.0

    # Intentar diferentes campos
    if "test_metrics" in metrics and metrics["test_metrics"] and "accuracy" in metrics["test_metrics"]:
        return metrics["test_metrics"]["accuracy"]
    elif "test_accuracy" in metrics:
        return metrics["test_accuracy"]
    elif "best_test_accuracy" in metrics:
        return metrics["best_test_accuracy"]
    elif "best_accuracy" in metrics:
        return metrics["best_accuracy"]
    elif "val_accuracy" in metrics:
        return metrics["val_accuracy"]

    return 0.0

def extract_model_info(model_dir, metrics):
    """Extrae información detallada del modelo"""
    name = os.path.basename(model_dir)

    # Determinar tipo de modelo
    if "forest" in name:
        model_type = "Random Forest"
        model_family = "Classic ML"
    elif "cnn" in name:
        model_type = "MobileNetV3"
        model_family = "CNN"
    elif "vit" in name:
        model_type = "ViT-B/16"
        model_family = "Transformer"
    else:
        model_type = "Unknown"
        model_family = "Unknown"

    # Determinar dataset
    is_demo = "demo" in name
    dataset = "Demo" if is_demo else "Full"

    # Extraer métricas
    accuracy = extract_accuracy(metrics)
    n_classes = metrics.get("n_classes", 0)
    epochs = metrics.get("epochs", "-")

    # Intentar obtener el modelo de metrics
    if "model" in metrics:
        model_name = metrics["model"]
    else:
        model_name = model_type

    return {
        "name": name,
        "type": model_type,
        "family": model_family,
        "dataset": dataset,
        "is_demo": is_demo,
        "classes": n_classes,
        "accuracy": accuracy,
        "epochs": epochs,
        "model_name": model_name
    }

def print_comparison_table(results):
    """Imprime tabla comparativa detallada"""
    WIDTH = 105
    print("\n" + "="*WIDTH)
    print("COMPARATIVA COMPLETA DE MODELOS".center(WIDTH))
    print("="*WIDTH)

    # Encabezado - ajustar anchos para que sume 105
    fmt = "{:<30} | {:<22} | {:<8} | {:>7} | {:>10} | {:>7}"
    print(fmt.format("Modelo", "Tipo", "Dataset", "Clases", "Accuracy", "Épocas"))
    print("-"*WIDTH)

    # Agrupar por dataset
    full_models = [r for r in results if not r["is_demo"]]
    demo_models = [r for r in results if r["is_demo"]]

    # Definir orden de visualización por tipo de modelo
    model_order = {'Classic ML': 1, 'CNN': 2, 'Transformer': 3}

    def sort_by_model_type(r):
        return model_order.get(r['family'], 99)

    if full_models:
        print("MODELOS COMPLETOS (163 clases)".center(WIDTH))
        print("-"*WIDTH)
        # Ordenar por tipo de modelo (Forest -> CNN -> ViT)
        for r in sorted(full_models, key=sort_by_model_type):
            epochs_str = str(r["epochs"]) if r["epochs"] != "-" else "-"
            print(fmt.format(
                r['name'],
                r['type'],
                r['dataset'],
                r['classes'],
                f"{r['accuracy']:.2%}",
                epochs_str
            ))

    if demo_models:
        print("-"*WIDTH)
        print("MODELOS DEMO (10 clases)".center(WIDTH))
        print("-"*WIDTH)
        # Ordenar por tipo de modelo (Forest -> CNN -> ViT)
        for r in sorted(demo_models, key=sort_by_model_type):
            epochs_str = str(r["epochs"]) if r["epochs"] != "-" else "-"
            print(fmt.format(
                r['name'],
                r['type'],
                r['dataset'],
                r['classes'],
                f"{r['accuracy']:.2%}",
                epochs_str
            ))

    print("="*WIDTH)

    # Resumen
    if full_models:
        best_full = max(full_models, key=lambda x: x["accuracy"])
        print(f"\n🏆 MEJOR MODELO (Full): {best_full['name']} - {best_full['type']} - {best_full['accuracy']:.2%}")

    if demo_models:
        best_demo = max(demo_models, key=lambda x: x["accuracy"])
        print(f"🏆 MEJOR MODELO (Demo): {best_demo['name']} - {best_demo['type']} - {best_demo['accuracy']:.2%}")

    print()

def create_bar_chart(results, outdir):
    """Crea gráfico de barras agrupadas por tipo de modelo"""
    setup_plot_style()  # Aplicar estilo consistente

    plt.figure(figsize=(14, 8))

    # Agrupar por familia y dataset
    families = {}
    for r in results:
        key = r["family"]
        if key not in families:
            families[key] = {"full": [], "demo": []}

        if r["is_demo"]:
            families[key]["demo"].append(r)
        else:
            families[key]["full"].append(r)

    # Preparar datos para gráfico
    family_names = sorted(families.keys())
    x = np.arange(len(family_names))
    width = 0.35

    full_accs = []
    demo_accs = []

    for fname in family_names:
        # Promedio si hay múltiples modelos del mismo tipo
        full = families[fname]["full"]
        demo = families[fname]["demo"]

        full_acc = np.mean([m["accuracy"] for m in full]) if full else 0
        demo_acc = np.mean([m["accuracy"] for m in demo]) if demo else 0

        full_accs.append(full_acc)
        demo_accs.append(demo_acc)

    # Crear barras usando colores consistentes
    bars1 = plt.bar(x - width/2, full_accs, width, label='Full Dataset (163 clases)',
                    color=DATASET_COLORS['full'], edgecolor='white', alpha=0.9, linewidth=2)
    bars2 = plt.bar(x + width/2, demo_accs, width, label='Demo Dataset (10 clases)',
                    color=DATASET_COLORS['demo'], edgecolor='white', alpha=0.9, linewidth=2)

    # Etiquetas
    plt.xlabel('Tipo de Modelo', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Comparativa de Modelos por Tipo y Dataset', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(x, family_names, fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Añadir valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.1%}', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(outdir, "comparison_by_type.png")
    save_figure(plt.gcf(), out_path)
    plt.close()

def create_scatter_plot(results, outdir):
    """Crea scatter plot de Accuracy vs Número de Clases"""
    setup_plot_style()  # Aplicar estilo consistente

    plt.figure(figsize=(12, 8))

    # Datos por familia
    families = {}
    for r in results:
        if r["family"] not in families:
            families[r["family"]] = {"full": [], "demo": []}

        if r["is_demo"]:
            families[r["family"]]["demo"].append(r)
        else:
            families[r["family"]]["full"].append(r)

    # Usar colores consistentes del proyecto
    colors_map = {
        'Classic ML': COLORS['forest'],
        'CNN': COLORS['cnn'],
        'Transformer': COLORS['transformer']
    }
    markers = {'full': 'o', 'demo': 's'}

    # Plot
    for fname, data in families.items():
        for dtype, models in data.items():
            if models:
                classes = [m["classes"] for m in models]
                accs = [m["accuracy"] for m in models]
                label = f'{fname} ({dtype.capitalize()})'
                plt.scatter(classes, accs, s=200, alpha=0.7,
                           c=colors_map.get(fname, '#95a5a6'),
                           marker=markers[dtype],
                           edgecolors='white', linewidths=2,
                           label=label)

    plt.xlabel('Número de Clases', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Accuracy vs Número de Clases por Tipo de Modelo', fontsize=18, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(0, 1.1)

    plt.tight_layout()
    out_path = os.path.join(outdir, "accuracy_vs_classes.png")
    save_figure(plt.gcf(), out_path)
    plt.close()

def create_ranking_chart(results, outdir):
    """Crea gráfico de ranking general"""
    setup_plot_style()  # Aplicar estilo consistente

    plt.figure(figsize=(14, 10))

    # Ordenar por accuracy
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    names = [r["name"] for r in sorted_results]
    accs = [r["accuracy"] for r in sorted_results]

    # Colores según tipo usando paleta consistente
    color_map = {
        'Classic ML': COLORS['forest'],
        'CNN': COLORS['cnn'],
        'Transformer': COLORS['transformer']
    }
    colors = [color_map.get(r["family"], '#95a5a6') for r in sorted_results]

    bars = plt.barh(names, accs, color=colors, edgecolor='white', alpha=0.85, height=0.7, linewidth=1.5)

    plt.xlabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Ranking General de Modelos', fontsize=18, fontweight='bold', pad=20)
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Añadir porcentajes
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        plt.text(acc + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.1%}', va='center', fontsize=11, fontweight='bold')

    # Leyenda
    legend_patches = [mpatches.Patch(color=color, label=family)
                     for family, color in color_map.items()]
    plt.legend(handles=legend_patches, loc='lower right', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(outdir, "ranking_general.png")
    save_figure(plt.gcf(), out_path)
    plt.close()

def generate_markdown_report(results, outdir):
    """Genera reporte completo en Markdown"""
    report_path = os.path.join(outdir, "model_comparison_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        # Encabezado
        f.write("# 📊 Reporte de Comparativa de Modelos\n\n")
        f.write(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total de modelos analizados:** {len(results)}\n\n")

        # Resumen Ejecutivo
        f.write("---\n\n")
        f.write("## 🏆 Resumen Ejecutivo\n\n")

        full_models = [r for r in results if not r["is_demo"]]
        demo_models = [r for r in results if r["is_demo"]]

        if full_models:
            best_full = max(full_models, key=lambda x: x["accuracy"])
            f.write(f"### Mejor Modelo (Dataset Completo - 163 clases)\n\n")
            f.write(f"- 🎯 **Modelo:** `{best_full['name']}`\n")
            f.write(f"- 🧠 **Tipo:** {best_full['type']}\n")
            f.write(f"- ⭐ **Accuracy:** **{best_full['accuracy']:.2%}**\n")
            f.write(f"- 📊 **Clases:** {best_full['classes']}\n\n")

        if demo_models:
            best_demo = max(demo_models, key=lambda x: x["accuracy"])
            f.write(f"### Mejor Modelo (Dataset Demo - 10 clases)\n\n")
            f.write(f"- 🎯 **Modelo:** `{best_demo['name']}`\n")
            f.write(f"- 🧠 **Tipo:** {best_demo['type']}\n")
            f.write(f"- ⭐ **Accuracy:** **{best_demo['accuracy']:.2%}**\n")
            f.write(f"- 📊 **Clases:** {best_demo['classes']}\n\n")

        # Tabla Completa organizada
        f.write("---\n\n")
        f.write("## 📋 Modelos Analizados\n\n")

        # Definir orden de modelos
        model_order = {'Classic ML': 1, 'CNN': 2, 'Transformer': 3}

        if full_models:
            f.write("### 🌳 Modelos FULL (163 clases)\n\n")
            f.write("| Modelo | Tipo | Clases | Accuracy | Épocas |\n")
            f.write("|--------|------|--------|----------|--------|\n")

            sorted_full = sorted(full_models, key=lambda x: (model_order.get(x['family'], 99), -x['accuracy']))
            for r in sorted_full:
                epochs_str = str(r["epochs"]) if r["epochs"] != "-" else "-"
                f.write(f"| {r['name']} | {r['type']} | {r['classes']} | {r['accuracy']:.2%} | {epochs_str} |\n")
            f.write("\n")

        if demo_models:
            f.write("### 🐣 Modelos DEMO (10 clases)\n\n")
            f.write("| Modelo | Tipo | Clases | Accuracy | Épocas |\n")
            f.write("|--------|------|--------|----------|--------|\n")

            sorted_demo = sorted(demo_models, key=lambda x: (model_order.get(x['family'], 99), -x['accuracy']))
            for r in sorted_demo:
                epochs_str = str(r["epochs"]) if r["epochs"] != "-" else "-"
                f.write(f"| {r['name']} | {r['type']} | {r['classes']} | {r['accuracy']:.2%} | {epochs_str} |\n")
            f.write("\n")

        # Gráficos
        f.write("---\n\n")
        f.write("## 📈 Visualizaciones\n\n")

        f.write("### Ranking General\n\n")
        f.write("![Ranking General](ranking_general.png)\n\n")

        f.write("### Comparativa por Tipo de Modelo\n\n")
        f.write("![Comparativa por Tipo](comparison_by_type.png)\n\n")

        f.write("### Accuracy vs Número de Clases\n\n")
        f.write("![Accuracy vs Clases](accuracy_vs_classes.png)\n\n")

        # Conclusiones mejoradas
        f.write("---\n\n")
        f.write("## 💡 Análisis y Conclusiones\n\n")

        # Análisis por familia
        families = {}
        for r in results:
            if r["family"] not in families:
                families[r["family"]] = []
            families[r["family"]].append(r["accuracy"])

        f.write("### Por Tipo de Modelo\n\n")
        family_order = ['Classic ML', 'CNN', 'Transformer']
        for family in family_order:
            if family in families:
                accs = families[family]
                avg = np.mean(accs)
                f.write(f"- **{family}:** Accuracy promedio = **{avg:.2%}**\n")
        f.write("\n")

        f.write("### Estadísticas Generales\n\n")
        if full_models:
            avg_acc = np.mean([r["accuracy"] for r in full_models])
            f.write(f"- **Accuracy promedio (Dataset Completo):** {avg_acc:.2%}\n")

        if demo_models:
            avg_acc_demo = np.mean([r["accuracy"] for r in demo_models])
            f.write(f"- **Accuracy promedio (Dataset Demo):** {avg_acc_demo:.2%}\n")

        f.write("\n---\n\n")
        f.write("*Reporte generado automáticamente por `06_analyze_metrics.py`*\n")

    print(f"✓ Reporte Markdown guardado: {os.path.abspath(report_path)}")

def main():
    ap = argparse.ArgumentParser("Comparador Avanzado de Modelos")
    ap.add_argument("--models", nargs="+", default=["all"],
                   help="Lista de carpetas de modelos o 'all' para todos")
    ap.add_argument("--outdir", default="06_analysis_out",
                   help="Donde guardar gráficas y reportes")
    args = ap.parse_args()

    # Determinar qué modelos analizar
    if args.models == ["all"] or "all" in args.models:
        print("--> Auto-detectando todos los modelos...")
        model_dirs = find_all_model_dirs()
        print(f"    Encontrados {len(model_dirs)} modelos")
    else:
        model_dirs = args.models

    if not model_dirs:
        print("❌ No se encontraron modelos para analizar.")
        return

    # Cargar métricas
    results = []
    print("\\n--> Cargando métricas...")
    for mdir in model_dirs:
        metrics = load_metrics(mdir)
        if metrics is None:
            print(f"    ⚠ No se encontró metrics.json en {mdir}")
            continue

        info = extract_model_info(mdir, metrics)
        results.append(info)
        print(f"    ✓ {info['name']}: {info['accuracy']:.2%}")

    if not results:
        print("❌ No se pudieron cargar métricas de ningún modelo.")
        return

    # Mostrar tabla
    print_comparison_table(results)

    # Crear directorio de salida
    os.makedirs(args.outdir, exist_ok=True)

    # Generar visualizaciones
    print("\\n--> Generando visualizaciones...")
    create_ranking_chart(results, args.outdir)
    create_bar_chart(results, args.outdir)
    create_scatter_plot(results, args.outdir)

    # Generar reporte
    print("\\n--> Generando reporte...")
    generate_markdown_report(results, args.outdir)

    print(f"\\n✅ Análisis completado. Resultados en: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
