# 🚗 Proyecto: Clasificación de Marcas de Coches (CompCars)

Este proyecto es una guía práctica y educativa para comprender el flujo completo de un proyecto profesional de **Visión por Computador**. Está diseñado para estudiantes del ciclo de **Desarrollo de Aplicaciones Multiplataforma** del centro **Metrodora FP**.

Cubrimos desde el análisis de datos (**EDA**) hasta el uso de modelos de última generación (**Transformers**), pasando por modelos clásicos.

> 🎓 **Idea clave para el alumnado:** este repositorio está pensado como material de estudio. Por eso verás explicaciones y pasos que pueden parecer redundantes: están ahí para ayudarte a no perderte.

---

## 🧭 Guía de Inicio Rápido

### ✅ Requisitos Previos

Necesitas tener **Python 3.9 o superior** instalado.

### 📦 Instalar dependencias

**Instalar dependencias:** Hemos preparado un archivo con todo lo necesario. Ejecuta:

```bash
pip install -r requirements.txt
```

*Nota: Si tienes una tarjeta gráfica NVIDIA, te recomendamos instalar PyTorch compatible con CUDA para que el entrenamiento sea más rápido (ver pytorch.org).*

---

## 🗂️ Estructura de Proyecto

Cada script genera su propia carpeta numerada para mantener el orden. Aquí está la estructura completa con todos los archivos generados:

```text
ConferencIA2/
├── CompCars/                       # 📁 DATASET ORIGINAL (No tocar)
│   ├── image/                      # Imágenes de coches
│   └── misc/                       # Metadatos (nombres de marcas, etc.)
│
├── 01_splits/                      # ✅ Generado por 01_prepare_splits.py
│   ├── train_all.txt               # Lista de imágenes para entrenamiento
│   ├── test_all.txt                # Lista de imágenes para prueba
│   └── make_counts.json            # Estadísticas de marcas
│
├── 02_eda_out/                     # 📊 Generado por 02_eda_main.py
│   ├── stats_counts.csv            # Conteo de imágenes por marca
│   ├── plot_*.png                  # 10+ gráficos de análisis
│   ├── analysis_*.png              # Gráficos avanzados (3D, calidad, etc.)
│   ├── dataset_summary.csv         # Resumen estadístico completo
│   └── grids/                      # Mosaicos de imágenes por marca
│
├── 03_models_forest/               # 🌳 Modelos Random Forest (163 clases)
│   ├── forest_model.joblib         # Modelo entrenado
│   ├── label_encoder.joblib        # Codificador de etiquetas
│   ├── feature_config.joblib       # Configuración de features
│   ├── label_names.json            # Nombres de marcas legibles
│   └── metrics.json                # Accuracy y métricas
│
├── 03_models_forest_demo/          # 🌳 Versión DEMO (10 clases)
│   └── (mismos archivos)
│
├── 04_models_cnn/                  # 🧠 Modelos CNN/MobileNetV3 (163 clases)
│   ├── cnn_best.pt                 # Pesos del modelo (mejor época)
│   ├── label_map.json              # Mapeo índice → nombre de marca
│   └── metrics.json                # Accuracy y métricas
│
├── 04_models_cnn_demo/             # 🧠 Versión DEMO (10 clases)
│   └── (mismos archivos)
│
├── 05_models_vit/                  # 🔮 Modelos Vision Transformer (163 clases)
│   ├── vit_best.pt                 # Pesos del modelo (mejor época)
│   ├── label_map.json              # Mapeo índice → nombre de marca
│   └── metrics.json                # Accuracy y métricas
│
├── 05_models_vit_demo/             # 🔮 Versión DEMO (10 clases)
│   └── (mismos archivos)
│
├── 06_analysis_out/                # 📊 Generado por 06_analyze_metrics.py
│   ├── ranking_general.png         # Gráfico de ranking de modelos
│   ├── comparison_by_type.png      # Comparativa por tipo de modelo
│   ├── accuracy_vs_classes.png     # Scatter plot accuracy vs clases
│   └── model_comparison_report.md  # Reporte detallado en Markdown
│
├── images/                         # 🖼️ Tus propias fotos (crear esta carpeta)
│   └── *.jpg, *.png, etc.          # Fotos de coches para predecir
│
└── Scripts principales:
    ├── 01_prepare_splits.py        # Preparar datos train/test
    ├── 02_eda_main.py              # Análisis exploratorio de datos
    ├── 03_fit_forest.py            # Entrenar Random Forest
    ├── 03_predict_forest.py        # Predecir con Random Forest
    ├── 04_fit_cnn.py               # Entrenar CNN (MobileNetV3)
    ├── 04_predict_cnn.py           # Predecir con CNN
    ├── 05_fit_vit.py               # Entrenar Vision Transformer
    ├── 05_predict_vit.py           # Predecir con ViT
    ├── 06_analyze_metrics.py       # Comparar todos los modelos
    ├── 07_live_demo.py             # Demo web interactiva (Streamlit)
    └── 08_predict_local.py         # Predicción masiva (todos los modelos)
```

### 📌 Leyenda

* 📁 Dataset original
* ✅ Archivos de configuración
* 📊 Visualizaciones y reportes
* 🌳 Modelos clásicos (Random Forest)
* 🧠 Redes neuronales (CNN)
* 🔮 Transformers (ViT)
* 🖼️ Tus imágenes

---

## 1️⃣ Paso 1: Preparar Datos

> Aquí generamos los ficheros `train/test` que usarán todos los modelos.

```bash
# Opción A: PREPARACIÓN COMPLETA
python 01_prepare_splits.py --compcars ./CompCars --outdir .

# Opción B: PREPARACIÓN DEMO (Rápida)
python 01_prepare_splits.py --compcars ./CompCars --outdir . --max-makes 10 --per-make 500 --top-makes
```

**Esto creará la carpeta `01_splits`.**

---

## 2️⃣ Paso 2: Entender los Datos (EDA)

> Antes de entrenar modelos, analizamos el dataset: cuántas imágenes hay por marca, ejemplos visuales, calidad, etc.

```bash
# Análisis Completo (Gráficos + Stats)
python 02_eda_main.py --compcars ./CompCars --splits-dir ./01_splits --mode full

# Solo visualizar Mosaicos
python 02_eda_main.py --compcars ./CompCars --mode visualize --topn 5
```

**Esto creará la carpeta `02_eda_out`.**

---

## 🧠 Entrenamiento de Modelos

> En este proyecto entrenamos 3 familias de modelos: **clásico (RF)**, **CNN (MobileNetV3)**, y **Transformer (ViT)**.
> Cada uno tiene **Modo DEMO (rápido, 10 clases)** y **Modo REAL/FULL (más lento, 163 clases)**.

---

### 🌳 Modelo A: Random Forest

#### Modo Demo

```bash
python 03_fit_forest.py --compcars ./CompCars --splits-dir ./01_splits --max-makes 10 --per-make 100 --n-estimators 50 --outdir 03_models_forest_demo
```

#### Modo Real

```bash
python 03_fit_forest.py --compcars ./CompCars --splits-dir ./01_splits --use-test --outdir 03_models_forest
```

**Esto creará la carpeta `03_models_forest`.**

#### Predicción

**Demo:**

```bash
python 03_predict_forest.py --image ./CompCars/image/72/710/2009/a7eb72b87cc67a.jpg --modeldir 03_models_forest_demo --compcars ./CompCars
python 03_predict_forest.py --image "images/Audi A5 Sportback.jpg" --modeldir 03_models_forest_demo --compcars ./CompCars
```

**Real:**

```bash
python 03_predict_forest.py --image ./CompCars/image/72/710/2009/a7eb72b87cc67a.jpg --modeldir 03_models_forest --compcars ./CompCars
python 03_predict_forest.py --image "images/Audi A5 Sportback.jpg" --modeldir 03_models_forest --compcars ./CompCars
```

---

### 🧠 Modelo B: CNN (MobileNetV3)

#### Modo Demo

```bash
python 04_fit_cnn.py --compcars ./CompCars --splits-dir ./01_splits --max-makes 10 --per-make 100 --epochs 3 --freeze-backbone --outdir 04_models_cnn_demo
```

#### Modo Real

```bash
python 04_fit_cnn.py --compcars ./CompCars --splits-dir ./01_splits --epochs 10 --outdir 04_models_cnn
```

**Esto creará la carpeta `04_models_cnn`.**

#### Predicción

**Demo:**

```bash
python 04_predict_cnn.py --image ./CompCars/image/72/710/2009/a7eb72b87cc67a.jpg --modeldir 04_models_cnn_demo --compcars ./CompCars
python 04_predict_cnn.py --image "images/Audi A5 Sportback.jpg" --modeldir 04_models_cnn_demo --compcars ./CompCars
```

**Real:**

```bash
python 04_predict_cnn.py --image ./CompCars/image/72/710/2009/a7eb72b87cc67a.jpg --modeldir 04_models_cnn --compcars ./CompCars
python 04_predict_cnn.py --image "images/Audi A5 Sportback.jpg" --modeldir 04_models_cnn --compcars ./CompCars
```

---

### 🔮 Modelo C: Vision Transformer (ViT)

#### Modo Demo

```bash
python 05_fit_vit.py --compcars ./CompCars --splits-dir ./01_splits --max-makes 10 --per-make 100 --epochs 3 --freeze-backbone --outdir 05_models_vit_demo
```

#### Modo Real

```bash
python 05_fit_vit.py --compcars ./CompCars --splits-dir ./01_splits --epochs 5 --outdir 05_models_vit
```

**Esto creará la carpeta `05_models_vit`.**

#### Predicción

**Demo:**

```bash
python 05_predict_vit.py --image ./CompCars/image/72/710/2009/a7eb72b87cc67a.jpg --modeldir 05_models_vit_demo --compcars ./CompCars
python 05_predict_vit.py --image "images/Audi A5 Sportback.jpg" --modeldir 05_models_vit_demo --compcars ./CompCars
```

**Real:**

```bash
python 05_predict_vit.py --image ./CompCars/image/72/710/2009/a7eb72b87cc67a.jpg --modeldir 05_models_vit --compcars ./CompCars
python 05_predict_vit.py --image "images/Audi A5 Sportback.jpg" --modeldir 05_models_vit --compcars ./CompCars
```

---

## 📊 Comparativa Final

Compara el rendimiento de todos tus modelos con gráficos y reportes automáticos:

### Comparar TODOS los modelos (Full + Demo)

```bash
python 06_analyze_metrics.py --models all
```

### Comparar solo modelos DEMO (10 clases)

```bash
python 06_analyze_metrics.py --models 03_models_forest_demo 04_models_cnn_demo 05_models_vit_demo
```

### Comparar solo modelos FULL (163 clases)

```bash
python 06_analyze_metrics.py --models 03_models_forest 04_models_cnn 05_models_vit
```

Genera automáticamente: **tabla comparativa, 3 gráficos profesionales y reporte en Markdown.**

---

## 🚀 LIVE DEMO (Web Interactiva)

Una vez tengas modelos entrenados, lanza la web interactiva para probarlos con diseño visual:

```bash
# python -m streamlit run 07_live_demo.py
```

*(Se abrirá automáticamente en tu navegador)*

---

## 📁 Predicción Masiva (Escanear carpeta completa)

¿Tienes varias fotos? Ponlas todas en la carpeta `images/` y escanéalas automáticamente con un solo script que funciona con todos los modelos:

```bash
# Con cualquier modelo - el script auto-detecta el tipo
python 08_predict_local.py --modeldir 03_models_forest_demo
python 08_predict_local.py --modeldir 04_models_cnn_demo
python 08_predict_local.py --modeldir 05_models_vit_demo

# También funciona con modelos full
python 08_predict_local.py --modeldir 03_models_forest
python 08_predict_local.py --modeldir 04_models_cnn
python 08_predict_local.py --modeldir 05_models_vit
```

Soporta 8 formatos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`, `.webp`

---


## 🧾 Fuentes oficiales

🔗 **Repositorio original del dataset (Universidad de Hong Kong)**  
https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

🔗 **Versión en Kaggle (puede requerir inicio de sesión)**  
https://www.kaggle.com/datasets/wheelernando/compcars

> ⚠️ Recuerda que el dataset no es de dominio público y su uso está limitado a fines educativos y de investigación, no comerciales. Consulta siempre los términos de uso de cada fuente antes de descargar y utilizar el contenido.

---

## 📜 Licencias

### 📌 Licencia del dataset *CompCars*

Este repositorio incluye el dataset **CompCars** solo con fines **educativos y de investigación** conforme a los términos indicados por sus autores.  
El dataset *CompCars* no es de dominio público y las imágenes no pueden ser reproducidas, vendidas o explotadas para fines **comerciales**.  
Los datos fueron recopilados de internet y no son propiedad del grupo que lo distribuye.  
Para más detalles sobre acceso y condiciones originales, visita la página oficial del dataset:  
https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ :contentReference[oaicite:1]{index=1}

> ⚠️ Si planeas usar el dataset con fines comerciales fuera del ámbito educativo, consulta los términos y contactos del proyecto original.*

---

### 🧾 Licencia del código de este repositorio

El código de este proyecto se publica bajo **The Unlicense**, lo que implica que se dedica al dominio público (donde la ley lo permita), y puede ser copiado, modificado, publicado y distribuido libremente:

```

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. ...

For more information, please refer to [https://unlicense.org](https://unlicense.org)

```

---

## 👤 Autor

**[Ángel Calle Serrano]** 
Colaborador del ciclo de Desarrollo de Aplicaciones Multiplataforma (DAM)  
Metrodora FP  
2026

Repositorio creado como material docente para introducir Visión por Computador y Deep Learning.

Contacto (opcional):  
- GitHub: https://github.com/AngelCalle
- LinKedin: https://www.linkedin.com/in/angelcalleserrano/
