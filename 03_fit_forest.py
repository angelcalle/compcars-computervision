#!/usr/bin/env python3
# 03_train_forest.py
#
# ==============================================================================
# PROYECTO: Clasificación de Coches (CompCars)
# PASO 3: MODELO CLASICO (RANDOM FOREST)
# ==============================================================================
#
# OBJETIVO:
#   Entrenar un modelo "clásico" de Machine Learning para tener una línea base.
#   Antes de usar Deep Learning, siempre hay que probar algo simple.
#
# ESTRATEGIA:
#   1. Extraer características (features) manuales de cada imagen:
#      - Usaremos Histogramas de Color (contar cuánto rojo, verde y azul hay).
#      - Es una técnica antigua pero rápida.
#   2. Entrenar un RandomForest (conjunto de árboles de decisión).
#   3. Evaluar qué tal funciona.
#
# USO:
#   python 03_fit_forest.py --compcars ./CompCars --splits-dir ./01_splits --use-test --outdir 03_models_forest

import os
import json
import argparse
import numpy as np

# Librerías de Modelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    import cv2 # OpenCV para leer imágenes
except ImportError:
    raise SystemExit("Falta OpenCV. Instala: pip install opencv-python")

try:
    import joblib # Para guardar el modelo entrenado
except ImportError:
    raise SystemExit("Falta joblib. Instala: pip install joblib")


# ==============================================================================
# FUNCIONES DE CARGA Y PREPROCESADO
# ==============================================================================

def read_list(txt_path):
    """Lee lista de rutas desde un txt."""
    rels = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rels.append(line.strip())
    return rels


def make_id_from_relpath(rel_path):
    """Extrae el ID de la marca (carpeta raíz) desde la ruta relativa."""
    parts = rel_path.replace("\\", "/").split("/")
    return parts[0] if parts else None

def extract_features(img_bgr, size=64, hist_bins=16):
    """
    INGENIERÍA DE CARACTERÍSTICAS (Feature Engineering):
    Convierte una imagen (matriz de píxeles) en un vector de números (features)
    que el Forest pueda entender.

    Técnica: HISTOGRAMA DE COLOR.
    1. Redimensionamos la imagen a pequeñita (size x size) para ir rápido.
    2. Calculamos histogramas para canales B, G y R.
    3. Concatenamos todo en un vector largo.
    """
    # Resize para estandarizar (aunque el histograma no depende del tamaño espacial, ayuda a limpiar ruido)
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    feats = []
    # Para cada canal (0=Blue, 1=Green, 2=Red en OpenCV)
    for ch in range(3):
        # cv2.calcHist(imagenes, canales, mascara, bins, rango)
        hist = cv2.calcHist([img], [ch], None, [hist_bins], [0, 256]).flatten()
        feats.append(hist)

    # Concatenar los 3 histogramas
    x = np.concatenate(feats).astype(np.float32)

    # Normalizar (para que sume 1 y no dependa del brillo absoluto/tamaño)
    s = x.sum()
    if s > 0:
        x /= s
    return x


def build_xy(compcars_root, rel_paths, size=64, hist_bins=16,
        per_make=None, max_makes=None, seed=42):
    """
    Carga imágenes cargando, calculando features (X) y guardando sus etiquetas (y).
    """
    print(f"--> Extrayendo features de {len(rel_paths)} imágenes...")
    rng = np.random.default_rng(seed)
    image_root = os.path.join(compcars_root, "image")

    # 1. Agrupar por marca para poder limitar/filtrar si queremos
    by_make = {}
    for rel in rel_paths:
        mk = make_id_from_relpath(rel)
        if mk is None: continue
        by_make.setdefault(mk, []).append(rel)

    makes = sorted(by_make.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # Filtro: Limitar número de marcas (clases)
    if max_makes is not None and max_makes > 0:
        makes = makes[:max_makes]

    selected = []
    for mk in makes:
        items = by_make[mk]
        rng.shuffle(items)
        # Filtro: Limitar imágenes por marca
        if per_make is not None and per_make > 0:
            items = items[:per_make]
        selected.extend(items)

    # 2. Procesar imágenes
    X, y = [], []
    missing = 0

    # tqdm wrapper
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs): return x

    for rel in tqdm(selected, desc="Extrayendo Features"):
        mk = make_id_from_relpath(rel)
        # Reconstruir ruta completa
        img_path = os.path.join(image_root, *rel.replace("\\", "/").split("/"))

        img = cv2.imread(img_path) # Carga BGR
        if img is None:
            missing += 1
            continue

        # Extraer vector de características
        features = extract_features(img, size=size, hist_bins=hist_bins)

        X.append(features)
        y.append(mk)

    if missing:
        print(f"Aviso: {missing} imágenes no encontradas o corruptas.")

    return np.array(X, dtype=np.float32), np.array(y)


def split_train_val(rel_paths, val_ratio=0.2, seed=42):
    """
    Divide la lista de entrenamiento en:
    - Train (para aprender)
    - Validacion para verificar mientras ajustamos parametros, durante el entrenamiento, verificas si el modelo está mejorando o empeorando (early stopping, ajustar learning rate, etc.)
    """
    rng = np.random.default_rng(seed)
    # Hacemos copia para no modificar la original
    idx = np.arange(len(rel_paths))
    rng.shuffle(idx)

    n_val = int(len(rel_paths) * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # Convertir a numpy array para indexar fácil y luego volver a lista
    arr_paths = np.array(rel_paths, dtype=object)
    return arr_paths[train_idx].tolist(), arr_paths[val_idx].tolist()


def load_make_map_from_mat(compcars_root):
    """Carga nombres de marcas"""
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


# ==============================================================================
# FUNCIONES DE ALTO NIVEL (Lógica del Pipeline)
# ==============================================================================

def load_and_split_data(splits_dir, val_ratio=0.2, seed=42):
    """Carga train_all.txt y divide en train/validation"""
    train_path = os.path.join(splits_dir, "train_all.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Error: falta {train_path}")

    rels = read_list(train_path)
    print(f"Total imágenes train: {len(rels)}")

    train_rels, val_rels = split_train_val(rels, val_ratio=val_ratio, seed=seed)
    print(f"Split Interno -> Train: {len(train_rels)} | Val: {len(val_rels)}")

    return train_rels, val_rels

def prepare_datasets(compcars_dir, train_rels, val_rels, max_makes=0, per_make=0, seed=42):
    """Construye matrices X,y para train y validation"""
    print("\n--- Procesando Dataset de ENTRENAMIENTO ---")
    X_train, y_train = build_xy(
        compcars_dir, train_rels,
        max_makes=max_makes, per_make=per_make, seed=seed
    )

    print("\n--- Procesando Dataset de VALIDACIÓN ---")
    X_val, y_val = build_xy(
        compcars_dir, val_rels,
        max_makes=max_makes, per_make=per_make, seed=seed
    )

    if len(X_train) == 0:
        raise ValueError("Error: Dataset de entrenamiento vacío.")

    return X_train, y_train, X_val, y_val

def encode_labels(y_train, y_val):
    """
    Convierte etiquetas de texto a números.
    Filtra validación para eliminar clases no vistas en train.
    """
    # -------------------------------------------------------------------------
    # TODO: Codificar Etiquetas (Label Encoding)
    #
    # TAREA: Los modelos ML no entienden texto ("Audi"), solo números.
    # 1. Instancia un sklearn.preprocessing.LabelEncoder
    # 2. Usa .fit_transform() con las etiquetas de TRAIN (y_train).
    # 3. IMPORTANTE: Filtra las clases desconocidas en VALIDACIÓN.
    #    - Es posible que en validation haya marcas que no salieron en train.
    #    - Crea una máscara booleana para mantener solo las que están en le.classes_.
    # 4. Transforma y_val usando solo esas instancias válidas.
    #
    # RETORNO ESPERADO:
    #   return le, y_train_enc, y_val_enc, mask_val, y_val_filtered
    # -------------------------------------------------------------------------

def train_model(X_train, y_train_enc, n_estimators=100, seed=42):
    """Entrena el modelo Random Forest"""
    # -------------------------------------------------------------------------
    # TODO: Entrenar el Modelo (Training)
    #
    # TAREA: Instanciar y entrenar el clasificador RandomForest.
    # 1. Crea una instancia de RandomForestClassifier.
    #    - Configura n_estimators con el argumento recibido.
    #    - Usa random_state=seed para reproducibilidad.
    #    - Usa n_jobs=-1 para que vaya rápido (usa todos los núcleos de CPU).
    # 2. Llama al método .fit() pasándole:
    #    - Los datos de entrenamiento (X_train).
    #    - Las etiquetas codificadas (y_train_enc).
    #
    # RETORNO ESPERADO:
    #   return clf  (el modelo ya entrenado)
    # -------------------------------------------------------------------------

def evaluate_model(clf, X_val, y_val_enc, le, compcars_dir):
    """Evalúa el modelo en validación y muestra reporte"""
    # -------------------------------------------------------------------------
    # TODO: Evaluar el Modelo
    #
    # TAREA: Calcular métricas de rendimiento con los datos de validación.
    # 1. Usa clf.predict() con los datos de validación (X_val).
    # 2. Compara esas predicciones con las reales (y_val_enc) usando:
    #    - accuracy_score() para el porcentaje global de aciertos.
    # 3. (Opcional) Genera un reporte detallado con classification_report().
    #    - TRUCO: Si hay muchas clases (>20), el reporte es ilegible.
    #      Haz un 'if' para mostrarlo solo si len(le.classes_) < 20.
    #    - Usa target_names para mostrar nombres como "Audi" en vez de "0".
    #
    # RETORNO ESPERADO:
    #   No suele retornar nada crítico, pero imprime los resultados en pantalla.
    # -------------------------------------------------------------------------

def evaluate_test_set(clf, le, compcars_dir, splits_dir, max_makes=0, per_make=0, seed=42):
    """Evalúa el modelo en el conjunto de test (opcional)"""
    test_path = os.path.join(splits_dir, "test_all.txt")
    if not os.path.exists(test_path):
        return None

    print("\n--- Procesando Dataset de TEST (Final) ---")
    test_rels = read_list(test_path)
    X_test, y_test = build_xy(
        compcars_dir, test_rels,
        max_makes=max_makes, per_make=per_make, seed=seed
    )

    # Filtrar clases desconocidas
    known = set(le.classes_)
    mask_test = np.array([y in known for y in y_test])
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    if len(X_test) == 0:
        print("Aviso: No hay imágenes de test válidas (todas son clases desconocidas o no encontradas).")
        return None

    y_test_enc = le.transform(y_test)
    pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test_enc, pred_test)
    print(f"Accuracy en TEST: {acc_test:.2%}")

    return {"accuracy": acc_test, "samples": len(X_test)}

def save_model_artifacts(clf, le, compcars_dir, outdir, acc_val, metrics_test=None):
    """Guarda modelo, encoder y metadatos"""
    os.makedirs(outdir, exist_ok=True)
    print(f"\nGuardando resultados en {os.path.abspath(outdir)} ...")

    # Guardar modelo y encoder
    joblib.dump(clf, os.path.join(outdir, "forest_model.joblib"))
    joblib.dump(le, os.path.join(outdir, "label_encoder.joblib"))

    # Guardar configuración de features
    joblib.dump({"size": 64, "hist_bins": 16}, os.path.join(outdir, "feature_config.joblib"))

    # Mapeo de nombres de marcas
    make_map = load_make_map_from_mat(compcars_dir)
    # Crear mapeo de índices a nombres legibles para el JSON (formato idx_to_label esperado por demo)
    idx_to_name = {}
    for idx, make_id in enumerate(le.classes_):
        make_name = make_map.get(make_id, make_id)
        idx_to_name[str(idx)] = make_name

    # Guardar formato compatible con el nuevo standard
    # 1. label_names.json simple (Legacy pero útil)
    label_names_simple = {str(c): make_map.get(c, str(c)) for c in le.classes_}
    with open(os.path.join(outdir, "label_names.json"), "w", encoding='utf-8') as f:
         json.dump(label_names_simple, f, indent=2, ensure_ascii=False)

    # 2. label_map.json (Standard nuevo usado por demo)
    with open(os.path.join(outdir, "label_map.json"), "w", encoding='utf-8') as f:
        json.dump({"idx_to_label": idx_to_name}, f, ensure_ascii=False, indent=2)

    # Métricas
    metrics = {
        "model_type": "Random Forest",
        "num_classes": len(le.classes_),
        "val_accuracy": acc_val,
        "test_metrics": metrics_test
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Modelo guardado: forest_model.joblib")
    print(f"✅ {len(le.classes_)} clases detectadas")
    print(f"✅ Accuracy final (val): {acc_val:.2%}")

# ==============================================================================
# MAIN (Pipeline simplificado)
# ==============================================================================

def main():
    # -------------------------------------------------------------------------
    # TODO: Configurar Argumentos (argparse)
    #
    # TAREA: Define los argumentos necesarios para el script:
    # 1. --compcars (obligatorio): Ruta al dataset
    # 2. --splits-dir (default "01_splits"): Dónde están los .txt
    # 3. --outdir (default "03_models_forest"): Dónde guardar resultados
    # 4. Hiperparámetros del Random Forest (--n-estimators, --seed, etc.)
    # 5. Límites para demo (--max-makes, --per-make)
    #
    # PISTA: Usa argparse.ArgumentParser()
    # -------------------------------------------------------------------------
    # ap = ...
    # ...
    # args = ap.parse_args()

    try:
        # 1. Cargar y dividir datos
        train_rels, val_rels = load_and_split_data(args.splits_dir, args.val_ratio, args.seed)

        # 2. Preparar datasets (Features + Labels)
        X_train, y_train, X_val, y_val = prepare_datasets(
            args.compcars, train_rels, val_rels,
            max_makes=args.max_makes, per_make=args.per_make, seed=args.seed
        )

        # 3. Encoding de etiquetas
        le, y_train_enc, y_val_enc, mask_val, y_val_filtered = encode_labels(y_train, y_val)
        X_val = X_val[mask_val] # Aplicar máscara a X_val

        # 4. Entrenar
        clf = train_model(X_train, y_train_enc, args.n_estimators, args.seed)

        # 5. Evaluar en Validación
        acc_val = evaluate_model(clf, X_val, y_val_enc, le, args.compcars)

        # 6. Evaluar en Test (Opcional)
        metrics_test = None
        if args.use_test:
            metrics_test = evaluate_test_set(
                clf, le, args.compcars, args.splits_dir,
                max_makes=args.max_makes, per_make=args.per_make, seed=args.seed
            )

        # 7. Guardar artefactos
        save_model_artifacts(clf, le, args.compcars, args.outdir, acc_val, metrics_test)

        print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE.")

    except Exception as e:
        print(f"\nError fatal: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()
