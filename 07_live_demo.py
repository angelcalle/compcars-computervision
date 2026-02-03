import streamlit as st
import os
import json
import torch
import cv2
import numpy as np
import joblib
from PIL import Image
from torchvision import transforms, models
from torchvision.models import mobilenet_v3_large, vit_b_16

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="CompCars Demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Clasificador de Coches en Tiempo Real")
st.markdown("""
Esta aplicación permite probar los modelos de Inteligencia Artificial entrenados en el curso.
Sube una foto de un coche y ve cómo "piensan" los distintos modelos.
""")

# ==============================================================================
# FUNCIONES DE CARGA DE MODELOS (CACHED)
# ==============================================================================

# Labels map como fallback (ya no se necesita con label_names.json)
labels_map = {}

@st.cache_resource
def load_rf_model(model_dir):
    """Carga Random Forest"""
    model_path = os.path.join(model_dir, "forest_model.joblib")
    enc_path = os.path.join(model_dir, "label_encoder.joblib")
    cfg_path = os.path.join(model_dir, "feature_config.joblib")
    names_path = os.path.join(model_dir, "label_names.json")

    if not os.path.exists(model_path): return None, None, None, None

    clf = joblib.load(model_path)
    le = joblib.load(enc_path)
    config = joblib.load(cfg_path)

    # Cargar nombres de marcas si existe el archivo
    label_names = None
    if os.path.exists(names_path):
        with open(names_path, "r", encoding='utf-8') as f:
            label_names = json.load(f).get("idx_to_label", {})

    return clf, le, config, label_names

@st.cache_resource
def load_cnn_model(model_dir, device):
    """Carga MobileNetV3"""
    path = os.path.join(model_dir, "cnn_best.pt")
    map_path = os.path.join(model_dir, "label_map.json")

    if not os.path.exists(path): return None, None

    with open(map_path, "r") as f:
        idx_to_label = json.load(f)["idx_to_label"]

    num_classes = len(idx_to_label)
    model = mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, idx_to_label

@st.cache_resource
def load_vit_model(model_dir, device):
    """Carga Vision Transformer"""
    path = os.path.join(model_dir, "vit_best.pt")
    map_path = os.path.join(model_dir, "label_map.json")

    if not os.path.exists(path): return None, None

    with open(map_path, "r") as f:
        idx_to_label = json.load(f)["idx_to_label"]

    num_classes = len(idx_to_label)
    model = vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, idx_to_label

# ==============================================================================
# PREPROCESAMIENTO
# ==============================================================================

def preprocess_rf(pil_img, size=64, bins=16):
    """Extract features para Random Forest (Color Histogram)"""
    img = np.array(pil_img.convert("RGB"))
    # Convertir a BGR (OpenCV standard)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (size, size))
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [bins], [0, 256]).flatten()
        feats.append(hist)
    x = np.concatenate(feats).astype(np.float32)
    x /= (x.sum() + 1e-6)
    return x.reshape(1, -1)

def preprocess_torch(pil_img, size=224):
    """Tensor transform para CNN/ViT"""
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tf(pil_img).unsqueeze(0)

# ==============================================================================
# INTERFAZ LATERAL
# ==============================================================================

st.sidebar.header("Configuración")

model_type = st.sidebar.radio(
    "Elige el Modelo:",
    ("Random Forest (Clásico)", "MobileNetV3 (CNN)", "Vision Transformer (ViT)")
)

# Detectar carpetas disponibles
base_dirs = {
    "MobileNetV3 (CNN)": "04_models_cnn",
    "Vision Transformer (ViT)": "05_models_vit",
    "Random Forest (Clásico)": "03_models_forest"
}

# Permitir al usuario cambiar la carpeta si usa nombres custom (demo vs full)
default_dir = base_dirs[model_type]
custom_dir = st.sidebar.text_input("Carpeta del Modelo:", value=default_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Dispositivo de inferencia: **{device.upper()}**")

# ==============================================================================
# LÓGICA PRINCIPAL
# ==============================================================================

# Cargar imagen
uploaded_file = st.file_uploader("Sube una foto de un coche...", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Imagen Subida", use_container_width=True)

    with col2:
        st.subheader("Predicción")
        with st.spinner(f"Analizando con {model_type}..."):

            # --- Lógica RF ---
            if model_type == "Random Forest (Clásico)":
                clf, le, cfg, label_names = load_rf_model(custom_dir)
                if clf is None:
                    st.error(f"No se encontró el modelo en {custom_dir}")
                else:
                    x = preprocess_rf(image, size=cfg["size"], bins=cfg["hist_bins"])
                    probs = clf.predict_proba(x)[0]
                    pred_idx = np.argmax(probs)

                    # Usar label_names si está disponible, sino usar labels_map
                    if label_names:
                        pred_name = label_names.get(str(pred_idx), str(pred_idx))
                    else:
                        pred_id_str = le.inverse_transform([pred_idx])[0]
                        pred_name = labels_map.get(pred_id_str, pred_id_str)

                    confidence = probs[pred_idx]

                    # Top 3
                    top_indices = np.argsort(probs)[::-1][:3]
                    top_probs = probs[top_indices]

                    if label_names:
                        top_names = [label_names.get(str(i), str(i)) for i in top_indices]
                    else:
                        top_ids = le.inverse_transform(top_indices)
                        top_names = [labels_map.get(tid, tid) for tid in top_ids]

            # --- Lógica Deep Learning (CNN / ViT) ---
            else:
                if "CNN" in model_type:
                    net, labels = load_cnn_model(custom_dir, device)
                else:
                    net, labels = load_vit_model(custom_dir, device)

                if net is None:
                    st.error(f"No se encontró el modelo en {custom_dir}")
                    clf = None # Flag de error
                else:
                    x = preprocess_torch(image).to(device)
                    with torch.no_grad():
                        out = net(x)
                        probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()

                    pred_idx = np.argmax(probs)
                    # labels keys are text indices "0", "1"... convert to int match
                    pred_id_str = labels[str(pred_idx)]
                    # Map ID -> Name
                    pred_name = labels_map.get(pred_id_str, pred_id_str)

                    confidence = probs[pred_idx]

                    # Top 3
                    top_indices = np.argsort(probs)[::-1][:3]
                    top_probs = probs[top_indices]
                    # labels[str(i)] da el ID (ej "39"), labels_map da "Audi"
                    top_names = [labels_map.get(labels[str(i)], labels[str(i)]) for i in top_indices]
                    clf = True # Flag ok

            # --- Visualización de Resultados ---
            if clf is not None:
                # Barra de confianza grande
                if confidence > 0.8:
                    st.success(f"**{pred_name}** ({confidence:.1%})")
                    st.balloons()
                elif confidence > 0.5:
                    st.warning(f"Creo que es **{pred_name}** ({confidence:.1%})")
                else:
                    st.error(f"No estoy seguro... ¿**{pred_name}**? ({confidence:.1%})")

                # Gráfico de barras simple para Top 3
                st.write("---")
                st.write("**Top 3 Probabilidades:**")
                for n, p in zip(top_names, top_probs):
                    st.write(f"{n}")
                    st.progress(float(p))

else:
    st.info("Sube una imagen para empezar o usa el menú de la izquierda para cambiar de modelo.")

st.write("---")
st.caption("Proyecto Educativo CompCars - Demo Generada Automáticamente")
