
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests
import gc

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="LungScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

# --------------------------------------------------------------------------------
# Custom CSS - Redesigned for Medical Dashboard Results
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f6f8;
        color: #1a1a1a;
    }

    /* Header */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: white;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2e7d32; /* Primary Medical Green */
        letter-spacing: -1px;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #555;
    }

    /* Results Container */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #eee;
    }

    /* Diagnosis Text Styles */
    .diagnosis-header {
        font-size: 1rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    .diagnosis-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        margin-top: 1rem;
    }

    .diagnosis-value-red {
        font-size: 1.6rem;
        font-weight: 800;
        color: #c62828; /* Strong Red for alerts/malignant */
        line-height: 1.2;
    }
    
    .diagnosis-value-green {
        font-size: 1.6rem;
        font-weight: 800;
        color: #2e7d32; /* Green for benign/normal */
        line-height: 1.2;
    }

    /* Image Styling */
    .scan-image {
        border-radius: 12px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .status-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Custom Button Areas - Streamlit buttons are hard to style via CSS alone, 
       but we can target them generally */
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }

    /* Primary Action (Green) - We will rely on st.button type="primary" if available or default styling */
    
    /* Secondary Action (White) */
    
    /* Upload Area */
    [data-testid="stFileUploader"] {
        padding: 2rem;
        border: 2px dashed #a5d6a7;
        border-radius: 12px;
        background-color: #fff;
    }

    /* Hide Streamlit cruft */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Logic & Helpers
# --------------------------------------------------------------------------------
@st.cache_resource
def check_and_download_models():
    """Ensures models are downloaded but DOES NOT load them into memory."""
    configs = {
        "ResNet50": {
            "path": "chest-ct-scan-ResNet50.keras", 
            "url": "https://drive.google.com/uc?export=download&id=1pCpBhr3TMddm9xuEj7M7WXml5M6KkKP4",
        },
        "Severity": {
            "path": "Lung_Cancer_Severity_Model.h5", 
            "url": "https://drive.google.com/uc?export=download&id=1_6p87Sb-uKuHVLivvgYLZGbGTAvQwxwr",
        }
    }
    
    for name, info in configs.items():
        if not os.path.exists(info["path"]):
            with st.spinner(f"📥 Initializing {name} Engine..."):
                try:
                    response = requests.get(info["url"], stream=True)
                    if response.status_code == 200:
                        with open(info["path"], 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                except:
                    st.error(f"Failed to download {name}")
    return configs

def process_image(image_data, target_size):
    image = ImageOps.fit(image_data, target_size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    return np.expand_dims(np.array(image), axis=0)

def build_resnet_model():
    """Reconstructs the ResNet50 architecture to bypass deserialization errors."""
    try:
        base_model = tf.keras.applications.ResNet50(
            include_top=False, 
            weights=None, 
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        base_model._name = "resnet50"
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dropout(0.5, name="dropout"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.BatchNormalization(name="batch_normalization"),
            tf.keras.layers.Dropout(0.5, name="dropout_1"),
            tf.keras.layers.Dense(4, activation='softmax', name="dense") # 4 Classes
        ], name="resnet_sequential")
        
        model.build((None, 224, 224, 3))
        return model
    except Exception as e:
        return None

def predict_sequentially(uploaded_file, model_configs):
    """Loads one model, predicts, clears memory, loads next."""
    results = {}
    
    # Class Mappings
    # ResNet (4 classes): Adenocarcinoma, Large cell carcinoma, Normal, Squamous cell carcinoma
    resnet_classes = ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']
    
    # Severity (3 classes): Benign, Malignant, Normal
    severity_classes = ['Benign', 'Malignant', 'Normal']
    
    img = Image.open(uploaded_file)

    # 1. ResNet50 (Visual Detection)
    st.toast("Running Visual Detection Engine...", icon="🔍")
    tf.keras.backend.clear_session()
    gc.collect()
    
    try:
        # Build empty model structure
        model = build_resnet_model()
        if model is None:
             raise Exception("Failed to reconstruct ResNet architecture.")
             
        # Load weights
        model.load_weights(model_configs["ResNet50"]["path"], skip_mismatch=True)
        
        input_data = process_image(img, (224, 224))
        pred = model.predict(input_data)
        score = tf.nn.softmax(pred[0])
        idx = np.argmax(score)
        
        label = resnet_classes[idx]
        
        results["ResNet50"] = {
            "label": label,
            "conf": 100 * np.max(score),
            "scores": score,
            "classes": resnet_classes,
            "success": True
        }
        del model
    except Exception as e:
         results["ResNet50"] = {"success": False, "error": str(e)}

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()

    # 2. Severity (Severity Analysis)
    st.toast("Running Severity Analysis Engine...", icon="⚖️")
    
    try:
        # Load H5 model directly
        model = tf.keras.models.load_model(model_configs["Severity"]["path"], compile=False)
        
        input_data = process_image(img, (256, 256))
        pred = model.predict(input_data)
        score = tf.nn.softmax(pred[0])
        idx = np.argmax(score)
        
        label = severity_classes[idx]
        
        results["Severity"] = {
            "label": label,
            "conf": 100 * np.max(score),
            "scores": score,
            "classes": severity_classes,
            "success": True
        }
        del model
    except Exception as e:
         results["Severity"] = {"success": False, "error": str(e)}

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()
    
    return results

# --------------------------------------------------------------------------------
# Main Content
# --------------------------------------------------------------------------------

# 1. Header
st.markdown("""
<div class="header-container">
    <div style="font-size: 3.5rem;">🫁</div>
    <div class="app-title">Lung Cancer Detection</div>
    <div class="app-subtitle">AI Diagnostic System</div>
</div>
""", unsafe_allow_html=True)

# 2. Upload Section
col_spacer1, col_up, col_spacer2 = st.columns([1, 2, 1])
with col_up:
    uploaded_file = st.file_uploader("Upload Chest CT Scan for Analysis", type=["jpg", "png", "jpeg"])

# 3. Analysis Section
if uploaded_file:
    st.markdown("---")
    
    col_img, col_res = st.columns([1, 1.5], gap="large")
    
    with col_img:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.write("#### 📷 Input Scan")
        st.image(uploaded_file, use_column_width=True, caption="Source Image")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        configs = check_and_download_models()
        
        with st.spinner("🧠 Initializing Multi-Stage Analysis..."):
            # Run Sequential Prediction
            results = predict_sequentially(uploaded_file, configs)
            
            if results["ResNet50"]["success"] and results["Severity"]["success"]:
                r_res = results["ResNet50"]
                r_sev = results["Severity"]
                
                # --- CONSENSUS LOGIC ---
                # Check Malignancy Map
                # ResNet Classes: ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']
                res_is_malignant = r_res["label"] in ['Adenocarcinoma', 'Large cell carcinoma', 'Squamous cell carcinoma']
                res_is_benign = False # ResNet doesn't output Benign explicitly
                res_is_normal = r_res["label"] == 'Normal'
                
                # Severity Classes: ['Benign', 'Malignant', 'Normal']
                sev_is_malignant = r_sev["label"] == 'Malignant'
                sev_is_benign = r_sev["label"] == 'Benign'
                sev_is_normal = r_sev["label"] == 'Normal'
                
                final_status = "Unknown"
                if res_is_malignant or sev_is_malignant:
                    final_status = "MALIGNANT DETECTED"
                    final_color = "#c62828" # Red
                    icon = "🚨"
                elif sev_is_benign:
                    final_status = "BENIGN DETECTED"
                    final_color = "#f57f17" # Orange
                    icon = "⚠️"
                elif res_is_normal and sev_is_normal:
                    final_status = "NORMAL / HEALTHY"
                    final_color = "#2e7d32" # Green
                    icon = "✅"
                else:
                    # Fallback / Mixed
                    final_status = "INCONCLUSIVE / CHECK"
                    final_color = "#888"
                    icon = "❓"

                avg_conf = (r_res["conf"] + r_sev["conf"]) / 2

                # 1. Consensus Box
                st.markdown(f"""
                <div class="consensus-box" style="background: linear-gradient(135deg, {final_color} 0%, #333 100%);">
                    <div class="consensus-title">AI Consolidated Diagnosis</div>
                    <div class="consensus-result">{icon} {final_status}</div>
                    <div style="font-size:0.9rem; opacity:0.8;">Average Confidence: {avg_conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # 2. Detailed Breakdown
                c1, c2 = st.columns(2)
                
                # Colors for individual cards
                res_color_class = "diagnosis-value-red" if res_is_malignant else ("diagnosis-value-green" if res_is_normal else "diagnosis-value-green") # Default green for others?? No, keep logical
                if res_is_malignant: res_css = "color:#c62828;" 
                elif res_is_normal: res_css = "color:#2e7d32;"
                else: res_css = "color:#333;"
                
                sev_color_class = "diagnosis-value-red" if sev_is_malignant else ("diagnosis-value-green" if sev_is_normal or sev_is_benign else "diagnosis-value-green")
                if sev_is_malignant: sev_css = "color:#c62828;"
                elif sev_is_normal or sev_is_benign: sev_css = "color:#2e7d32;"
                else: sev_css = "color:#333;"

                with c1:
                    st.markdown(f"""
                    <div class="model-card resnet">
                        <div class="model-name">🔍 Visual Detection (ResNet50)</div>
                        <div style="font-size:1.1rem; font-weight:bold; {res_css}">{r_res['label']}</div>
                        <div style="color:#666; font-size:0.8rem;">Conf: {r_res['conf']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                     st.markdown(f"""
                    <div class="model-card severity">
                        <div class="model-name">⚖️ Severity Analysis</div>
                        <div style="font-size:1.1rem; font-weight:bold; {sev_css}">{r_sev['label']}</div>
                        <div style="color:#666; font-size:0.8rem;">Conf: {r_sev['conf']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed Probabilities
                with st.expander("📊 View Detailed Probabilities"):
                    st.write("**Visual Detection Model Probabilities:**")
                    st.bar_chart({lbl: float(r_res['scores'][i]) for i, lbl in enumerate(r_res['classes'])})
                    
                    st.write("**Severity Analysis Model Probabilities:**")
                    st.bar_chart({lbl: float(r_sev['scores'][i]) for i, lbl in enumerate(r_sev['classes'])})

            else:
                st.error("One or more models failed to execute.")
                if not results["ResNet50"]["success"]:
                    st.error(f"ResNet Error: {results['ResNet50'].get('error')}")
                if not results["Severity"]["success"]:
                    st.error(f"Severity Error: {results['Severity'].get('error')}")

else:
    # Empty State Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 5rem; color: #888;">
        <small>Powered by Dual-Engine Architecture | 2026 Medical Research</small>
    </div>
    """, unsafe_allow_html=True)
