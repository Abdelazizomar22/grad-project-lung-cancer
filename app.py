
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

def predict_sequentially(uploaded_file, model_configs):
    """Loads one model, predicts, clears memory, loads next."""
    results = {}
    classes = ['Benign', 'Malignant', 'Normal']
    
    img = Image.open(uploaded_file)

    # 1. ResNet50
    st.toast("Running Visual Detection Engine...", icon="🔍")
    tf.keras.backend.clear_session()
    gc.collect()
    
    try:
        model = tf.keras.models.load_model(model_configs["ResNet50"]["path"], compile=False)
        input_data = process_image(img, (224, 224))
        pred = model.predict(input_data)
        score = tf.nn.softmax(pred[0])
        idx = np.argmax(score)
        results["ResNet50"] = {
            "label": classes[idx],
            "conf": 100 * np.max(score),
            "scores": score,
            "success": True
        }
        del model
    except Exception as e:
         results["ResNet50"] = {"success": False, "error": str(e)}

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()

    # 2. Severity
    st.toast("Running Severity Analysis Engine...", icon="⚖️")
    
    try:
        model = tf.keras.models.load_model(model_configs["Severity"]["path"], compile=False)
        input_data = process_image(img, (256, 256))
        pred = model.predict(input_data)
        score = tf.nn.softmax(pred[0])
        idx = np.argmax(score)
        results["Severity"] = {
            "label": classes[idx],
            "conf": 100 * np.max(score),
            "scores": score,
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

# 2. Upload Section (Show only if no file is uploaded or if we are not in result mode)
if 'result_mode' not in st.session_state:
    st.session_state.result_mode = False

if not st.session_state.result_mode:
    col_spacer1, col_up, col_spacer2 = st.columns([1, 2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload Chest CT Scan for Analysis", type=["jpg", "png", "jpeg"])
        
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.result_mode = True
        st.rerun()

# 3. Analysis & Results Section
if st.session_state.result_mode and 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state.uploaded_file
    
    # We use a container for the white card effect
    with st.container():
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # New 2-Column Layout
        col_img, col_info = st.columns([1.2, 1.5], gap="large")
        
        with col_img:
            st.markdown('<div class="status-badge">✅ Analysis Completed</div>', unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True, channels="RGB")
            st.markdown('<p style="text-align:center; color:#888; font-size:0.8rem; margin-top:5px;">Input Scan Source</p>', unsafe_allow_html=True)

        with col_info:
            configs = check_and_download_models()
            
            # Run or retrieve results (simple caching mechanism could be added here, but we run for demo)
            if 'prediction_results' not in st.session_state:
                 with st.spinner("🧠 Processing Scan..."):
                    st.session_state.prediction_results = predict_sequentially(uploaded_file, configs)
            
            results = st.session_state.prediction_results
            
            if results["ResNet50"]["success"] and results["Severity"]["success"]:
                r_res = results["ResNet50"]
                r_sev = results["Severity"]
                
                # Determine colors based on malignancy
                is_malignant_res = "Malignant" in r_res["label"]
                is_malignant_sev = "Malignant" in r_sev["label"]
                
                res_color_class = "diagnosis-value-red" if is_malignant_res else "diagnosis-value-green"
                sev_color_class = "diagnosis-value-red" if is_malignant_sev else "diagnosis-value-green"

                # -- Title --
                st.markdown('<div class="diagnosis-header">Diagnosis Result</div>', unsafe_allow_html=True)
                
                # -- Prediction 1 --
                st.markdown('<div class="diagnosis-label">PREDICTED TYPE:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{res_color_class}">{r_res["label"]} ({r_res["conf"]:.2f}%)</div>', unsafe_allow_html=True)
                
                # -- Prediction 2 --
                st.markdown('<div class="diagnosis-label">SEVERITY LEVEL:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="{sev_color_class}">{r_sev["label"]} ({r_sev["conf"]:.2f}%)</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # -- Action Buttons --
                # 1. Download Report (Placeholder)
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f"Report for {r_res['label']}, {r_sev['label']}",
                    file_name="lung_scan_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                    type="primary" # Makes it green usually in Streamlit default theme or custom theme
                )
                
                # 2. Upload New (Reset)
                if st.button("⬆️ Upload New Scan", use_container_width=True):
                    # Clear state to reset
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
                    
            else:
                st.error("Analysis Failed")
                st.write(results)

        st.markdown('</div>', unsafe_allow_html=True) # End results-container

else:
    # Empty State Footer (only visible when not in result mode)
    st.markdown("""
    <div style="text-align: center; margin-top: 5rem; color: #888;">
        <small>Powered by Dual-Engine Architecture | 2026 Medical Research</small>
    </div>
    """, unsafe_allow_html=True)
