
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests
import datetime

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="MedTech AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed", # Collapsed to focus on the main UI
)

# --------------------------------------------------------------------------------
# Custom CSS - The "MedTech" Look
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f6f8; /* Light gray background */
        color: #1e293b;
    }

    /* Navbar Styling */
    .navbar {
        background-color: white;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border-radius: 12px;
    }
    .navbar-brand {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .navbar-menu {
        display: flex;
        gap: 2rem;
        font-weight: 600;
        color: #64748b;
    }
    .navbar-menu span {
        cursor: pointer;
        transition: color 0.2s;
    }
    .navbar-menu span:hover {
        color: #0f172a;
    }
    .navbar-menu span.active {
        color: #0f172a;
        border-bottom: 2px solid #0f172a;
        padding-bottom: 2px;
    }
    .user-profile {
        width: 40px;
        height: 40px;
        background-color: #e2e8f0;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #475569;
    }

    /* Main Container & Headings */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1e293b;
    }
    .sub-title {
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Card Styling */
    .white-card {
        background-color: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        height: 100%;
    }

    /* Image Container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        position: relative;
    }
    .image-overlay {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #10b981; /* Green */
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Results Typography */
    .result-section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #0f172a;
    }
    .prediction-row {
        margin-bottom: 1rem;
    }
    .prediction-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .prediction-value {
        color: #ef4444; /* Red for alert */
        font-size: 1.25rem;
        font-weight: 700;
    }
    .prediction-value.safe {
        color: #10b981; /* Green for safe */
    }

    /* Buttons */
    .stButton button {
        background-color: #10b981; /* Green */
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1rem;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton button:hover {
        background-color: #059669;
    }
    .stDownloadButton button {
         background-color: #fff !important;
         color: #10b981 !important;
         border: 2px solid #10b981 !important;
    }
    .stDownloadButton button:hover {
        background-color: #f0fdf4 !important;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Logic & Helpers
# --------------------------------------------------------------------------------
@st.cache_resource
def download_model(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

@st.cache_resource
def load_prediction_model(model_path, model_url=None):
    if not os.path.exists(model_path):
        if model_url:
            with st.spinner(f"Setting up AI Engine..."):
                download_model(model_url, model_path)
        else:
             return None 
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except:
        return None

def process_image(image_data, target_size=(256, 256)):
    image = ImageOps.fit(image_data, target_size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    return np.expand_dims(np.array(image), axis=0)

# Models Config
AVAILABLE_MODELS = {
    "Advanced Severity Model": {
        "path": "Lung_Cancer_Severity_Model.h5", 
        "url": "https://drive.google.com/uc?export=download&id=1_6p87Sb-uKuHVLivvgYLZGbGTAvQwxwr",
    },
    "ResNet50 (Deep Scan)": {
        "path": "chest-ct-scan-ResNet50.keras", 
        "url": "https://drive.google.com/uc?export=download&id=1pCpBhr3TMddm9xuEj7M7WXml5M6KkKP4",
    },
    "Standard Model": {
        "path": "model_notebook.h5", 
        "url": None,
    }
}

# --------------------------------------------------------------------------------
# Navbar
# --------------------------------------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span>🏥</span> MedTech
    </div>
    <div class="navbar-menu">
        <span>Home</span>
        <span>Upload</span>
        <span>Analysis</span>
        <span class="active">Results</span>
        <span>Hospitals</span>
    </div>
    <div class="user-profile">
        👤
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Main Content
# --------------------------------------------------------------------------------

st.markdown('<div class="main-title">Analysis Results</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered lung cancer detection outcome</div>', unsafe_allow_html=True)

# Layout: Two columns (Image | Results)
col_img, col_res = st.columns([1.2, 1])

# File Uploader (Hidden in "Upload New Scan" logical flow, but placed here for functionality)
# We style it to look minimal or just use standard Streamlit uploader at top if needed.
# For this UI, we'll assume the user is on the "Results" page after upload. 
# To make it functional, we need the uploader to be accessible.

uploaded_file = st.sidebar.file_uploader("Upload Scan", type=["jpg", "png"]) # Moving uploader to sidebar to keep UI clean, or...

# Actually, let's put the uploader in the main flow if no file is present
if uploaded_file is None:
    st.info("👆 Please upload a CT scan using the sidebar to see the analysis results.")
    
    # Placeholder for visual design
    with col_img:
        st.markdown("""
        <div class="white-card" style="display:flex; align-items:center; justify-content:center; min-height:400px; color:#cbd5e1;">
            Preview will appear here
        </div>
        """, unsafe_allow_html=True)
    with col_res:
         st.markdown("""
        <div class="white-card">
            <div class="result-section-title">Diagnosis Result</div>
             <p style="color:#94a3b8;">Waiting for scan...</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # -- IMAGE COLUMN --
    with col_img:
        # Show Image in a "Card"
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True, caption="")
        st.markdown("""
            <div style="text-align:right; margin-top:10px;">
                <span style="background:#ecfdf5; color:#059669; padding:5px 10px; border-radius:15px; font-size:0.8rem; font-weight:600;">
                    ✅ Analysis Completed
                </span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -- RESULTS COLUMN --
    with col_res:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-section-title">Diagnosis Result</div>', unsafe_allow_html=True)
        
        # Select Model (Hidden in sidebar or integrated?)
        # For simplicity, default to Advanced, user can change in sidebar
        selected_model = st.sidebar.selectbox("AI Model", list(AVAILABLE_MODELS.keys()))
        model_info = AVAILABLE_MODELS[selected_model]
        
        with st.spinner("Analyzing..."):
            model = load_prediction_model(model_info['path'], model_info['url'])
            
            if model:
                # Prediction Logic
                try:
                    img = Image.open(uploaded_file)
                    target_size = (224, 224) if "ResNet" in selected_model else (256, 256)
                    processed_img = process_image(img, target_size)
                    preds = model.predict(processed_img)
                    
                    scores = tf.nn.softmax(preds[0])
                    class_labels = ['Benign', 'Malignant (Squamous Cell Carcinoma)', 'Normal']
                    
                    # Logic to pick specific detailed names if available, simplifying here
                    idx = np.argmax(scores)
                    result_label = class_labels[idx]
                    confidence = 100 * np.max(scores)
                    
                    # Determine styling based on result
                    is_safe = "Normal" in result_label
                    color_class = "safe" if is_safe else ""
                    
                    # HTML Result
                    st.markdown(f"""
                    <div class="prediction-row">
                        <div class="prediction-label">PREDICTED TYPE:</div>
                        <div class="prediction-value {color_class}">{result_label}</div>
                        <div style="font-size:0.9rem; color:#64748b; margin-top:0.2rem;">Confidence: {confidence:.2f}%</div>
                    </div>
                    
                    <div class="prediction-row">
                        <div class="prediction-label">SEVERITY LEVEL:</div>
                        <div class="prediction-value {color_class}">
                             {'Normal' if is_safe else 'High / Malignant'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button("Download Report", "Report Content...", file_name="report.txt")
                    with col_btn2:
                        # This button is just visual as uploader handles state
                        st.button("Scan New Image") 
                        
                except Exception as e:
                    st.error("Error analyzing image")
            else:
                 st.error("Model not loaded")

        st.markdown('</div>', unsafe_allow_html=True)

