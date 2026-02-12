
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --------------------------------------------------------------------------------
# Configuration and Styling
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Lung Cancer Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2 {
        color: #333;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        color: white;
        background-color: #1f77b4;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
    }
    .stFileUploader>div>div>button {
        color: #1f77b4;
        border: 2px solid #1f77b4;
        border-radius: 8px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------
@st.cache_resource
def download_model(url, save_path):
    """Downloads a file from a URL to the specified path."""
    import requests
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

@st.cache_resource
def load_prediction_model(model_name, model_info):
    """Loads a Keras model, downloading it first if necessary."""
    model_path = model_info["path"]
    model_url = model_info.get("url") # URL is optional if file exists locallly

    if not os.path.exists(model_path):
        if model_url:
            with st.spinner(f"Downloading {model_name} (this may take a while)..."):
                success = download_model(model_url, model_path)
                if not success:
                    st.error(f"Failed to download model from {model_url}")
                    return None
        else:
            st.error(f"Model file {model_path} not found and no URL provided.")
            return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(image_data, target_size=(256, 256)):
    """
    Preprocesses the image for the model.
    - Resizes to target_size
    - Converts to RGB
    - Normalizes pixel values (if required by model, usually handled in model layers for recent Keras models)
    - Expands dimensions to match batch shape (1, H, W, C)
    """
    image = ImageOps.fit(image_data, target_size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0) # (1, 256, 256, 3)
    return img_array

# --------------------------------------------------------------------------------
# Sidebar - Configuration
# --------------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=100)
st.sidebar.title("Configuration")

# Define available models
# In a real deployment, these paths should be relative to the app.py location
# Assuming models are in the same directory or a 'models' subdirectory
# Define available models
# For files > 100MB (GitHub limit), you MUST provide a direct download URL (e.g., from Dropbox, Google Drive, or Hugging Face).
# If the file is local (smaller than 100MB), you can leave "url" as None.
AVAILABLE_MODELS = {
    "Notebook Model (Custom CNN)": {
        "path": "model_notebook.h5", 
        "url": None  # Assuming this is small enough (<100MB)
    },
    "Severity Model": {
        "path": "Lung_Cancer_Severity_Model.h5", 
        "url": "https://drive.google.com/uc?export=download&id=1_6p87Sb-uKuHVLivvgYLZGbGTAvQwxwr" # > 100MB
    },
    "ResNet50 CT Scan": {
        "path": "chest-ct-scan-ResNet50.keras", 
        "url": "https://drive.google.com/uc?export=download&id=1pCpBhr3TMddm9xuEj7M7WXml5M6KkKP4" # > 100MB
    },
    "Lung Model (General)": {
        "path": "Lung_Model.h5", 
        "url": None # Assuming <100MB, otherwise add link
    },
    "SMOTE Prediction Model": {
        "path": "lung_cancer_prediction_model_smote.h5", 
        "url": None
    }
}

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(AVAILABLE_MODELS.keys()),
    index=0
)

# Confidence Threshold
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application uses Deep Learning models to analyze chest CT scans and classify them into categories like Benign, Malignant, or Normal."
)

# --------------------------------------------------------------------------------
# Main Content
# --------------------------------------------------------------------------------
st.title("🫁 Lung Cancer Classification AI")
st.markdown("### Upload a Chest CT Scan for Analysis")

# File Uploader
uploaded_file = st.file_uploader("Choose a CT Scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Scan", use_column_width=True)
    
    with col2:
        st.write("### Analysis Results")
        
        # Load Model
        # Look up the selected model's info
        model_info = AVAILABLE_MODELS[selected_model_name]
        
        load_start = st.empty() # Placeholder for loading message
        
        model = load_prediction_model(selected_model_name, model_info)
            
        if model is None:
            st.error("Could not load model.")
        else:
            # Process Image
            image = Image.open(uploaded_file)
            
            # Determine target size based on model (heuristic or specific logic)
            # Default to 256x256 as used in the notebook
            target_size = (256, 256) 
            if "ResNet50" in selected_model_name:
                 target_size = (224, 224) # ResNet usually expects 224

            processed_image = process_image(image, target_size=target_size)
            
            # Predict
            try:
                predictions = model.predict(processed_image)
                
                # Class Names - logic depends on the specific model trained
                # Defaulting to the ones from the notebook analysis
                class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']
                
                # Handling different output shapes (e.g. binary vs multi-class)
                if predictions.shape[1] == 3:
                     # Softmax output
                    score = tf.nn.softmax(predictions[0])
                    predicted_class = class_names[np.argmax(score)]
                    confidence = 100 * np.max(score)
                else:
                    # Fallback for binary or unknown output shapes
                    st.warning(f"Model output shape {predictions.shape} is different than expected (3 classes). Showing raw output.")
                    predicted_class = "Unknown"
                    confidence = 0.0

                # Display Result
                if confidence/100 >= confidence_threshold:
                    color = "#28a745" if "Normal" in predicted_class else "#dc3545"
                    st.markdown(f"""
                        <div class="prediction-box" style="background-color: {color}20; border: 2px solid {color}; color: {color};">
                            Prediction: {predicted_class}<br>
                            Confidence: {confidence:.2f}%
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability Bar Chart
                    st.write("#### Class Probabilities:")
                    if predictions.shape[1] == 3:
                        st.bar_chart(dict(zip(class_names, predictions[0])))
                        
                else:
                    st.warning(f"Prediction confidence ({confidence:.2f}%) is below the threshold ({confidence_threshold*100}%). Result is uncertain.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.write("Debug info: input shape", processed_image.shape)

else:
    st.info("Please upload an image to start.")

# --------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Developed for Medical AI Research | Not for Clinical Diagnosis</small>
    </div>
    """,
    unsafe_allow_html=True
)
