import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Constants
IMAGE_SIZE = (350, 350)
NORMALIZATION = 1.0 / 255.0
TEST_DIR = "./test_merged"  # Add your test directory path here

# Class names (based on your dataset's structure)
CLASS_NAMES = {
    0: "Adenocarcinoma",
    1: "Squamous Cell Carcinoma",
    2: "Large Cell Carcinoma",
    3: "Small Cell Lung Cancer (SCLC)"
}

# Streamlit App Configuration
st.set_page_config(page_title="Luca AI", page_icon="🫁", layout="wide")
st.sidebar.image("x.png", width=400, use_container_width=True)
st.sidebar.title("Luca AI 🩺")
st.sidebar.markdown(
    """**Luca AI** is a cutting-edge medical image classification tool powered by advanced deep learning models."""
)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate the App", ["Overview", "Predictions", "FAQ"])

st.sidebar.title("About")
st.sidebar.markdown(
    """ 
    This app is designed to:<br>
    🩺 **Accurate Predictions**: Classify lung cancer types with state-of-the-art models.<br>
    🚀 **User-Friendly Design**: Ideal for research, education, and experimentation.
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.write("**Created with 💜 by HITANANDSINGH JAUNKY**")

def load_all_models():
    model_paths = {
        'VGG16': 'best_vgg16_model.keras',
        'InceptionV3': 'best_inceptionv3_model.keras',
        'EfficientNetB0': 'best_efficientnetb0_model.keras',
        'ResNet50': 'best_resnet50_model.keras',
        'DenseNet121': 'best_densenet121_model.keras',
        'Simple CNN': 'simple_cnn_model.keras',
        'Custom CNN': 'custom_cnn_model.keras',
        'Ensemble ALL': 'ensemble_all_model.keras',
        'Ensemble 1': 'ensemble_best_model.keras'
    }
    models = {}
    accuracies = {}

    # Prepare the test dataset
    if os.path.exists(TEST_DIR):
        test_gen = ImageDataGenerator(rescale=NORMALIZATION).flow_from_directory(
            TEST_DIR,
            target_size=IMAGE_SIZE,
            batch_size=32,
            class_mode="categorical",
            shuffle=False
        )

        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    model = load_model(path)
                    models[name] = model

                    # Evaluate the model on the test dataset
                    _, accuracy = model.evaluate(test_gen, verbose=0)
                    accuracies[name] = accuracy * 100
                except Exception as e:
                    st.warning(f"Could not load {name}: {e}")
            else:
                st.warning(f"Model file for {name} is missing at {path}")
    else:
        st.error("Test directory is missing. Please ensure `test_merged` exists.")

    return models, accuracies


models, model_accuracies = load_all_models()

if page == "Overview":
    st.title("Welcome to Luca AI 🤖")
    st.markdown(
        """
        ### Features:
        - **Cutting-edge Models 🧠**: Advanced AI models for classification.
        - **User-friendly Design ⚙️**: A seamless interface for researchers and educators.

        ---
        ### Types of Lung Cancer 🫁
        1. **Adenocarcinoma 🌿**  
        2. **Squamous Cell Carcinoma 🚬**  
        3. **Large Cell Carcinoma 🌪️**  
        4. **Small Cell Lung Cancer (SCLC) ⚡**

        ---
        🩺 **Key Takeaway:** Early detection saves lives! 💪
        """
    )

elif page == "Predictions":
    st.title("Predictions 🔬")
    uploaded_file = st.file_uploader("Upload an image for classification 🖼️", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Open and preprocess the image
        img = Image.open(uploaded_file).convert("RGB").resize(IMAGE_SIZE)  # Ensure the image is in RGB format
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize and add batch dimension

        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Loop through all models and generate predictions
        for name, model in models.items():
            # Generate predictions
            pred = model.predict(img_array)  # Predict using the model
            predicted_index = np.argmax(pred[0])  # Get the class with the highest probability
            predicted_class = CLASS_NAMES[predicted_index]  # Map index to class name

            # Get the model's actual accuracy
            model_accuracy = model_accuracies.get(name, "N/A")

            # Display the prediction and model accuracy
            st.write(
                f"**{name} Prediction 🧬:**\n"
                f"- **Predicted Class:** {predicted_class}\n"
                f"- **Model Accuracy:** {model_accuracy:.2f}%"
            )

        st.warning("⚠️ This is an AI-based tool and is not a substitute for professional medical advice. Please consult a doctor for a proper diagnosis.")

elif page == "FAQ":
    st.title("FAQ ❓")
    st.markdown(
        """
        **Q: What does Luca AI do?**  
        A: Luca AI classifies medical images, specifically lung cancer types, using deep learning models.

        **Q: Who can use this tool?**  
        A: Researchers, educators, and anyone interested in medical AI applications.

        **Q: How can I contribute?**  
        A: Feel free to reach out or submit issues on the project repository!
        """
    )
