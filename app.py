import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("plant_disease_model.h5")

# Define class labels (must match your model output)
class_labels = [
    "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Healthy Apple",
    "Corn Gray Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Healthy Corn",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Healthy Grape",
    "Potato Early Blight", "Potato Late Blight", "Healthy Potato",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", 
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Healthy Tomato"
]

st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # or the original training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    st.write("Raw prediction:", prediction)

    pred_index = int(np.argmax(prediction))

    if pred_index < len(class_labels):
        predicted_class = class_labels[pred_index]
        st.success(f"🌱 Predicted Disease: **{predicted_class}**")
    else:
        st.error("⚠️ Prediction index out of range. Check class label count.")
else:
    st.info("📷 Please upload a plant leaf image to predict the disease.")
