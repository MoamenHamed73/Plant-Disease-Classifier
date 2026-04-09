import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# تحميل الموديل
model_path = os.path.join(os.path.dirname(__file__), "plant_model_final.h5")
model = load_model(model_path)

class_names = ['healthy','multiple_diseases','rust','scab']

# إعداد الصفحة
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# 🔥 CSS لتجميل الشكل
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2e7d32;
    }
    .sub {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f5e9;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #1b5e20;
    }
    </style>
""", unsafe_allow_html=True)

# 🎯 Title
st.markdown('<div class="title">🌿 Plant Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload a leaf image and get instant prediction</div>', unsafe_allow_html=True)

# 📂 Upload
uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # 📷 عرض الصورة
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # 🔍 Prediction
    with col2:
        img = img.resize((224,224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)

        # 🎯 النتيجة
        st.markdown(
            f'<div class="result-box">Prediction: {class_names[pred_class]}<br>Confidence: {confidence:.2f}</div>',
            unsafe_allow_html=True
        )

        st.write("")

        # 📊 Probabilities
        st.subheader("📊 Class Probabilities")

        for i, cls in enumerate(class_names):
            st.write(f"**{cls}**")
            st.progress(float(pred[0][i]))

# Footer
st.markdown("---")
st.markdown("Made by by Moamen hamed  AI Engineer")