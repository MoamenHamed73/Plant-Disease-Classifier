# 🌿 Plant Disease Classification using Deep Learning (CNN)

## 📌 Overview
Plant diseases can significantly impact agricultural productivity. Early and accurate detection is crucial to reduce crop loss.

This project builds a **Deep Learning model using CNN (MobileNetV2)** to classify plant leaf images into different disease categories.

---

## 🎯 Problem Statement
Farmers often rely on manual inspection to detect plant diseases, which is:
- Time-consuming  
- Error-prone  
- Not scalable  

Additionally, real-world datasets introduce challenges such as:
- Class imbalance  
- Visual similarity between diseases  
- Noisy and inconsistent images  

---

## 💡 Solution
A robust image classification system using:
- **Transfer Learning (MobileNetV2)**
- **Data Augmentation**
- **Class Weights for imbalance handling**
- **Fine-tuning for better performance**

---

## 🧠 Model Architecture
- Base Model: **MobileNetV2 (Pre-trained on ImageNet)**
- Added Layers:
  - GlobalAveragePooling
  - Dense (ReLU)
  - Dropout
  - Output Layer (4 classes - Softmax)

---

## 📊 Dataset
- Source: Plant Pathology Dataset (Kaggle)
- Classes:
  - Healthy 🌱  
  - Rust 🍂  
  - Scab 🍁  
  - Multiple Diseases 🦠  

---

## 🔄 Data Preprocessing
- Image resizing: **224 × 224**
- Normalization (rescale = 1./255)

### 🔁 Data Augmentation
- Rotation  
- Zoom  
- Shear  
- Width & Height Shift  
- Horizontal Flip  
- Brightness Adjustment  

---

## ⚖️ Handling Imbalanced Data
- Applied **Class Weights** to improve minority class performance  
- Improved detection of `multiple_diseases` class significantly  

---

## 🏋️ Training Strategy
- Optimizer: Adam  
- Learning Rate Scheduling  
- Fine-tuning last layers of MobileNetV2  
- Validation split for monitoring performance  

---

## 📈 Results

| Metric | Value |
|------|------|
| Train Accuracy | ~90% |
| Validation Accuracy | **91% 🔥** |

### 📊 Classification Performance
- Strong performance across major classes  
- Significant improvement in minority class detection  

---

## 📉 Confusion Matrix & Evaluation
- Used:
  - Classification Report  
  - Confusion Matrix  
- Evaluated precision, recall, and F1-score  

---

## 💻 Deployment
The model is deployed using **Streamlit**.

### Features:
- Upload leaf image  
- Get predicted class  
- View prediction probabilities  

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/plant-disease-classifier.git
cd plant-disease-classifier
pip install -r requirements.txt
streamlit run app.py

### Author:Moamen hamed
