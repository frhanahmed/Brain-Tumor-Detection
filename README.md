# 🧠 Brain Tumor Detection – Streamlit Version (v1)

An AI-powered web application that detects brain tumors from MRI images using a Convolutional Neural Network (CNN).

This project represents the **first version** of my Brain Tumor Detection system, built using Streamlit for rapid prototyping and experimentation.

---

## 🚀 Live Demo

🌐 Streamlit Deployment:  
https://brain-tumor-detection-cnn-app.streamlit.app/

> ⚠️ Note: The application may take some time to load due to Streamlit Cloud free-tier cold start behavior.

---

## 🚀 Project Overview

This application allows users to:

- Upload MRI images
- Automatically preprocess images
- Detect tumor presence using a trained CNN model
- View prediction results in real-time

This version was designed as a quick and interactive deep learning deployment using Streamlit.

---

## 🧠 Features

- MRI image upload (`.jpg`, `.jpeg`, `.png`)
- CLAHE-based contrast enhancement
- CNN-based binary classification:
  - 🚨 Tumor
  - ✅ No Tumor
- Interactive Streamlit UI
- Real-time prediction display

---

## 🛠 Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

---

## 📌 Model Details

- Convolutional Neural Network (CNN)
- Input Size: 128x128
- Binary Classification (Tumor vs No Tumor)
- Preprocessing:
  - CLAHE contrast enhancement
  - Normalization
  - Resizing

---

## ⚠ Deployment Limitation (Why v2 Was Built)

This project was deployed using **Streamlit Cloud (Free Tier)**.

However, during deployment, the following issues were observed:

- Application frequently went to sleep
- Cold start delays
- Memory limitations
- Unexpected runtime interruptions
- Limited scalability and backend control

Because of these constraints, I redesigned the system with a more scalable architecture.

---

## 🚀 Next Version (Production Upgrade)

To overcome Streamlit’s free-tier limitations, I rebuilt the entire system as a full-stack production-ready application:

👉 **NeuroScan AI – Full Stack Version**  
🔗 https://github.com/frhanahmed/NeuroScan-AI  

### Improvements in v2:

- Flask REST API backend
- Separate frontend (HTML + Tailwind CSS)
- PDF upload support
- Deployment on Render + Vercel
- Lazy model loading (memory optimized)
- TensorFlow CPU version for lightweight deployment
- Better scalability and architecture control

The new version follows production deployment best practices and resolves all free-tier performance issues encountered in this version.

---

## 📈 Learning Outcome

This project helped me:

- Understand CNN-based medical image classification
- Implement real-time AI inference in web apps
- Deploy ML models on cloud platforms
- Identify limitations of free-tier hosting
- Design better scalable architecture (v2)

---

## 👨‍💻 Author

**Farhan Ahmed**  

- LinkedIn: https://www.linkedin.com/in/farhanahmedf21  
- GitHub: https://github.com/frhanahmed  
- Portfolio: https://frhanahmed.github.io/Portfolio/

---

## ⭐ If You Like This Project

Give it a star on GitHub ⭐
