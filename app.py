import streamlit as st
import numpy as np
import cv2
import time
import os
import fitz  # PyMuPDF
from PIL import Image
import tensorflow as tf

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠")

IMG_SIZE = 128

# =============================
# LAZY LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "brain-tumor-model.keras")
    return tf.keras.models.load_model(MODEL_PATH)

# =============================
# PDF TO IMAGE USING PyMuPDF
# =============================
def convert_pdf_to_image(file_bytes):
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    page = pdf.load_page(0)
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

# =============================
# IMAGE PREPROCESSING
# =============================
def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return np.expand_dims(img, axis=0)

# =============================
# SIDEBAR (UNCHANGED)
# =============================
with st.sidebar:
    try:
        image = Image.open("MyPhoto.jpg")
        st.image(image, width=150)
    except:
        st.warning("Profile image not found.")

    st.markdown("<h3 style='text-align: center;'>Farhan Ahmed</h3>", unsafe_allow_html=True)

    st.markdown("### 🤝 Connect With Me")
    st.markdown("""
    - 📧 [frhanahmedf21@gmail.com](mailto:frhanahmedf21@gmail.com)
    - 💼 [LinkedIn](https://linkedin.com/in/farhanahmedf21)
    - 💻 [GitHub](https://github.com/frhanahmed)
    - 💬 [WhatsApp](https://wa.me/918910080891)
    """)

    st.markdown("### 🗂️ Source Code")
    st.markdown("[🔗 GitHub Repository](https://github.com/frhanahmed/Brain-Tumor-Detection.git)")

# =============================
# MAIN UI
# =============================
st.title("🧠 MRI-Based Brain Tumor Detection Tool")

uploaded_file = st.file_uploader(
    "Upload an MRI image or PDF:",
    type=["jpg", "jpeg", "png", "pdf"]
)

image = None

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    file_bytes = uploaded_file.read()

    if uploaded_file.type == "application/pdf":
        image = convert_pdf_to_image(file_bytes)
    else:
        image = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             caption="Preview of Uploaded MRI",
             width=250)

# =============================
# PREDICTION
# =============================
if uploaded_file and st.button("🔍 Predict"):
    with st.spinner("Loading model & analyzing..."):
        model = load_model()   # Lazy load here
        processed = preprocess_image(image)
        prediction = model.predict(processed)

        confidence = float(np.max(prediction))
        result = "🚨 Tumor Detected" if np.argmax(prediction) == 1 else "✅ No Tumor"

    st.subheader("Prediction Result:")
    if "No Tumor" in result:
        st.success(f"{result} ({confidence*100:.2f}% confidence)")
    else:
        st.error(f"{result} ({confidence*100:.2f}% confidence)")

# =============================
# CONTACT SECTION (UNCHANGED)
# =============================
st.write("Feel free to send me a message using the form below!")

with st.expander("📬 Contact Me"):
    contact_form = """
        <form action="https://formsubmit.co/frhanahmedf21@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required style="width: 100%; padding: 8px;border-radius: 5px;background-color: azure;color: black;"><br><br>
        <input type="email" name="email" placeholder="Your Email" required style="width: 100%; padding: 8px;border-radius: 5px;background-color: azure;color: black;"><br><br>
        <textarea name="message" placeholder="Your message here..." rows="5" required style="width: 100%; padding: 8px;border-radius: 5px;background-color: azure;color: black;"></textarea><br><br>
        <div style="text-align: center;">
        <button type="submit" 
            style="padding: 10px 20px; border-radius: 5px; background-color: rgb(149, 68, 224); color: white;margin-bottom: 5px;">
            Send Message
        </button>
    </div>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)