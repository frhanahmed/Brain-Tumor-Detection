import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
from pdf2image import convert_from_bytes
import tensorflow as tf

# Set Page Configuration
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠")

# Load trained model
model = tf.keras.models.load_model('brain-tumor-model.keras')
img_size = 128  # must match model input size

# Sidebar
with st.sidebar:
    # Profile Image
    image = Image.open(r"C:\Users\frhan\Desktop\Photo2-photoaidcom-cropped.jpg")
    st.image(image, width=150)
    
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

# App Header
st.title("🧠 MRI-Based Brain Tumor Detection Tool")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image or PDF:", type=["jpg", "jpeg", "png", "pdf"]) 

image = None  # Placeholder

# Handle file upload
try:
    if uploaded_file:
        st.success("✅ File uploaded successfully!")

        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read())
            image = np.array(images[0])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = Image.open(uploaded_file)
            image = np.array(image.convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Preview of Uploaded MRI",width=200)
    # else:
    #     st.warning("⚠️ File not uploaded yet.")
except:
    st.warning("⚠️ File not uploaded!!!")

# Image preprocessing
def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, (img_size, img_size)) / 255.0
    return np.expand_dims(img, axis=0)

# Predict button
if uploaded_file and st.button("🔍 Predict"):
    with st.spinner("Processing..."):
        time.sleep(2)  # simulate loading
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        result = "🚨 Tumor Detected" if np.argmax(prediction) == 1 else "✅ No Tumor"
    
    st.subheader("Prediction Result:")
    if "No Tumor" in result:
        st.success(result)
    else:
        st.error(result)

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