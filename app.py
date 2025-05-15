import streamlit as st
import tensorflow as tf  # type: ignore
import numpy as np
from PIL import Image
import os
import gdown  # type: ignore

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Penyakit Kulit", layout="centered")
st.title("🩺 Prediksi Penyakit Kulit dari Gambar")
st.write("Silakan upload gambar kulit untuk diprediksi oleh model klasifikasi.")

@st.cache_resource
def load_model():
    model_path = "model86.keras"
    gdrive_url = "https://drive.google.com/uc?id=11Onil-lL7fdjtx3Ncvs7PRfFNmfFvIDs"

    if not os.path.exists(model_path):
        with st.spinner("📥 Mengunduh model dari Google Drive..."):
            try:
                output = gdown.download(gdrive_url, model_path, quiet=False)
                if output is None:
                    st.error("❌ Gagal mengunduh model. Periksa URL atau izin file Google Drive.")
                    st.stop()
                st.success("✅ Model berhasil diunduh.")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat mengunduh: {e}")
                st.stop()

    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Label kelas (6 kelas)
class_names = ['Enfeksiyonel', 'Ekzama', 'Akne', 'Pigment', 'Benign', 'Malign']

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload Gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼️ Gambar yang Diunggah', use_column_width=True)

    if st.button("🔍 Prediksi"):
        # Preprocessing gambar
        image_resized = image.resize((150, 150))  # Sesuaikan dengan input model
        img_array = tf.keras.utils.img_to_array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi kelas
        prediction = model.predict(img_array)
        pred_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        # Tampilkan hasil prediksi
        st.success(f"✅ Prediksi: **{pred_class}**")
        st.info(f"Tingkat kepercayaan: **{confidence:.2f}%**")
