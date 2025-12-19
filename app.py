import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SIBI Classifier",
    page_icon="🤟",
    layout="centered"
)

# --- 2. LOAD MODEL (Hanya sekali agar cepat) ---
@st.cache_resource
def load_model():
    # Ganti nama file sesuai model terbaik Anda
    model = tf.keras.models.load_model('model_mobilenet.keras')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error: Model tidak ditemukan. Pastikan file 'model_mobilenet.keras' ada di folder ini. ({e})")

# --- 3. DAFTAR KELAS (A-Z) ---
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# --- 4. TAMPILAN UI ---
st.title("🤟 Deteksi Bahasa Isyarat (SIBI)")
st.write("""
Aplikasi ini menggunakan **MobileNetV2** untuk mengenali alfabet SIBI.
Silakan unggah foto tangan (latar belakang polos lebih baik).
""")

file = st.file_uploader("Upload Foto Tangan", type=["jpg", "png", "jpeg"])

# --- 5. LOGIKA PREDIKSI ---
if file is not None:
    # Tampilkan gambar user
    image = Image.open(file)
    st.image(image, caption="Foto yang diunggah", use_container_width=True)
    
    # Tombol Prediksi
    if st.button("Deteksi Huruf"):
        with st.spinner('Sedang menganalisis...'):
            # a. Preprocessing (Samakan dengan training!)
            # Resize ke 224x224
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            
            # Ubah ke Array & Normalisasi (0-1)
            img_array = np.asarray(image)
            normalized_image_array = (img_array.astype(np.float32) / 255.0)
            
            # Tambah dimensi batch: (1, 224, 224, 3)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # b. Prediksi
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

# ... (kode sebelumnya tetap sama) ...

            st.success(f"Prediksi: **Huruf {class_name}**")
            st.info(f"Tingkat Keyakinan: **{confidence_score * 100:.2f}%**")

            # --- BAGIAN INI YANG DIUBAH AGAR MUNCUL HURUF ---
            st.write("---")
            st.write("Detail Probabilitas:")
            
            # Kita bungkus data prediksi ke dalam DataFrame Pandas
            # index=class_names --> Ini yang mengubah angka 0,1,2 jadi A,B,C
            df_prob = pd.DataFrame(prediction[0], index=class_names, columns=['Probability'])
            
            # Tampilkan chart menggunakan DataFrame tersebut
            st.bar_chart(df_prob)