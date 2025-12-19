import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ==========================================
# 1. KONFIGURASI
# ==========================================
# Ganti path ini sesuai lokasi dataset Anda
BASE_DIR = r"D:\Akmal\Bahasa Isyarat\Dataset_SIBI_Final"
MODEL_PATH = 'model_mobilenet.keras'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# 2. MUAT DATA VALIDASI (DATA UJI)
# ==========================================
print("Sedang memuat data validasi...")
# PENTING: shuffle=False agar urutan prediksi cocok dengan label asli
val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

# ==========================================
# 3. JALANKAN PREDIKSI
# ==========================================
print("Memuat model dan melakukan prediksi...")
model = tf.keras.models.load_model(MODEL_PATH)

# Prediksi probabilitas
Y_pred = model.predict(validation_generator, verbose=1)
# Ambil kelas dengan probabilitas tertinggi (0-25)
y_pred = np.argmax(Y_pred, axis=1)
# Ambil label asli dari folder
y_true = validation_generator.classes
# Ambil nama kelas ('A', 'B', ...)
class_labels = list(validation_generator.class_indices.keys())

# ==========================================
# 4. BUAT TABEL CLASSIFICATION REPORT (SEPERTI JURNAL)
# ==========================================
print("\nMenghitung Precision, Recall, dan F1-Score...")

# Membuat dictionary laporan
report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)

# Mengubah ke DataFrame Pandas agar rapi seperti Tabel Jurnal
df_report = pd.DataFrame(report_dict).transpose()

# Filter: Kita ambil kolom precision, recall, f1-score, support
# Ini persis seperti Gambar 5 di jurnal rujukan
df_report = df_report[['precision', 'recall', 'f1-score', 'support']]

# Format angka agar cuma 4 desimal (biar rapi)
df_report = df_report.round(4)

# Tampilkan di Terminal
print("\n--- HASIL EVALUASI ---")
print(df_report)

# SIMPAN KE CSV (Bisa dibuka di Excel untuk Copy-Paste ke Word)
df_report.to_csv('tabel_evaluasi_jurnal.csv')
print("\n[INFO] Tabel berhasil disimpan sebagai 'tabel_evaluasi_jurnal.csv'.")
print("Silakan buka file CSV tersebut di Excel untuk ditempel ke Paper.")

# ==========================================
# 5. BUAT CONFUSION MATRIX (GAMBAR 4 DI JURNAL)
# ==========================================
print("\nMembuat Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 12)) # Ukuran besar agar jelas
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)

plt.title('Confusion Matrix (Evaluasi Data Validasi)')
plt.ylabel('True Label (Label Asli)')
plt.xlabel('Predicted Label (Prediksi Model)')
plt.tight_layout()

# Simpan Gambar
plt.savefig('confusion_matrix_jurnal.png', dpi=300)
print("[INFO] Gambar 'confusion_matrix_jurnal.png' berhasil disimpan.")