import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- KONFIGURASI ---
BASE_DIR = r"D:\Akmal\Bahasa Isyarat\Dataset_SIBI_Final"
MODEL_PATH = 'model_mobilenet.keras'
IMG_SIZE = (224, 224)

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)

# Load Val Data (Ambil 1 batch saja untuk sampel)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Val'),
    target_size=IMG_SIZE,
    batch_size=25, # Ambil 25 gambar
    class_mode='categorical',
    shuffle=True   # Acak biar dapat macam-macam huruf
)

# Ambil gambar dan label
images, labels = next(val_generator)
preds = model.predict(images)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(labels, axis=1)
class_names = list(val_generator.class_indices.keys())

# --- PLOT GRID 5x5 ---
plt.figure(figsize=(15, 15))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i])
    
    # Tentukan warna (Hijau=Benar, Merah=Salah)
    pred_name = class_names[pred_labels[i]]
    true_name = class_names[true_labels[i]]
    
    color = 'green' if pred_name == true_name else 'red'
    
    plt.title(f"Asli: {true_name}\nPred: {pred_name}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('visualisasi_prediksi.png')
plt.show()
print("Gambar 'visualisasi_prediksi.png' berhasil disimpan! Masukkan ini ke Poster/Paper.")