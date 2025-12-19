import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- KONFIGURASI PATH ---
BASE_DIR = r"D:\Akmal\Bahasa Isyarat\Dataset_SIBI_Final"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Transfer learning biasanya cepat konvergen

# --- DATA GENERATOR (Sama seperti sebelumnya) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Sedikit lebih agresif
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'Val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- ARSITEKTUR TRANSFER LEARNING (MobileNetV2) ---
def build_mobilenet_model(num_classes):
    # 1. Download "Otak" MobileNet tanpa bagian kepalanya (include_top=False)
    #    Weights='imagenet' artinya kita pakai pengetahuan yang sudah ada.
    base_model = MobileNetV2(input_shape=(224, 224, 3),
                             include_top=False,
                             weights='imagenet')
    
    # 2. Bekukan "Otak" dasar agar tidak rusak saat training awal
    base_model.trainable = False 

    # 3. Tambahkan "Kepala" baru khusus SIBI
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Lebih bagus dari Flatten untuk MobileNet
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),             # Mencegah overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = build_mobilenet_model(num_classes=train_generator.num_classes)

# Gunakan Learning Rate agak besar di awal karena kita cuma latih layer terakhir
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- TRAINING ---
checkpoint = ModelCheckpoint('model_mobilenet.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("Mulai Transfer Learning...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)

# --- PLOT ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy (MobileNetV2)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss (MobileNetV2)')
plt.legend()

plt.savefig('grafik_mobilenet.png')
plt.show()