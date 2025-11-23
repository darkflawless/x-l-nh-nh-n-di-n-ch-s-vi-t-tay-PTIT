import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============ CẤU HÌNH ============
DATA_DIR = r"C:\Users\Admin\Desktop\xla\hinhhoc\shapes"
IMG_SIZE = 64
SAMPLES_PER_CLASS = 300

# ============ HÀM TẢI DỮ LIỆU ============
def load_data():
    """Tải dữ liệu đơn giản"""
    
    classes = ['circle', 'square', 'star', 'triangle']
    X = []
    y = []
    
    for idx, class_name in enumerate(classes):
        folder = os.path.join(DATA_DIR, class_name)
        files = os.listdir(folder)[:SAMPLES_PER_CLASS]
        
        print(f"Đang tải {class_name}: {len(files)} ảnh")
        
        for file in files:
            img_path = os.path.join(folder, file)
            
            img = Image.open(img_path).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            
            X.append(img_array)
            y.append(idx)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = keras.utils.to_categorical(y, num_classes=4)
    
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nTrain: {len(X_train)} ảnh")
    print(f"Test: {len(X_test)} ảnh")
    
    return X_train, y_train, X_test, y_test

# ============ XÂY DỰNG MODEL CẢI TIẾN ============
def build_model():
    """CNN mạnh hơn với Dropout và BatchNorm"""
    
    model = models.Sequential([
        # Block 1: 32 filters
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 2: 64 filters
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 3: 128 filters (THÊM MỚI - tăng khả năng học)
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Fully connected
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============ DATA AUGMENTATION (QUAN TRỌNG!) ============

data_augmentation = ImageDataGenerator(
    rotation_range=20,        # Xoay ±20 độ
    width_shift_range=0.1,    # Dịch ngang 10%
    height_shift_range=0.1,   # Dịch dọc 10%
    zoom_range=0.1,           # Phóng to/thu nhỏ 10%
    horizontal_flip=True,     # Lật ngang
    vertical_flip=True,       # Lật dọc
    fill_mode='nearest'
)

# ============ MAIN ============
print("=== CNN NHẬN DẠNG HÌNH (CẢI TIẾN) ===\n")

X_train, y_train, X_test, y_test = load_data()

model = build_model()
model.summary()

print("\n--- Bắt đầu train với Data Augmentation ---")
history = model.fit(
    data_augmentation.flow(X_train, y_train, batch_size=32),
    epochs=30,  # Tăng lên 30 epochs
    validation_data=(X_test, y_test),
    verbose=1
)

print("\n--- Kết quả ---")
loss, acc = model.evaluate(X_test, y_test)
print(f"Độ chính xác: {acc*100:.2f}%")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('result_improved.png')
plt.show()

# LƯU MODEL MỚI
model.save('best_shape_model.keras')
print("\nĐã lưu model: best_shape_model.keras (Cải tiến)")