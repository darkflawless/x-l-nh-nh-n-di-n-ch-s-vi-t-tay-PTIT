import numpy as np
import os
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2  # Thêm import cv2

# ============ CẤU HÌNH ============
DATA_DIR = r"C:\Users\Admin\Desktop\xla\chuso\augmented_images\augmented_images1"
IMG_SIZE = 64
SAMPLES_PER_CLASS = 200
BATCH_SIZE = 32
EPOCHS = 50
ROTATION_AUGMENT = True  # Bật/tắt augmentation xoay

def get_classes():
    classes = []
    for i in range(10):  
        classes.append(str(i))
    return classes

CLASSES = get_classes()
NUM_CLASSES = len(CLASSES)
CLASS_LABELS = CLASSES

print(f"✓ Số lớp: {NUM_CLASSES}")
print(f"✓ Danh sách: {CLASS_LABELS}")

# ============ HÀM TẢI DỮ LIỆU ============
def load_data():
    X = []
    y = []
    
    for idx, class_name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, class_name)
        
        if not os.path.exists(folder):
            print(f"⚠ Không tìm thấy: {folder}")
            continue
        
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(files) < SAMPLES_PER_CLASS:
            print(f"⚠ Lớp '{class_name}' chỉ có {len(files)} ảnh")
        
        files = files[:SAMPLES_PER_CLASS]
        
        if ROTATION_AUGMENT:
            print(f"Đang tải '{class_name}': {len(files)} ảnh (x4 với xoay 0°/90°/180°/270°)")
        else:
            print(f"Đang tải '{class_name}': {len(files)} ảnh")
        
        for file in files:
            img_path = os.path.join(folder, file)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                
                # Thêm ảnh gốc
                X.append(img_array)
                y.append(idx)
                
                # Thêm các phiên bản xoay nếu bật ROTATION_AUGMENT
                if ROTATION_AUGMENT:
                    # Xoay 90°
                    img_90 = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
                    X.append(img_90)
                    y.append(idx)
                    
                    # Xoay 180°
                    img_180 = cv2.rotate(img_array, cv2.ROTATE_180)
                    X.append(img_180)
                    y.append(idx)
                    
                    # Xoay 270° (hay -90°)
                    img_270 = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    X.append(img_270)
                    y.append(idx)
            except:
                pass
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
    
    # In phân bố
    y_indices = np.argmax(y, axis=1)
    print(f"\n{'='*40}")
    print("Phân bố dữ liệu:")
    for i in range(NUM_CLASSES):
        count = np.sum(y_indices == i)
        print(f"  Lớp {CLASS_LABELS[i]}: {count} mẫu")
    print(f"{'='*40}\n")
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_indices
    )
    
    print(f"✓ Train: {len(X_train)} ảnh")
    print(f"✓ Test: {len(X_test)} ảnh")
    
    return X_train, y_train, X_test, y_test

# ============ XÂY DỰNG MODEL ============
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============ MAIN ============
if __name__ == "__main__":
    print("=" * 50)
    print("   CNN NHẬN DIỆN CHỮ SỐ 0-9")
    print("   (Hỗ trợ xoay 0°/90°/180°/270°)")
    print("=" * 50 + "\n")
    
    X_train, y_train, X_test, y_test = load_data()
    
    model = build_model()
    model.summary()
    
    # Data augmentation - Nhẹ vì đã có rotation augment trong data
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,       # Giảm xuống 15° (chỉ cho nhiễu nhẹ)
        width_shift_range=0.05,   
        height_shift_range=0.05,
        zoom_range=0.05,
        fill_mode='nearest'
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),  # Tăng patience
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7)
    ]
    
    print("\n--- Bắt đầu training ---")
    print("    (Dataset đã bao gồm xoay 0°/90°/180°/270°)")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Lưu model
    model.save("handwriting_model.keras")
    print("\n✓ Đã lưu model: handwriting_model.keras")
    
    # Đánh giá
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*40}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    print(f"{'='*40}")
    # Simplified evaluation output
    correct = 0
    total = len(X_test)

    for i in range(total):
        pred = model.predict(X_test[i:i+1], verbose=0)
        pred_class = np.argmax(pred)
        true_class = np.argmax(y_test[i])
        if pred_class == true_class:
            correct += 1

    print(f"\nKết quả: {correct}/{total} đúng ({correct/total*100:.1f}%)")
    # Test mẫu
    print("\n--- Một số dự đoán mẫu ---")
    for i in range(min(15, len(X_test))):
        pred = model.predict(X_test[i:i+1], verbose=0)
        pred_class = CLASS_LABELS[np.argmax(pred)]
        true_class = CLASS_LABELS[np.argmax(y_test[i])]
        
        symbol = "✓" if pred_class == true_class else "✗"
        print(f"  {symbol} Dự đoán: {pred_class} | Thực tế: {true_class}")