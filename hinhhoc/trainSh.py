import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============ CẤU HÌNH ============
DATA_DIR = r"C:\Users\Admin\Desktop\xla\hinhhoc\shapes"
IMG_SIZE = 64
SAMPLES_PER_CLASS = 300  # Giảm xuống 300 - VỪA ĐỦ cho bài toán đơn giản
BATCH_SIZE = 32

# ============ HÀM TẢI DỮ LIỆU (CHIA 3 TẬP ĐÚNG CHUẨN) ============
def load_data():
    """Tải dữ liệu và chia thành Train-Val-Test riêng biệt"""
    classes = ['circle', 'square', 'star', 'triangle']
    X = []
    y = []
    
    print("="*60)
    print("ĐANG TẢI DỮ LIỆU")
    print("="*60)
    
    for idx, class_name in enumerate(classes):
        folder = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(folder):
            print(f"❌ Không tìm thấy thư mục: {folder}")
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))][:SAMPLES_PER_CLASS]
        print(f"✅ {class_name.capitalize():>10}: {len(files)} ảnh")
        
        for file in files:
            try:
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).convert('L')  # Grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(idx)
            except Exception as e:
                print(f"⚠️  Lỗi đọc {file}: {e}")
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    
    print(f"\n📦 Tổng số ảnh: {len(X)}")
    
    # CHIA 3 TẬP RIÊNG BIỆT
    # Bước 1: Tách Test (15%) ra trước - TẬP NÀY KHÔNG BAO GIỜ CHẠM VÀO KHI TRAIN
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Bước 2: Từ 85% còn lại, chia thành Train (70% tổng) và Val (15% tổng)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )  # 0.176 * 0.85 ≈ 0.15 (15% của tổng)
    
    # Chuyển sang one-hot
    y_train = keras.utils.to_categorical(y_train, num_classes=4)
    y_val = keras.utils.to_categorical(y_val, num_classes=4)
    y_test = keras.utils.to_categorical(y_test, num_classes=4)
    
    print(f"\n📊 Phân chia dữ liệu:")
    print(f"   • Train:      {len(X_train):>4} ảnh ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   • Validation: {len(X_val):>4} ảnh ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   • Test:       {len(X_test):>4} ảnh ({len(X_test)/len(X)*100:.1f}%)")
    print("="*60 + "\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model():
    """Xây dựng CNN model tối ưu"""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                     input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_results(history, y_true, y_pred, class_names):
    """Vẽ các biểu đồ kết quả"""
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Số lượng'}, annot_kws={'size': 14})
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Nhãn thực tế', fontsize=13)
    plt.xlabel('Nhãn dự đoán', fontsize=13)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: confusion_matrix.png")
    
    # 2. Training History
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[0, 0].set_title('Loss theo Epoch', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Acc', linewidth=2, color='#2E86AB')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2, color='#A23B72')
    axes[0, 1].set_title('Accuracy theo Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy per class (bar chart)
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    x_pos = np.arange(len(class_names))
    axes[1, 0].bar(x_pos - 0.2, precision, 0.2, label='Precision', color='#06A77D')
    axes[1, 0].bar(x_pos, recall, 0.2, label='Recall', color='#D5A021')
    axes[1, 0].bar(x_pos + 0.2, f1, 0.2, label='F1-Score', color='#F18F01')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].set_title('Metrics từng lớp', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Sample distribution
    axes[1, 1].bar(class_names, [np.sum(y_true == i) for i in range(len(class_names))], color='#5E60CE')
    axes[1, 1].set_title('Phân bố mẫu Test', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Số lượng')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: training_analysis.png")
    
    plt.show()


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔷 CNN NHẬN DẠNG HÌNH HỌC - PHIÊN BẢN CHUẨN 🔷")
    print("="*60 + "\n")
    
    # 1. Load dữ liệu (đã chia 3 tập)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # 2. Data Augmentation - ĐƠN GIẢN VÀ HIỆU QUẢ
    train_datagen = ImageDataGenerator(
        rotation_range=20,           # Xoay ±20 độ
        width_shift_range=0.1,       # Dịch ngang 10%
        height_shift_range=0.1,      # Dịch dọc 10%
        zoom_range=0.1,              # Zoom 10%
        fill_mode='nearest'          # Lấp đầy vùng trống
    )
    
    # Validation/Test không augment
    val_datagen = ImageDataGenerator()
    
    # 3. Callbacks
    callbacks_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_shape_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 4. Build và train model
    print("🔨 Xây dựng model...")
    model = build_model()
    model.summary()
    
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU TRAINING")
    print("="*60 + "\n")
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=50,  # Giảm xuống 50 epochs
        validation_data=val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 5. Đánh giá trên tập TEST (chưa từng thấy)
    print("\n" + "="*60)
    print("📊 ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✨ Kết quả cuối cùng:")
    print(f"   • Test Loss:     {test_loss:.4f}")
    print(f"   • Test Accuracy: {test_acc*100:.2f}%")
    
    # 6. Dự đoán và phân tích chi tiết
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    class_names = ['Circle', 'Square', 'Star', 'Triangle']
    
    print(f"\n📈 Báo cáo chi tiết:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Ma trận nhầm lẫn:")
    print("          ", "  ".join([f"{c:>8}" for c in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>10}:", "  ".join([f"{val:>8}" for val in row]))
    
    # 7. Vẽ biểu đồ
    print(f"\n🎨 Đang vẽ biểu đồ...")
    plot_results(history, y_true, y_pred, class_names)
    
    print("\n" + "="*60)
    print("✅ HOÀN TẤT!")
    print("="*60)
    print(f"\n📁 File đã lưu:")
    print(f"   • best_shape_model.keras (Model tốt nhất)")
    print(f"   • confusion_matrix.png")
    print(f"   • training_analysis.png")
    print("\n")