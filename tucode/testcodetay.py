import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODEL_PATH = r"C:\Users\Admin\Desktop\xla\tucode\best_shape_balanced.keras"

# Thay đường dẫn ảnh bạn muốn kiểm tra ở đây
# Ví dụ: Lấy thử một ảnh tròn hoặc vuông bất kỳ để test
IMAGE_TO_TEST = r"C:\Users\Admin\Desktop\xla\hinhhoc\shapes\circle\150.png" 

# --- 1. LOAD MODEL ---
print(f"Đang tải model từ: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("❌ Lỗi: Không tìm thấy file model (.keras). Hãy chạy file huấn luyện trước!")
    exit()

model = load_model(MODEL_PATH)
print("✓ Đã tải model thành công!")

# --- 2. HÀM XỬ LÝ ẢNH (QUAN TRỌNG NHẤT) ---
# Nguyên tắc: Lúc train xử lý thế nào (resize, grayscale, chia 255) thì lúc test phải y hệt
def preprocess_image(image_path):
    try:
        # Mở ảnh
        img = Image.open(image_path)
        
        # Convert sang Grayscale (L) và Resize về 64x64 (giống lúc train)
        img = img.convert('L').resize((64, 64))
        
        # Chuyển sang mảng numpy và chuẩn hóa về 0-1
        img_array = np.array(img) / 255.0
        
        # Reshape để phù hợp với input của model: (Batch_size, Height, Width, Channels)
        # Ta test 1 ảnh nên batch_size = 1 -> (1, 64, 64, 1)
        img_input = img_array.reshape(1, 64, 64, 1)
        
        return img_input, img_array # Trả về cả img_input để model đoán và img_array để vẽ
    except Exception as e:
        print(f"❌ Lỗi khi đọc ảnh: {e}")
        return None, None

# --- 3. THỰC HIỆN DỰ ĐOÁN ---
X_test, img_display = preprocess_image(IMAGE_TO_TEST)

if X_test is not None:
    # Model dự đoán (trả về 1 con số xác suất)
    prediction = model.predict(X_test)
    score = prediction[0][0] # Lấy giá trị thực ra khỏi mảng 2 chiều

    # Logic phân loại (Ngưỡng 0.5)
    if score > 0.5:
        label = "HÌNH TRÒN (Circle)"
        confidence = score * 100 # Tỉ lệ tự tin
    else:
        label = "HÌNH VUÔNG (Square)"
        confidence = (1 - score) * 100 # Đảo ngược tỉ lệ nếu là 0

    # --- 4. HIỂN THỊ KẾT QUẢ ---
    print("\n" + "="*30)
    print(f"🔍 KẾT QUẢ DỰ ĐOÁN")
    print(f"Label dự đoán: {label}")
    print(f"Giá trị Raw (Sigmoid): {score:.4f}")
    print(f"Độ tin cậy: {confidence:.2f}%")
    print("="*30)

    # Vẽ ảnh lên để xem
    plt.figure(figsize=(4, 4))
    plt.imshow(img_display, cmap='gray') # Vẽ ảnh xám
    plt.title(f"AI đoán: {label}\n({confidence:.1f}%)")
    plt.axis('off') # Tắt trục tọa độ cho đẹp
    plt.show()