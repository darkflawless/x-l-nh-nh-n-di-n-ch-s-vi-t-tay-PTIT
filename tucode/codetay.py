import os
from PIL import Image
import numpy as np

# --- CẤU HÌNH ---
base_folder = r"C:\Users\Admin\Desktop\xla\hinhhoc\shapes"

X = []
y = []

folder_circle = os.path.join(base_folder, "circle")
files_circle = os.listdir(folder_circle)[:30] # Lấy tạm 10 cái

print("Đang load Tròn (Label 1)...")
for file in files_circle:
    # Xử lý ảnh (tạo X)
    path = os.path.join(folder_circle, file)
    img = Image.open(path).convert('L').resize((64, 64))
    img_array = np.array(img) / 255.0
    X.append(img_array)
    
    # Dán nhãn (tạo y) -> QUAN TRỌNG: Dòng bạn thiếu
    y.append(1) 

# --- BƯỚC 2: LOAD HÌNH VUÔNG (Dạy nó: Đây là KHÔNG - số 0) ---
folder_square = os.path.join(base_folder, "square")
files_square = os.listdir(folder_square)[:30] # Lấy tạm 10 cái đối chứng

print("Đang load Vuông (Label 0)...")
for file in files_square:
    # Xử lý ảnh (tạo X)
    path = os.path.join(folder_square, file)
    img = Image.open(path).convert('L').resize((64, 64))
    img_array = np.array(img) / 255.0
    X.append(img_array)
    
    # Dán nhãn (tạo y) -> QUAN TRỌNG
    y.append(0)

# --- BƯỚC 3: CHỐT ĐƠN ---
X = np.array(X).reshape(-1, 64, 64, 1)
y = np.array(y) # Với Sigmoid (0/1) thì không cần to_categorical, để nguyên array số là được

print("\n--- KẾT QUẢ ---")
print("Shape của X (Dữ liệu):", X.shape) # Sẽ là (20, 64, 64, 1) -> 20 ảnh
print("Nội dung của y (Đáp án):", y)     # Sẽ là [1, 1... 0, 0...]

from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3) , activation='relu', input_shape=(64,64,1)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # QUAN TRỌNG: Sigmoid đi cặp với Binary Crossentropy
    metrics=['accuracy']
)

model.fit(X, y, epochs=10, batch_size=4)

model.save(r"C:\Users\Admin\Desktop\xla\tucode\best_shape_balanced.keras")
print("✓ Đã lưu model!")
