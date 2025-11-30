import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Canvas, Button, Label
from tensorflow import keras

# ============ LOAD MODEL ============
MODEL_PATH = 'shape_model.keras'
IMG_SIZE = 64  # Model được train với 64x64
CLASSES = ['circle', 'square', 'star', 'triangle']
CLASSES_VN = ['Hình tròn', 'Hình vuông', 'Ngôi sao', 'Tam giác']

model = keras.models.load_model(r"C:\Users\Admin\Desktop\xla\hinhhoc\best_shape_balanced.keras")
print("Đã load model!")

# ============ HÀM DỰ ĐOÁN ============
def predict_image(image_path):
    """Dự đoán từ file ảnh"""
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    pred = model.predict(img_array, verbose=0)
    idx = np.argmax(pred)
    
    return CLASSES_VN[idx], pred[0]

def predict_from_array(img_array):
    """Dự đoán từ numpy array (canvas vẽ tay)"""
    img = Image.fromarray(img_array.astype('uint8')).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    pred = model.predict(arr, verbose=0)
    idx = np.argmax(pred)
    
    return CLASSES_VN[idx], pred[0]

# ============ GIAO DIỆN ============
class ShapeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Nhận dạng hình - CNN")
        self.root.geometry("500x600")
        
        # Tiêu đề
        Label(self.root, text="VẼ HÌNH HOẶC CHỌN ẢNH", 
              font=("Arial", 14, "bold")).pack(pady=10)
        
        # Canvas để vẽ - kích thước cố định
        self.canvas_size = 300
        self.canvas = Canvas(self.root, width=self.canvas_size, height=self.canvas_size, 
                            bg='white', cursor='cross')
        self.canvas.pack(pady=10)
        self.canvas.bind('<B1-Motion>', self.draw)
        
        # Lưu nét vẽ với kích thước canvas (300x300)
        self.image_data = np.ones((self.canvas_size, self.canvas_size)) * 255
        
        # Nút bấm
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        Button(btn_frame, text="Dự đoán", command=self.predict_drawing,
               width=10, bg='green', fg='white').pack(side='left', padx=5)
        Button(btn_frame, text="Xóa", command=self.clear_canvas,
               width=10).pack(side='left', padx=5)
        Button(btn_frame, text="Chọn ảnh", command=self.load_image,
               width=10).pack(side='left', padx=5)
        
        # Kết quả
        self.result_label = Label(self.root, text="Kết quả: ...", 
                                  font=("Arial", 16, "bold"))
        self.result_label.pack(pady=20)
        
        # Chi tiết xác suất
        self.prob_label = Label(self.root, text="", font=("Arial", 10))
        self.prob_label.pack()
        
        self.root.mainloop()
    
    def draw(self, event):
        """Vẽ lên canvas"""
        x, y = event.x, event.y
        r = 8  # Độ dày nét
        
        # Vẽ lên canvas
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        
        # Lưu vào array với kiểm tra bounds chặt chẽ
        y_min = max(0, y-r)
        y_max = min(self.canvas_size, y+r+1)
        x_min = max(0, x-r)
        x_max = min(self.canvas_size, x+r+1)
        
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                if 0 <= i < self.canvas_size and 0 <= j < self.canvas_size:
                    self.image_data[i, j] = 0
    
    def clear_canvas(self):
        """Xóa canvas"""
        self.canvas.delete('all')
        self.image_data = np.ones((self.canvas_size, self.canvas_size)) * 255
        self.result_label.config(text="Kết quả: ...")
        self.prob_label.config(text="")
    
    def predict_drawing(self):
        """Dự đoán hình vẽ tay"""
        result, probs = predict_from_array(self.image_data)
        self.show_result(result, probs)
    
    def load_image(self):
        """Chọn ảnh từ máy"""
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            result, probs = predict_image(path)
            self.show_result(result, probs)
            
            # Hiển thị ảnh lên canvas
            self.canvas.delete('all')
            self.canvas.create_text(150, 150, text=f"Ảnh: {path.split('/')[-1]}", 
                                   font=("Arial", 10))
    
    def show_result(self, result, probs):
        """Hiển thị kết quả"""
        conf = max(probs) * 100
        self.result_label.config(text=f"Kết quả: {result} ({conf:.1f}%)")
        
        # Chi tiết
        detail = "  |  ".join([f"{CLASSES_VN[i]}: {probs[i]*100:.1f}%" 
                              for i in range(4)])
        self.prob_label.config(text=detail)

# ============ CHẠY ============
if __name__ == "__main__":
    print("Khởi động giao diện...")
    app = ShapeApp()