import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Canvas, Button, Label
from tensorflow import keras

# ============ CẤU HÌNH ============
MODEL_PATH = r"C:\Users\Admin\Desktop\xla\chuso\handwriting_model.keras"
IMG_SIZE = 64

# Tạo danh sách lớp (giống train)
CLASSES = [str(i) for i in range(10)]  # 0-9

# Label hiển thị
CLASS_LABELS = []
for cls in CLASSES:
    if '_caps' in cls:
        CLASS_LABELS.append(cls[0])  # A_caps -> A
    else:
        CLASS_LABELS.append(cls)

model = keras.models.load_model(MODEL_PATH)
print("✓ Đã load model!")

# ============ HÀM DỰ ĐOÁN ============
def predict_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    pred = model.predict(img_array, verbose=0)
    idx = np.argmax(pred)
    return CLASS_LABELS[idx], pred[0], idx

def predict_from_array(img_array):
    img = Image.fromarray(img_array.astype('uint8')).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    pred = model.predict(arr, verbose=0)
    idx = np.argmax(pred)
    return CLASS_LABELS[idx], pred[0], idx

# ============ GIAO DIỆN ============
class HandwritingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Nhận dạng Chữ số & Chữ cái - CNN")
        self.root.geometry("500x650")
        
        Label(self.root, text="VẼ CHỮ SỐ/CHỮ CÁI HOẶC CHỌN ẢNH", 
              font=("Arial", 14, "bold")).pack(pady=10)
        
        self.canvas_size = 300
        self.canvas = Canvas(self.root, width=self.canvas_size, height=self.canvas_size, 
                            bg='white', cursor='cross', bd=2, relief='solid')
        self.canvas.pack(pady=10)
        self.canvas.bind('<B1-Motion>', self.draw)
        
        self.image_data = np.ones((self.canvas_size, self.canvas_size)) * 255
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        Button(btn_frame, text="Dự đoán", command=self.predict_drawing,
               width=10, bg='green', fg='white').pack(side='left', padx=5)
        Button(btn_frame, text="Xóa", command=self.clear_canvas,
               width=10).pack(side='left', padx=5)
        Button(btn_frame, text="Chọn ảnh", command=self.load_image,
               width=10).pack(side='left', padx=5)
        
        self.result_label = Label(self.root, text="Kết quả: ...", 
                                  font=("Arial", 24, "bold"), fg='blue')
        self.result_label.pack(pady=20)
        
        self.prob_label = Label(self.root, text="", font=("Arial", 10))
        self.prob_label.pack()
        
        # Top 5 dự đoán
        self.top5_label = Label(self.root, text="", font=("Arial", 10), justify='left')
        self.top5_label.pack(pady=10)
        
        self.root.mainloop()
    
    def draw(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        
        for i in range(max(0, y-r), min(self.canvas_size, y+r+1)):
            for j in range(max(0, x-r), min(self.canvas_size, x+r+1)):
                self.image_data[i, j] = 0
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image_data = np.ones((self.canvas_size, self.canvas_size)) * 255
        self.result_label.config(text="Kết quả: ...")
        self.prob_label.config(text="")
        self.top5_label.config(text="")
    
    def predict_drawing(self):
        result, probs, idx = predict_from_array(self.image_data)
        self.show_result(result, probs)
    
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            result, probs, idx = predict_image(path)
            self.show_result(result, probs)
            self.canvas.delete('all')
            self.canvas.create_text(150, 150, text=f"Ảnh: {path.split('/')[-1][:30]}", 
                                   font=("Arial", 9))
    
    def show_result(self, result, probs):
        conf = max(probs) * 100
        self.result_label.config(text=f"Kết quả: {result} ({conf:.1f}%)")
        
        # Top 5 dự đoán
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_text = "Top 5 dự đoán:\n"
        for i in top5_idx:
            top5_text += f"  {CLASS_LABELS[i]}: {probs[i]*100:.1f}%\n"
        self.top5_label.config(text=top5_text)

if __name__ == "__main__":
    print("Khởi động giao diện...")
    app = HandwritingApp()