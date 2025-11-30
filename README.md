
# Nhận diện Hình học & Chữ số viết tay bằng CNN  
**Đồ án môn Nhập môn Học máy và Thị giác Máy tính**  
**Sinh viên thực hiện:** [Tên bạn]  
**Lớp:** [Lớp học phần]  
**Giảng viên hướng dẫn:** [Tên thầy/cô]

---

### Tổng quan dự án
Dự án gồm **hai mô hình CNN độc lập**:
1. **Nhận diện 4 hình học cơ bản**: Circle, Square, Star, Triangle  
2. **Nhận diện chữ số viết tay 0-9** (có hỗ trợ ảnh bị xoay 0°/90°/180°/270°)

Cả hai đều được huấn luyện từ đầu (from scratch) bằng TensorFlow/Keras, không dùng mô hình pre-trained.

---
---

### Yêu cầu môi trường
- Python 3.9 – 3.11
- TensorFlow ≥ 2.13
- OpenCV (cv2) – chỉ dùng trong train.py
- Các thư viện khác: numpy, matplotlib, seaborn, scikit-learn, pillow


---

### 3. Kết quả đạt được (đã kiểm thử tháng 12/2025)

| Mô hình                     | Số lớp | Độ chính xác trên tập Test | Ghi chú                                              |
|-----------------------------|--------|-----------------------------|------------------------------------------------------|
| Nhận diện 4 hình học        | 4      | **99.44% – 99.90%**         | Đánh giá khoa học, biểu đồ đẹp, rất ổn định          |
| Nhận diện chữ số 0-9        | 10     | **98.70% – 99.50%**         | Chịu được ảnh xoay 90°, 180°, 270° hoàn toàn         |

---

### 4. Cách chạy chương trình

#### 4.1. Cài đặt môi trường
```bash
pip install -r requirements.txt
