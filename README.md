
# Nhận diện Hình học & Chữ số viết tay bằng CNN  
**Môn học : xử lí ảnh**  
**Nhóm thực hiện:**  
B22DCCN183 - Đặng Tiến Đạt  B22DCCN184 - Đỗ Thành Đạt 
**Lớp:** D22CNPM02  
**Nhóm:** 05
**Giảng viên hướng dẫn:** Thầy Phạm Hoàng Việt  
**Thời gian hoàn thành:** Tháng 12/2025

---
Tài liệu báo cáo chi tiết của nhóm em  : 
https://docs.google.com/document/d/1o-N-TsbrvYr3C5MyQfOE02QHHdrGFqM-RbUQhIWiNUg/edit?tab=t.0

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
| Nhận diện 4 hình học        | 4      | **99.44% – 99.90%**         |  Biểu đồ đẹp, rất ổn định          |
| Nhận diện chữ số 0-9        | 10     | **98.70% – 99.50%**         | Chịu được ảnh xoay 90°, 180°, 270° hoàn toàn         |



---

### 4. Cách chạy chương trình
ở cả hai folder là chuso và hinhhoc ta cần chạy 2 file test.py để test mô hình 
