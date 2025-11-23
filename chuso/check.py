import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============ CẤU HÌNH ============
DATA_DIR = r"C:\Users\Admin\Desktop\xla\chuso\augmented_images\augmented_images1"
IMG_SIZE = 64

def get_classes():
    """Lấy danh sách tất cả các lớp"""
    classes = []
    
    # Chữ số 0-9
    for i in range(10):
        classes.append(str(i))
    
    # Chữ thường a-z
    for c in 'abcdefghijklmnopqrstuvwxyz':
        classes.append(c)
    
    # Chữ hoa A_caps - Z_caps
    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        classes.append(f"{c}_caps")
    
    return classes

def check_data_quality():
    """Kiểm tra và hiển thị ảnh mẫu từ mỗi lớp"""
    
    CLASSES = get_classes()
    
    # Kiểm tra thư mục tồn tại
    print(f"Đang kiểm tra thư mục: {DATA_DIR}\n")
    if not os.path.exists(DATA_DIR):
        print(f"❌ Thư mục không tồn tại!")
        return
    
    # Thống kê
    print("=" * 60)
    print("THỐNG KÊ DỮ LIỆU")
    print("=" * 60)
    
    stats = []
    for idx, class_name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, class_name)
        
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            display_name = class_name[0] if '_caps' in class_name else class_name
            stats.append((display_name, class_name, len(files), folder))
            print(f"{display_name:3s} ({class_name:8s}): {len(files):4d} ảnh - {folder}")
        else:
            print(f"❌ THIẾU: {class_name}")
    
    print("=" * 60)
    print(f"Tổng số lớp tìm thấy: {len(stats)}/62")
    print("=" * 60 + "\n")
    
    # Hiển thị ảnh mẫu
    print("Đang hiển thị ảnh mẫu từ một số lớp...")
    
    # Chọn 20 lớp ngẫu nhiên để hiển thị
    sample_classes = np.random.choice(len(stats), min(20, len(stats)), replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Mẫu ảnh từ các lớp khác nhau', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for plot_idx, class_idx in enumerate(sample_classes):
        display_name, class_name, count, folder = stats[class_idx]
        
        # Lấy ảnh đầu tiên
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if files:
            img_path = os.path.join(folder, files[0])
            try:
                img = Image.open(img_path).convert('L')
                img_resized = img.resize((IMG_SIZE, IMG_SIZE))
                
                axes[plot_idx].imshow(img_resized, cmap='gray')
                axes[plot_idx].set_title(f"'{display_name}' ({count} ảnh)", fontsize=10, fontweight='bold')
                axes[plot_idx].axis('off')
                
                # In thông tin chi tiết
                print(f"  ✓ Lớp '{display_name}': {img.size} -> {img_resized.size}")
                
            except Exception as e:
                axes[plot_idx].text(0.5, 0.5, 'Lỗi đọc ảnh', 
                                   ha='center', va='center', fontsize=8)
                axes[plot_idx].set_title(f"'{display_name}' (LỖI)", fontsize=10, color='red')
                axes[plot_idx].axis('off')
                print(f"  ❌ Lỗi đọc '{display_name}': {e}")
    
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Đã lưu ảnh mẫu vào: data_samples.png")
    plt.show()
    
    # Kiểm tra phân bố số lượng
    counts = [s[2] for s in stats]
    print(f"\n📊 PHÂN BỐ SỐ LƯỢNG ẢNH:")
    print(f"   - Trung bình: {np.mean(counts):.0f} ảnh/lớp")
    print(f"   - Min: {np.min(counts)} ảnh")
    print(f"   - Max: {np.max(counts)} ảnh")
    print(f"   - Độ lệch chuẩn: {np.std(counts):.1f}")
    
    if np.std(counts) > 100:
        print(f"\n⚠️  CẢNH BÁO: Dữ liệu không cân bằng! (độ lệch chuẩn cao)")
    
    # Kiểm tra kích thước ảnh
    print(f"\n🔍 KIỂM TRA KÍCH THƯỚC ẢNH GỐC:")
    sizes = {}
    for display_name, class_name, count, folder in stats[:5]:  # Kiểm tra 5 lớp đầu
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in files[:3]:  # Kiểm tra 3 ảnh đầu
            try:
                img = Image.open(os.path.join(folder, file))
                size_key = f"{img.size[0]}x{img.size[1]}"
                sizes[size_key] = sizes.get(size_key, 0) + 1
            except:
                pass
    
    print("   Các kích thước phổ biến:")
    for size, count in sorted(sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {size}: {count} ảnh")

if __name__ == "__main__":
    check_data_quality()