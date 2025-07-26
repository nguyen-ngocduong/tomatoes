import os
from PIL import Image

# Thư mục gốc (read-only)
src_root = '/home/nguyenngocduong/Documents/Python/tomatoes/tomato'
src_train = os.path.join(src_root, 'train')
src_val = os.path.join(src_root, 'val')

# Thư mục đích (nơi bạn có thể ghi dữ liệu)
dst_root = '/home/nguyenngocduong/Documents/Python/tomatoes/tomato_clean'
dst_train = os.path.join(dst_root, 'train')
dst_val = os.path.join(dst_root, 'val')

# Tạo các thư mục gốc nếu chưa có
os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_val, exist_ok=True)

def make_working_data(src_path, dst_path):
    """
    Duyệt qua các thư mục class và đổi tên từng ảnh, lưu vào thư mục mới.
    """
    for class_name in os.listdir(src_path):
        class_src_path = os.path.join(src_path, class_name)
        class_dst_path = os.path.join(dst_path, class_name.replace(' ', '_'))
        if not os.path.isdir(class_src_path):
            continue

        os.makedirs(class_dst_path, exist_ok=True)

        for idx, image_name in enumerate(os.listdir(class_src_path)):
            image_src_path = os.path.join(class_src_path, image_name)

            try:
                with Image.open(image_src_path) as img:
                    new_name = f"{class_name.replace(' ', '_')}_{idx:04d}.jpg"
                    image_dst_path = os.path.join(class_dst_path, new_name)
                    img.save(image_dst_path)
            except Exception as e:
                print(f"Lỗi khi xử lý {image_src_path}: {e}")
make_working_data(src_path= src_train, dst_path=dst_train)
make_working_data(src_path=src_val, dst_path=dst_val)