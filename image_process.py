import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skfuzzy.cluster import cmeans

#Buoc 1 chuyen anh sang sang khong gian mau L(do sang)
def remake_image_L(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L_channel = image_lab[:, :, 0]
    return L_channel
# Bước 2: Phân cụm ảnh L bằng FCM và visualize kết quả
def fcm_segmentation(L_channel, n_clusters=3, visualize=True):
    # 1. Reshape ảnh L thành vector 1D
    data = L_channel.reshape(-1, 1).T.astype(np.float64)  # shape: (1, N)

    # 2. Áp dụng FCM
    cntr, u, _, _, _, _, _ = cmeans(data, c=n_clusters, m=2, error=0.005, maxiter=1000)

    # 3. Lấy nhãn cụm có giá trị membership cao nhất cho mỗi pixel
    cluster_labels = np.argmax(u, axis=0)  # shape: (N,)

    # 4. Reshape lại về dạng ảnh
    cluster_labels_img = cluster_labels.reshape(L_channel.shape)

    # 5. Tìm cụm tương ứng với lá (thường là cụm có diện tích lớn nhất ở giữa ảnh)
    h, w = L_channel.shape
    center_mask = np.zeros_like(L_channel, dtype=np.uint8)
    center_mask[h//4:3*h//4, w//4:3*w//4] = 1
    center_labels = cluster_labels_img[center_mask == 1]

    # Đếm số pixel mỗi cụm trong vùng trung tâm
    unique, counts = np.unique(center_labels, return_counts=True)
    leaf_cluster = unique[np.argmax(counts)]  # cụm có nhiều pixel nhất ở giữa ảnh → là lá

    # 6. Tạo mask nhị phân
    mask_leaf = (cluster_labels_img == leaf_cluster).astype(np.uint8)

    # 7. Visualize nếu cần
    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(L_channel, cmap='gray')
        plt.title('Ảnh L (gốc)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cluster_labels_img, cmap='jet')
        plt.title('Ảnh sau phân cụm FCM')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mask_leaf, cmap='gray')
        plt.title('Mask lá cây')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return mask_leaf
# Bước 3: Thay nền bằng màu xám và hiển thị ảnh kết quả
def replace_background_and_show(image, mask_leaf, background_color=(128, 128, 128), visualize = True):
    # Đảm bảo mask có shape (H, W, 1) để broadcast
    mask_3ch = np.repeat(mask_leaf[:, :, np.newaxis], 3, axis=2)

    # Tạo ảnh nền màu xám
    bg = np.full_like(image, background_color, dtype=np.uint8)

    # Thay nền: nếu mask == 1 → giữ pixel gốc; mask == 0 → thay bằng màu xám
    result = np.where(mask_3ch == 1, image, bg)

    # Hiển thị ảnh gốc, mask, ảnh sau khi thay nền
    if visualize:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Ảnh gốc')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_leaf, cmap='gray')
        plt.title('Mask lá cây')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Sau khi thay nền bằng màu xám')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return result
#BƯỚC 4: Tăng độ tương phản vùng lá bằng CLAHE
def enhance_leaf_contrast(image_clean, mask_leaf,visualize = True):
    """
    Tang độ tương phản vùng lá bằng CLAHE
    """
    lab = cv2.cvtColor(image_clean, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    l[mask_leaf == 1] = l_clahe[mask_leaf == 1]

    lab_clahe = cv2.merge((l, a, b))
    image_final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    # ve minh hoa
    if visualize:
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(image_clean, cv2.COLOR_BGR2RGB))
        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Ảnh sau tiền xử lý')
        plt.show()
    return image_final

#BƯỚC 5: Resize và chuẩn hóa ảnh để phù hợp ViT
def transform_for_ViT(image_final):
    """ Resize va chuan hoa anh de phu hop voi ViT"""
    transform = transforms.Compose([
        transforms.ToPILImage(),         # Nếu ảnh đầu vào là NumPy
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),           # Từ [0,255] → [0,1]
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # ImageNet
    ])
    return transform(image_final)

#BƯỚC 6: Trích xuất embedding từ mô hình ViT
def extract_embedding(image_tensor, model):
    """
    Trich xuat embedding tu ViT
    """
    image_tensor = image_tensor.unsqueeze(0)  # Chuyen ve (1, 3, 224, 224)
    with torch.no_grad():
        features = model(image_tensor)  # Trich xuat
    return features.squeeze(0)  # Ve lai ve (768,)