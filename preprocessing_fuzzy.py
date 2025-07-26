import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans

def reshape_image_for_fuzzy(image_tensor):
    """
    Chuyển đổi ảnh từ dạng (C, H, W) sang dạng (H*W, C) để sử dụng trong Fuzzy C-Means.
    """
    if image_tensor.dim() == 4:
        # (batch, channels, height, width)
        b, c, h, w = image_tensor.shape
        return image_tensor.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
    elif image_tensor.dim() == 3:
        # (channels, height, width)
        c, h, w = image_tensor.shape
        return image_tensor.permute(1, 2, 0).reshape(-1, c).cpu().numpy()
    else:
        raise ValueError("Input tensor must be 3D or 4D")

def visualize_clustering(image_tensor, num_clusters):
    """
    Hiển thị ảnh gốc và ảnh sau khi phân cụm Fuzzy C-Means.
    """
    image_2d = reshape_image_for_fuzzy(image_tensor)
    
    # Thực hiện phân cụm Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = cmeans(
        image_2d.T, num_clusters, 2, error=0.005, maxiter=1000, init=None
    )
    #u la ma tran phan bo xs
    # cntr la trung tam cua cac cluster
    # Lấy nhãn của các cluster
    cluster_labels = np.argmax(u, axis=0)
    
    # Đưa về shape ảnh gốc (h, w)
    if image_tensor.dim() == 3:
        h, w = image_tensor.shape[1], image_tensor.shape[2]
    else:
        h, w = image_tensor.shape[2], image_tensor.shape[3]
    
    # Tạo ảnh phân cụm
    clustered_image = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            clustered_image[i, j] = cluster_labels[i * w + j]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(clustered_image, cmap='jet', alpha=0.5)
    plt.title("Fuzzy C-Means Clustering")
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()