import numpy as np
from skimage import measure

def analyze_spot_patterns(segmented_mask, cluster_id=None):
    """
    Tính toán spot_density: tỉ lệ pixel thuộc vùng bệnh (cluster_id) trên toàn ảnh
    """
    #segmented_mask: ma trận phân cụm, mỗi pixel có giá trị là nhãn của cluster
    #cluster_id: id của cluster cần phân tích, nếu None thì tính cho tất cả
    H,W = segmented_mask.shape
    unique_clusters, counts = np.unique(segmented_mask, return_counts=True)
    #unique_clusters: danh sách các cluster duy nhất
    #counts: số lượng pixel thuộc mỗi cluster
    if cluster_id is not None:
        cluster_id = np.argmin(unique_clusters == cluster_id)
    # Tạo mặt nạ cho cụm cần phân tích
    mask = (segmented_mask == cluster_id).astype(np.uint8)
    # tinh toan mat do diem 
    spot_density = sp.sum(mask) / (H*W)
    return spot_density

def analyze_color_distribution(segmented_mask, image_rgb, cluster_id=None):
    """
    Tính toán tỉ lệ pixel thuộc từng cluster trên toàn ảnh
    """
    unique_clusters, counts = np.unique(segmented_mask, return_counts=True)
    if cluster_id is None:
        cluster_id = unique_clusters[np.argmin(counts)] #gia dinh vung benh la vung nho nhat
    mask = (segmented_mask == cluster_id).astype(np.uint8) #vung benh = 1, vung khong benh = 0
    mask_pixels = image_rgb[mask ==1]
    #tinh do lech chuan cua cac pixel thuoc vung benh
    #mask_pixels: cac pixel thuoc vung benh
    #image_rgb: anh goc
    #tra ve do lech chuan trung binh cua cac pixel thuoc vung
    if len(mask_pixels) > 0:
        std_rgb = np.std(mask_pixels, axis=0)  # độ lệch chuẩn trên 3 kênh
        return np.mean(std_rgb)  # trung bình độ lệch chuẩn các kênh R, G, B
    else:
        return 0.0

def analyze_leaf_shape(segmented_mask, cluster_id=None):
    """
    Tính shape_deformation: trung bình độ bất thường hình dạng (perimeter² / area)
    """
    unique_cluster, counts = np.unique(segmented_mask, return_counts=True)
    if cluster_id is None:
        cluster_id = unique_cluster[np.argmin(counts)]
    mask = (segmented_mask == cluster_id).astype(np.uint8)
    #phan tich vung benh
    labels = measure.label(mask, connectivity=2) #gan nhan cho cac vung lien thong
    #labels: ma tran nhan, moi vung lien thong co mot nhan duy nhất
    props = measure.regionprops(labels) #props chua thong tin ve cac vung lien thong

    bat_thuong = []

    for prop in props:
        if prop.area > 5: #dien tich vung  >  5
            chu_vi = prop.perimeter # chu vi vung
            dien_tich = prop.area # dien tich vung
            batthuong = (chu_vi ** 2) / (4 * np.pi * dien_tich)
            bat_thuong.append(batthuong)
    
    if len(bat_thuong) > 0:
        return np.mean(bat_thuong)
    else:
        return 0.0