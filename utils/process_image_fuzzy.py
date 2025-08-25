import cv2
import numpy as np
from skimage.feature import hog
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def preprocessing_image(train_image):
    """
    Hàm tiền xử lý ảnh:
    1. Resize về 600x400
    2. Sử dụng HoG (OpenCV)
    3. Chuyển sang LAB
    """
    for image in train_image:
        img = cv2.imread(image)
        img = cv2.resize(img, (600, 400))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HOG với sklearn
        hog_features, hog_image = hog(img_gray,
                                      orientations=9,
                                      pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2),
                                      block_norm='L2-Hys',
                                      visualize=True)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        yield img_lab, hog_features, hog_image

class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, error=1e-6, random_state=42):
        self.n_clusters = n_clusters
        self.m = m  # Fuzziness parameter
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

    def _initialize_membership_matrix(self, n_points):
        """Khởi tạo ma trận membership ngẫu nhiên"""
        np.random.seed(self.random_state)
        membership = np.random.random((n_points, self.n_clusters))
        # Normalize để tổng mỗi hàng = 1
        membership = membership / np.sum(membership, axis=1, keepdims=True)
        return membership

    def _calculate_cluster_centers(self, X, membership):
        """Tính toán tâm cụm"""
        um = membership ** self.m
        centers = np.dot(um.T, X) / np.sum(um.T, axis=1, keepdims=True)
        return centers

    def _calculate_membership_matrix(self, X, centers):
        """Cập nhật ma trận membership"""
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i, center in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)

        # Tránh chia cho 0
        distances = np.fmax(distances, np.finfo(np.float64).eps)

        # Tính membership matrix
        membership = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                membership[:, i] += (distances[:, i] / distances[:, j]) ** (2/(self.m-1))
            membership[:, i] = 1.0 / membership[:, i]

        return membership

    def fit(self, X):
        """Huấn luyện Fuzzy C-Means"""
        n_points = X.shape[0]

        # Khởi tạo membership matrix
        membership = self._initialize_membership_matrix(n_points)

        for iteration in range(self.max_iter):
            # Tính tâm cụm
            centers = self._calculate_cluster_centers(X, membership)

            # Cập nhật membership matrix
            new_membership = self._calculate_membership_matrix(X, centers)

            # Kiểm tra hội tụ
            if np.linalg.norm(new_membership - membership) < self.error:
                break

            membership = new_membership

        self.centers = centers
        self.membership = membership
        return self

    def predict(self, X=None):
        """Dự đoán cụm cho mỗi điểm"""
        if X is None:
            return np.argmax(self.membership, axis=1)
        else:
            membership = self._calculate_membership_matrix(X, self.centers)
            return np.argmax(membership, axis=1)

def segment_image_fcm(image_lab, n_clusters=3):
    """
    Phân đoạn ảnh bằng Fuzzy C-Means thay vì SLIC
    """
    h, w, c = image_lab.shape

    # Reshape ảnh thành vector 2D (pixels x features)
    pixels = image_lab.reshape(-1, c)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    pixels_normalized = scaler.fit_transform(pixels)

    # Áp dụng Fuzzy C-Means
    fcm = FuzzyCMeans(n_clusters=n_clusters, m=2.0, max_iter=100)
    fcm.fit(pixels_normalized)

    # Lấy nhãn cụm cho mỗi pixel
    labels = fcm.predict()

    # Reshape về dạng ảnh
    segments = labels.reshape(h, w)

    return segments, fcm

def visualize_image_segments(image, segments, numSegments):
    """
    ham hien thi anh va cac phan doan
    """
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")