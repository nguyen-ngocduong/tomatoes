import numpy as np
from tqdm import tqdm
class MultiClassADALINE:
    def __init__(self, lr=0.01, epoches=100, input_dim=None, num_classes=3):
        self.lr = lr
        self.epoches = epoches
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.weights = None  # Will be initialized in fit
        self.bias = None
        self.cost_history = []
        
    def one_hot_encode(self, y):
        """Chuyển labels thành one-hot encoding"""
        encoded = np.zeros((len(y), self.num_classes))
        for i, label in enumerate(y):
            encoded[i, label] = 1
        return encoded
        
    def softmax(self, z):
        """Softmax activation"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    def activation(self, X):
        """Forward pass"""
        return np.dot(X, self.weights) + self.bias
        
    def cost(self, y_true, y_pred):
        """Cross-entropy loss"""
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        
    def fit(self, X, y):
        """
        Huấn luyện Multi-class ADALINE
        X: features matrix (n_samples, n_features)  
        y: labels vector (n_samples,) với giá trị 0 (nền), 1 (lá), 2 (sâu bệnh)
        """
        n_samples, n_features = X.shape
        self.input_dim = n_features
        
        # Khởi tạo weights và bias
        self.weights = np.random.normal(0, 0.01, (n_features, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))
        
        # Normalize dữ liệu
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self.mean) / self.std
        
        # One-hot encode labels
        y_encoded = self.one_hot_encode(y)
        
        # Training loop
        for epoch in tqdm(range(self.epoches)):
            # Forward pass
            linear_output = self.activation(X_normalized)
            y_pred = self.softmax(linear_output)
            
            # Tính cost
            cost = self.cost(y_encoded, y_pred)
            self.cost_history.append(cost)
            
            # Backward pass
            errors = y_encoded - y_pred
            self.weights += self.lr * np.dot(X_normalized.T, errors) / n_samples
            self.bias += self.lr * np.mean(errors, axis=0, keepdims=True)
    
    def predict(self, X):
        """Dự đoán với Multi-class ADALINE"""
        X_normalized = (X - self.mean) / self.std
        linear_output = self.activation(X_normalized)
        probabilities = self.softmax(linear_output)
        return np.argmax(probabilities, axis=1), probabilities