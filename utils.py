import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Hàm để lưu và tải mô hình
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Mô hình đã được lưu tại: {filepath}")

def load_model(model, filepath, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Đã tải mô hình từ: {filepath}")
    return model

# Hàm để hiển thị ma trận hình ảnh
def show_image_grid(dataloader, num_images=25, figsize=(10, 10)):
    batch = next(iter(dataloader))
    images, labels = batch
    
    # Lấy một số ảnh từ batch
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Tạo grid và hiển thị
    grid = make_grid(images, nrow=int(np.sqrt(num_images)), padding=2, normalize=True)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Image Grid')
    plt.savefig('results/image_grid.png')
    plt.show()
    
    return images, labels

# Hàm để lưu lịch sử huấn luyện
def save_training_history(history, filepath='results/training_history.npy'):
    np.save(filepath, history)
    print(f"Lịch sử huấn luyện đã được lưu tại: {filepath}")

def load_training_history(filepath='results/training_history.npy'):
    if os.path.exists(filepath):
        history = np.load(filepath, allow_pickle=True).item()
        print(f"Đã tải lịch sử huấn luyện từ: {filepath}")
        return history
    else:
        print(f"Không tìm thấy file lịch sử tại: {filepath}")
        return None

# Hàm để hiển thị biểu đồ phân phối nhãn
def plot_label_distribution(dataset):
    labels = [label for _, label in dataset]
    
    plt.figure(figsize=(10, 6))
    counts = np.bincount(labels)
    
    # Sử dụng biểu đồ thanh
    plt.bar(range(10), counts, color='skyblue')
    
    # Thêm số lượng cụ thể trên mỗi cột
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center')
    
    plt.xlabel('Chữ số')
    plt.ylabel('Số lượng mẫu')
    plt.title('Phân phối của các chữ số trong dữ liệu')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('results/label_distribution.png')
    plt.show()
    
    # Dùng biểu đồ pie để hiển thị phân bố phần trăm
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=[f"{i} ({counts[i]})" for i in range(10)], 
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Phân bố phần trăm các chữ số')
    plt.savefig('results/label_distribution_pie.png')
    plt.show()
    
    return counts

# Hàm để trực quan hóa feature vectors trong không gian 2D
def visualize_feature_space(model, dataloader, method='tsne'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Thu thập feature vectors
    features = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Forward pass qua model cho đến lớp fully-connected cuối cùng
            # Giả sử model có phương thức extract_features
            if hasattr(model, 'extract_features'):
                feat = model.extract_features(images)
            else:
                # Nếu không có, ta cần tự implement
                # Đây chỉ là giả định, cần điều chỉnh theo cấu trúc mô hình cụ thể
                x = images
                for module in model.children():
                    if isinstance(module, torch.nn.Linear) and module.out_features == 10:
                        break
                    x = module(x)
                feat = x.view(x.size(0), -1)
            
            features.append(feat.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    # Ghép các batch
    features = np.vstack(features)
    labels = np.concatenate(labels_list)
    
    # Giảm chiều dữ liệu xuống 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
    
    features_2d = reducer.fit_transform(features)
    
    # Trực quan hóa
    plt.figure(figsize=(10, 8))
    
    # Sử dụng scatter plot với màu sắc khác nhau cho từng chữ số
    for i in range(10):
        idx = labels == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                   label=f'Digit {i}', alpha=0.7, edgecolors='w', linewidths=0.5)
    
    plt.title(f'Feature Space Visualization using {method.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/feature_space_{method}.png')
    plt.show()

# Kiểm tra folder tồn tại
def check_folder_structure():
    required_folders = ['data', 'data/digits', 'models', 'results']
    for folder in required_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Đã tạo thư mục: {folder}")
    
    # Kiểm tra thư mục dữ liệu chữ số
    for i in range(10):
        digit_folder = f'data/digits/{i}'
        if not os.path.exists(digit_folder):
            os.makedirs(digit_folder)
            print(f"Đã tạo thư mục: {digit_folder}")

if __name__ == "__main__":
    # Kiểm tra cấu trúc thư mục
    check_folder_structure()