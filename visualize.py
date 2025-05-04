import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# Hàm để hiển thị một số ví dụ từ dữ liệu
def show_examples(loader, num_examples=10):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Hiển thị
    plt.figure(figsize=(15, 3))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        img = images[i].squeeze().numpy()
        img = img * 0.5 + 0.5  # Denormalize
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/data_examples.png')
    plt.close()  # Đóng biểu đồ sau khi lưu

# Hàm để hiển thị một số ví dụ dự đoán
def visualize_predictions(model, test_loader, num_examples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Lấy một batch từ test_loader
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Dự đoán
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Chọn một số mẫu từ batch
    indices = torch.randperm(len(images))[:num_examples]
    
    # Vẽ hình ảnh và dự đoán
    fig = plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, num_examples, i + 1)
        img = images[idx].cpu().squeeze().numpy()
        img = (img * 0.5 + 0.5)  # Denormalize
        ax.imshow(img, cmap='gray')
        
        pred_label = predicted[idx].item()
        true_label = labels[idx].item()
        
        title_color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/example_predictions.png')
    plt.close()  # Đóng biểu đồ sau khi lưu

# Hàm để dự đoán một chữ số từ hình ảnh bên ngoài
def predict_custom_digit(model, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Đọc và tiền xử lý hình ảnh
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    # Dự đoán
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    return predicted_class, probabilities.cpu().numpy()