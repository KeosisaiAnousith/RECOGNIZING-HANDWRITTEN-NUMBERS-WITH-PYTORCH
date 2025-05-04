import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Đánh giá mô hình trên tập test
def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Đánh giá mô hình"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    print(f"Độ chính xác trên tập test: {accuracy:.2f}%")
    
    # Vẽ biểu đồ confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # In báo cáo phân loại
    report = classification_report(all_labels, all_preds, digits=4)
    print("\nBáo cáo phân loại:")
    print(report)
    
    return accuracy, cm, report

if __name__ == "__main__":
    # Test module độc lập
    import torch
    from model import DigitRecognitionCNN
    from dataset import HandwrittenDigitDataset, get_transforms
    from torch.utils.data import DataLoader, random_split
    
    # Tạo dữ liệu test
    _, test_transform = get_transforms()
    full_dataset = HandwrittenDigitDataset('data/digits', transform=test_transform)
    
    # Chia dataset với tỷ lệ 80/20
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.transform = test_transform
    
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Tải mô hình
    model = DigitRecognitionCNN()
    try:
        model.load_state_dict(torch.load('models/best_digit_model.pth'))
        evaluate_model(model, test_loader)
    except:
        print("Không tìm thấy mô hình đã huấn luyện. Vui lòng huấn luyện mô hình trước.")