import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
import os

# Huấn luyện mô hình
def train_model(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")
    
    model = model.to(device)
    
    # Tạo criterion với weight để đối phó với class imbalance 
    criterion = nn.CrossEntropyLoss()
    
    # Sử dụng tỷ lệ học thấp hơn và weight decay cho regularization
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    early_stop_patience = 20
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100.0 * correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        
        # Điều chỉnh learning rate
        scheduler.step(epoch_val_loss)
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {elapsed_time:.2f}s")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%")
        
        # Kiểm tra và lưu mô hình tốt nhất
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), 'models/best_digit_model.pth')
            print(f"Model saved with validation accuracy: {best_val_accuracy:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Lưu checkpoint định kỳ
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, f'models/checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
        
    # Vẽ biểu đồ loss và accuracy cuối cùng
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title('Loss over Epochs', fontsize=14)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title('Accuracy over Epochs', fontsize=14)
    plt.ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('results/training_validation_curves.png')
    plt.close()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == "__main__":
    # Test module độc lập
    from model import DigitRecognitionCNN
    from dataset import HandwrittenDigitDataset, get_transforms
    from torch.utils.data import DataLoader, random_split
    
    # Tạo dữ liệu test
    train_transform, test_transform = get_transforms()
    full_dataset = HandwrittenDigitDataset('data/digits', transform=train_transform)
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = test_transform
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    model = DigitRecognitionCNN()
    train_model(model, train_loader, val_loader, num_epochs=100)