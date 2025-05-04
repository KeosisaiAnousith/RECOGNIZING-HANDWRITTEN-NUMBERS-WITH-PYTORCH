import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Tạo tập dữ liệu tùy chỉnh
class HandwrittenDigitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Đọc tất cả các hình ảnh và nhãn
        for digit in range(10):
            digit_dir = os.path.join(data_dir, str(digit))
            if not os.path.exists(digit_dir):
                continue
                
            for img_name in os.listdir(digit_dir):
                if img_name.endswith('.png'):
                    img_path = os.path.join(digit_dir, img_name)
                    self.samples.append((img_path, digit))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Chuyển sang ảnh xám
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Augmentation dữ liệu
def get_transforms():
    train_transform = transforms.Compose([
        # Tăng cường data augmentation
        transforms.RandomAffine(
            degrees=30,  # Góc xoay
            translate=(0.2, 0.2),  # Dịch chuyển
            scale=(0.8, 1.2),  # Co giãn
            shear=10  # Độ nghiêng
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomInvert(p=0.1),  # Đảo ngược màu đôi khi
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return train_transform, test_transform