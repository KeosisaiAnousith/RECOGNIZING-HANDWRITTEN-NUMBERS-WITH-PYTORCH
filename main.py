import os
from generate_data import setup_directories, generate_digit_images
from dataset import get_transforms, HandwrittenDigitDataset
from model import DigitRecognitionCNN
from train import train_model
from evaluate import evaluate_model
from visualize import show_examples, visualize_predictions
from interface import create_drawing_interface
from torch.utils.data import DataLoader, random_split

def main():
    print("==== Nhận diện chữ số viết tay bằng PyTorch ====")
    
    # Thiết lập thư mục
    setup_directories()
    
    # Kiểm tra dữ liệu đã tồn tại chưa
    generate_data = True
    for i in range(10):
        if os.path.exists(f'data/digits/{i}') and len(os.listdir(f'data/digits/{i}')) > 0:
            generate_data = False
            break
    
    # Tạo dữ liệu nếu cần
    if generate_data:
        samples_per_digit = 100  # Tạo chính xác 100 mẫu cho mỗi chữ số
        generate_digit_images(num_samples_per_digit=samples_per_digit)
    else:
        print("Dữ liệu đã tồn tại, bỏ qua bước tạo dữ liệu.")
    
    # Thiết lập transformations
    train_transform, test_transform = get_transforms()
    
    # Tạo dataset
    full_dataset = HandwrittenDigitDataset('data/digits', transform=train_transform)
    
    # Chia tập train và test theo tỷ lệ 80/20
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)  # 80% cho training
    test_size = total_size - train_size  # 20% cho testing
    
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size]
    )
    
    # Cập nhật transform cho test
    test_dataset.dataset.transform = test_transform
    
    # Tạo dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Số lượng mẫu huấn luyện: {len(train_dataset)} (80%)")
    print(f"Số lượng mẫu test: {len(test_dataset)} (20%)")
    
    # Hiển thị một số ví dụ từ dữ liệu
    print("Hiển thị một số ví dụ từ dữ liệu huấn luyện:")
    show_examples(train_loader)
    
    # Khởi tạo mô hình
    model = DigitRecognitionCNN()
    
    # Kiểm tra mô hình đã được huấn luyện chưa
    model_path = 'models/best_digit_model.pth'
    if os.path.exists(model_path):
        load_model = input("Đã tìm thấy mô hình đã huấn luyện. Bạn có muốn tải nó? (y/n): ")
        if load_model.lower() == 'y':
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Đã tải mô hình thành công!")
        else:
            # Huấn luyện mô hình mới
            print("Bắt đầu huấn luyện mô hình mới...")
            num_epochs = 100  # Số epochs cao để tăng độ chính xác
            train_model(model, train_loader, test_loader, num_epochs=num_epochs)
    else:
        # Huấn luyện mô hình mới
        print("Bắt đầu huấn luyện mô hình mới...")
        num_epochs = 100  # Số epochs cao để tăng độ chính xác
        train_model(model, train_loader, test_loader, num_epochs=num_epochs)
    
    # Đánh giá mô hình
    print("Đánh giá mô hình trên tập test:")
    evaluate_model(model, test_loader)
    
    # Hiển thị một số ví dụ dự đoán
    print("Hiển thị một số ví dụ dự đoán:")
    visualize_predictions(model, test_loader)
    
    # Tạo giao diện vẽ và dự đoán
    create_interface = input("Bạn có muốn tạo giao diện vẽ và dự đoán? (y/n): ")
    if create_interface.lower() == 'y':
        create_drawing_interface()
    
    print("Hoàn tất đồ án!")

if __name__ == "__main__":
    main()