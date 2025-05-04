import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from tqdm import tqdm

# Thiết lập thư mục cho dự án
def setup_directories():
    dirs = ['data', 'data/digits', 'models', 'results']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    
    # Tạo thư mục cho từng chữ số
    for i in range(10):
        os.makedirs(f'data/digits/{i}', exist_ok=True)

# Tạo dữ liệu chữ số viết tay giả lập
def generate_digit_images(num_samples_per_digit=100, image_size=28):
    print("Đang tạo dữ liệu chữ số viết tay...")
    
    # Thiết lập các font và kích thước khác nhau
    fonts = ['arial.ttf', 'times.ttf', 'cour.ttf', 'calibri.ttf']
    # Kiểm tra font có tồn tại không
    available_fonts = []
    for font in fonts:
        try:
            ImageFont.truetype(font, 20)
            available_fonts.append(font)
        except IOError:
            pass
    
    # Nếu không có font nào tồn tại, sử dụng font mặc định
    if not available_fonts:
        print("Không tìm thấy các font cụ thể, sử dụng font mặc định.")
        available_fonts = [ImageFont.load_default()]
    
    for digit in range(10):
        print(f"Đang tạo {num_samples_per_digit} hình ảnh cho chữ số {digit}...")
        for i in tqdm(range(num_samples_per_digit)):
            # Tạo hình trắng với nền đen
            img = Image.new('L', (image_size, image_size), 0)
            draw = ImageDraw.Draw(img)
            
            # Chọn kiểu random
            if isinstance(available_fonts[0], str):
                font_size = random.randint(14, 24)  # Tạo font size đa dạng hơn
                font = ImageFont.truetype(random.choice(available_fonts), font_size)
            else:
                font = available_fonts[0]
            
            # Vị trí ngẫu nhiên
            offset_x = random.randint(-4, 4)
            offset_y = random.randint(-4, 4)
            position = (image_size//2 - 6 + offset_x, image_size//2 - 10 + offset_y)
            
            # Vẽ chữ số màu trắng
            draw.text(position, str(digit), fill=255, font=font)
            
            # Biến đổi ngẫu nhiên - tạo nhiều kiểu dạng khác nhau
            angle = random.randint(-30, 30)  # Góc xoay
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            
            # Thêm co giãn để tạo kích thước đa dạng
            stretch_factor = random.uniform(0.8, 1.2)
            new_width = int(img.width * stretch_factor)
            img = img.resize((new_width, img.height), Image.BILINEAR)
            
            # Cắt hoặc thêm viền để về lại kích thước chuẩn
            if new_width != image_size:
                temp_img = Image.new('L', (image_size, image_size), 0)
                paste_pos = ((image_size - new_width) // 2, 0)
                temp_img.paste(img, paste_pos)
                img = temp_img
            
            # Thêm nhiễu ngẫu nhiên
            img_array = np.array(img)
            noise_level = random.randint(1, 40)  # Đảm bảo noise_level > 0
            if noise_level > 0:
                noise = np.random.randint(0, noise_level, size=img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            # Tạo thêm biến thể cho các chữ số cụ thể
            if digit == 0:
                # Đảm bảo số 0 hình tròn/ellipse rõ ràng
                # Làm dày nét viền nếu cần
                pass
            elif digit == 5:
                # Đôi khi làm thẳng phần trên của số 5
                if random.random() < 0.3:
                    for y in range(5, 10):
                        for x in range(8, 20):
                            if 0 <= y < img_array.shape[0] and 0 <= x < img_array.shape[1]:
                                img_array[y, x] = 255
            elif digit == 9:
                # Đôi khi tạo số 9 với đường loop rõ ràng hơn
                if random.random() < 0.3:
                    for y in range(8, 15):
                        for x in range(15, 20):
                            if 0 <= y < img_array.shape[0] and 0 <= x < img_array.shape[1]:
                                img_array[y, x] = 255
            
            # Đôi khi thay đổi độ tương phản
            img = Image.fromarray(img_array)
            if random.random() < 0.3:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Lưu hình ảnh
            img.save(f'data/digits/{digit}/{i}.png')
    
    print("Tạo dữ liệu hoàn tất!")

if __name__ == "__main__":
    # Chạy riêng file này để chỉ tạo dữ liệu
    setup_directories()
    generate_digit_images(num_samples_per_digit=100)