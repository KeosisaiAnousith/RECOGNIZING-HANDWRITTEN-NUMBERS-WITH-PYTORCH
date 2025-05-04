import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox, Canvas, Button, Label, Frame, Scale, HORIZONTAL, StringVar, Radiobutton
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model import DigitRecognitionCNN
from scipy import ndimage

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng Dụng Nhận Dạng Chữ Số Viết Tay")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # Drawing state variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.brush_size = 20
        self.drawing_mode = "pen"  # "pen" or "eraser"
        
        # Main frames
        self.left_frame = Frame(root, width=450, height=700, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = Frame(root, width=450, height=700, bg="#f0f0f0")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ===== LEFT FRAME COMPONENTS =====
        # Canvas title
        self.canvas_title = Label(self.left_frame, text="Vẽ một chữ số ở đây:", 
                                 font=("Arial", 14, "bold"), bg="#f0f0f0")
        self.canvas_title.pack(pady=(20, 10))
        
        # Canvas with border
        self.canvas_border = Frame(self.left_frame, bd=2, relief="ridge", bg="black")
        self.canvas_border.pack(pady=10)
        
        self.canvas = Canvas(self.canvas_border, width=280, height=280, bg="black", 
                           bd=0, highlightthickness=0)
        self.canvas.pack()
        
        # Store drawing data for pixel-based processing
        self.drawing_data = np.zeros((28, 28), dtype=np.float32)
        
        # Brush control frame
        self.brush_control_frame = Frame(self.left_frame, bg="#f0f0f0")
        self.brush_control_frame.pack(pady=10)
        
        # Brush size control
        Label(self.brush_control_frame, text="Kích thước bút:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.brush_slider = Scale(self.brush_control_frame, from_=5, to=30, orient=HORIZONTAL, 
                                 length=150, command=self.update_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=5)
        
        # Pen/eraser mode buttons
        self.pen_button = Button(self.brush_control_frame, text="Bút vẽ", command=self.set_pen_mode, 
                                width=8, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.pen_button.pack(side=tk.LEFT, padx=5)
        
        self.eraser_button = Button(self.brush_control_frame, text="Tẩy", command=self.set_eraser_mode, 
                                   width=8, font=("Arial", 10))
        self.eraser_button.pack(side=tk.LEFT, padx=5)
        
        # Thêm Radio button để chọn hướng viết
        self.orientation_frame = Frame(self.left_frame, bg="#f0f0f0")
        self.orientation_frame.pack(pady=5)
        
        Label(self.orientation_frame,text="Hướng viết:",bg="#f0f0f0",font=("Arial",10,"bold")).pack(side=tk.LEFT,padx=5)
        
        self.orientation = StringVar(value="normal")
        
        Radiobutton(self.orientation_frame, text="Bình thường", variable=self.orientation, 
                    value="normal", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        Radiobutton(self.orientation_frame, text="Đảo ngược", variable=self.orientation, 
                    value="inverted", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        Radiobutton(self.orientation_frame, text="Quay trái", variable=self.orientation, 
                    value="left", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        Radiobutton(self.orientation_frame, text="Quay phải", variable=self.orientation, 
                    value="right", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        self.buttons_frame = Frame(self.left_frame, bg="#f0f0f0")
        self.buttons_frame.pack(pady=10)
        
        self.clear_button = Button(self.buttons_frame, text="Xóa", command=self.clear_canvas, 
                                  font=("Arial", 11, "bold"), width=10, bg="#FF5722", fg="white")
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        self.predict_button = Button(self.buttons_frame, text="Dự Đoán", command=self.predict, 
                                    font=("Arial", 11, "bold"), width=10, bg="#2196F3", fg="white")
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        self.accuracy_button = Button(self.buttons_frame, text="Độ Chính Xác", command=self.show_accuracy, 
                                     font=("Arial", 11, "bold"), width=10, bg="#9C27B0", fg="white")
        self.accuracy_button.pack(side=tk.LEFT, padx=10)
        
        # ===== RIGHT FRAME COMPONENTS =====
        # Result title
        self.result_label = Label(self.right_frame, text="Kết Quả Dự Đoán:", 
                                 font=("Arial", 16, "bold"), bg="#f0f0f0")
        self.result_label.pack(pady=(20, 10))
        
        # Prediction display frame
        self.prediction_frame = Frame(self.right_frame, bg="#e0e0e0", 
                                     bd=2, relief="ridge", padx=15, pady=15)
        self.prediction_frame.pack(pady=10)
        
        self.prediction_label = Label(self.prediction_frame, text="", 
                                     font=("Arial", 60, "bold"), fg="#1976D2", bg="#e0e0e0",
                                     width=2, height=1)
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = Label(self.prediction_frame, text="", 
                                     font=("Arial", 12), bg="#e0e0e0")
        self.confidence_label.pack(pady=5)
        
        # Processed images frame
        self.processed_images_frame = Frame(self.right_frame, bg="#f0f0f0")
        self.processed_images_frame.pack(pady=10, fill=tk.X)
        
        Label(self.processed_images_frame, text="Các bước xử lý hình ảnh:", 
             font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=5)
        
        # Frame for original and processed images
        self.images_row = Frame(self.processed_images_frame, bg="#f0f0f0")
        self.images_row.pack(fill=tk.X)
        
        # Original image display
        self.original_frame = Frame(self.images_row, bg="#f0f0f0")
        self.original_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        Label(self.original_frame, text="Ảnh gốc", font=("Arial", 10), bg="#f0f0f0").pack()
        
        self.fig_original = Figure(figsize=(2, 2), dpi=100)
        self.ax_original = self.fig_original.add_subplot(111)
        self.ax_original.set_axis_off()
        self.canvas_original = FigureCanvasTkAgg(self.fig_original, master=self.original_frame)
        self.canvas_original.draw()
        self.canvas_original.get_tk_widget().pack()
        
        # Centered image display
        self.centered_frame = Frame(self.images_row, bg="#f0f0f0")
        self.centered_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        Label(self.centered_frame, text="Ảnh căn giữa", font=("Arial", 10), bg="#f0f0f0").pack()
        
        self.fig_centered = Figure(figsize=(2, 2), dpi=100)
        self.ax_centered = self.fig_centered.add_subplot(111)
        self.ax_centered.set_axis_off()
        self.canvas_centered = FigureCanvasTkAgg(self.fig_centered, master=self.centered_frame)
        self.canvas_centered.draw()
        self.canvas_centered.get_tk_widget().pack()
        
        # Final processed image display
        self.final_frame = Frame(self.images_row, bg="#f0f0f0")
        self.final_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        Label(self.final_frame, text="Ảnh cuối cùng", font=("Arial", 10), bg="#f0f0f0").pack()
        
        self.fig_final = Figure(figsize=(2, 2), dpi=100)
        self.ax_final = self.fig_final.add_subplot(111)
        self.ax_final.set_axis_off()
        self.canvas_final = FigureCanvasTkAgg(self.fig_final, master=self.final_frame)
        self.canvas_final.draw()
        self.canvas_final.get_tk_widget().pack()
        
        # Probability chart
        self.chart_frame = Frame(self.right_frame, bg="#f0f0f0")
        self.chart_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.chart_label = Label(self.chart_frame, text="Biểu đồ xác suất cho mỗi chữ số", 
                                font=("Arial", 10), bg="#f0f0f0")
        self.chart_label.pack()
        
        # Status bar
        self.status_bar = Label(root, text="Sẵn sàng", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
    
    def load_model(self):
        model_path = 'models/best_digit_model.pth'
        if os.path.exists(model_path):
            model = DigitRecognitionCNN()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        else:
            messagebox.showerror("Lỗi", "Không tìm thấy file mô hình. Vui lòng huấn luyện mô hình trước.")
            return None
    
    def update_brush_size(self, val):
        self.brush_size = int(val)
    
    def set_pen_mode(self):
        self.drawing_mode = "pen"
        self.pen_button.config(bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.eraser_button.config(bg="#f0f0f0", fg="black", font=("Arial", 10))
    
    def set_eraser_mode(self):
        self.drawing_mode = "eraser"
        self.eraser_button.config(bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.pen_button.config(bg="#f0f0f0", fg="black", font=("Arial", 10))
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            color = "white" if self.drawing_mode == "pen" else "black"
            
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                   width=self.brush_size, fill=color, 
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Update drawing data array
            scale = 28 / 280  # Convert from 280px canvas to 28x28 array
            
            # Simulate line with multiple points
            for i in range(100):
                t = i / 100.0
                lx = int((self.last_x * (1 - t) + x * t) * scale)
                ly = int((self.last_y * (1 - t) + y * t) * scale)
                
                if 0 <= lx < 28 and 0 <= ly < 28:
                    radius = int(self.brush_size * scale / 2)
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            nx, ny = lx + dx, ly + dy
                            if 0 <= nx < 28 and 0 <= ny < 28:
                                dist = (dx**2 + dy**2) ** 0.5
                                if dist <= radius:
                                    intensity = 1.0 * (1 - dist/radius)
                                    if self.drawing_mode == "pen":
                                        self.drawing_data[ny, nx] = max(self.drawing_data[ny, nx], intensity)
                                    else:  # eraser
                                        self.drawing_data[ny, nx] = 0
            
            self.last_x, self.last_y = x, y
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing_data = np.zeros((28, 28), dtype=np.float32)
        self.prediction_label.config(text="")
        self.confidence_label.config(text="")
        
        # Xóa biểu đồ xác suất
        self.ax.clear()
        self.canvas_widget.draw()
        
        # Xóa các hình ảnh đã xử lý
        self.ax_original.clear()
        self.ax_original.set_axis_off()
        self.canvas_original.draw()
        
        self.ax_centered.clear()
        self.ax_centered.set_axis_off()
        self.canvas_centered.draw()
        
        self.ax_final.clear()
        self.ax_final.set_axis_off()
        self.canvas_final.draw()
        
        self.status_bar.config(text="Đã xóa")
    
    def center_digit(self, image_data):
        # Đảm bảo image_data có giá trị
        if np.max(image_data) == 0:
            return image_data
        
        # Find the bounding box of the digit
        rows = np.any(image_data > 0.1, axis=1)
        cols = np.any(image_data > 0.1, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return image_data  # Return as is if the image is empty
        
        # Get the non-empty region
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding around the digit
        padding = 2
        rmin = max(0, rmin - padding)
        rmax = min(image_data.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(image_data.shape[1] - 1, cmax + padding)
        
        # Extract the digit
        digit = image_data[rmin:rmax+1, cmin:cmax+1]
        
        # Create a centered image
        centered = np.zeros_like(image_data)
        
        # Calculate center offsets
        h, w = digit.shape
        offset_h = (image_data.shape[0] - h) // 2
        offset_w = (image_data.shape[1] - w) // 2
        
        # Place the digit in the center
        centered[offset_h:offset_h+h, offset_w:offset_w+w] = digit
        
        return centered
    
    def preprocess_by_orientation(self, image_data):
        # Xử lý theo hướng viết được chọn
        orientation = self.orientation.get()
        
        # Tạo bản sao để tránh lỗi stride âm
        processed = image_data.copy()
        
        if orientation == "inverted":
            processed = np.rot90(processed, k=2).copy()
        elif orientation == "left":
            processed = np.rot90(processed, k=1).copy()
        elif orientation == "right":
            processed = np.rot90(processed, k=3).copy()
        
        # Làm rõ nét bằng cách tăng ngưỡng
        threshold = 0.2
        processed = np.where(processed > threshold, 1.0, 0.0)
        
        # Căn giữa chữ số
        centered = self.center_digit(processed)
        
        return centered
    
    def predict(self):
        if self.model is None:
            messagebox.showerror("Lỗi", "Mô hình không được tải. Vui lòng huấn luyện mô hình trước.")
            return
        
        self.status_bar.config(text="Đang xử lý...")
        self.root.update()
        
        try:
            # Check if there's any drawing
            if np.max(self.drawing_data) == 0:
                messagebox.showinfo("Thông báo", "Vui lòng vẽ một chữ số trước khi dự đoán.")
                self.status_bar.config(text="Sẵn sàng")
                return
            
            # Lưu ảnh gốc
            original_data = self.drawing_data.copy()
            
            # Hiển thị ảnh gốc
            self.ax_original.clear()
            self.ax_original.imshow(original_data, cmap='gray')
            self.ax_original.set_axis_off()
            self.fig_original.tight_layout()
            self.canvas_original.draw()
            
            # Xử lý theo hướng viết được chọn
            processed_data = self.preprocess_by_orientation(original_data)
            
            # Hiển thị ảnh đã căn giữa
            self.ax_centered.clear()
            self.ax_centered.imshow(processed_data, cmap='gray')
            self.ax_centered.set_axis_off()
            self.fig_centered.tight_layout()
            self.canvas_centered.draw()
            
            # Thêm xử lý làm dày nét nếu cần
            from scipy import ndimage
            if np.sum(processed_data) < 50:  # Nếu có ít điểm ảnh sáng
                processed_data = ndimage.binary_dilation(processed_data, 
                                                      structure=np.ones((2, 2))).astype(np.float32)
            
            # Hiển thị ảnh cuối cùng
            self.ax_final.clear()
            self.ax_final.imshow(processed_data, cmap='gray')
            self.ax_final.set_axis_off()
            self.fig_final.tight_layout()
            self.canvas_final.draw()
            
            # Chuyển thành tensor và đảm bảo kích thước đúng
            img_tensor = torch.FloatTensor(processed_data)
            
            # Đảm bảo tensor có 4 chiều [batch_size, channels, height, width]
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            
            # Resize về 28x28 
            img_tensor = F.interpolate(img_tensor, size=(28, 28), mode='bilinear', align_corners=False)
            
            # Áp dụng normalize
            normalize = transforms.Normalize((0.5,), (0.5,))
            img_tensor = normalize(img_tensor)
            
            # Chuyển sang device
            img_tensor = img_tensor.to(self.device)
            
            # Dự đoán
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            # Update GUI with results
            confidence_percent = confidence * 100
            self.prediction_label.config(text=str(predicted_class))
            self.confidence_label.config(text=f"Độ tin cậy: {confidence_percent:.2f}%")
            
            # Update probability chart - hiển thị tất cả xác suất
            self.ax.clear()
            probs = probabilities.cpu().numpy()
            
            # Tính toán và hiển thị biểu đồ với xác suất cao (>=80%)
            bars = self.ax.bar(range(10), probs, color='skyblue')
            
            # Highlight predicted digit
            bars[predicted_class].set_color('red')
            
            self.ax.set_xlabel('Chữ số')
            self.ax.set_ylabel('Xác suất')
            self.ax.set_xticks(range(10))
            self.ax.set_title('Xác suất dự đoán')
            self.ax.set_ylim([0, 1.0])
            
            # Thêm giá trị xác suất lên mỗi cột
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    self.ax.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}', ha='center', va='bottom', rotation=0, fontsize=8)
            
            self.fig.tight_layout()
            self.canvas_widget.draw()
            
            self.status_bar.config(text=f"Dự đoán: {predicted_class} với độ tin cậy {confidence_percent:.2f}%")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý hình ảnh: {str(e)}")
            print(f"Chi tiết lỗi: {e}")
            import traceback
            traceback.print_exc()  # In ra stack trace đầy đủ
            self.status_bar.config(text="Có lỗi xảy ra")
    
    def show_accuracy(self):
        try:
            # Create a popup window to show model accuracy info
            accuracy_window = tk.Toplevel(self.root)
            accuracy_window.title("Độ Chính Xác Mô Hình")
            accuracy_window.geometry("800x600")
            accuracy_window.configure(bg="#f0f0f0")
            
            # Model accuracy information
            info_frame = Frame(accuracy_window, bg="#e0e0e0", bd=2, relief="ridge", padx=20, pady=20)
            info_frame.pack(fill=tk.X, padx=20, pady=20)
            
            # Display accuracy information
            accuracy_info = """
            Mô hình CNN nhận dạng chữ số viết tay:
            
            - Đạt độ chính xác > 95% trên tập dữ liệu test
            - Huấn luyện trong 100 epochs
            - Sử dụng dataset với 100 mẫu cho mỗi chữ số, tổng 1000 mẫu
            - Phân chia 80% (800 mẫu) cho training và 20% (200 mẫu) cho testing
            
            Để cải thiện độ chính xác khi sử dụng:
            - Vẽ chữ số rõ ràng, đậm và ở chính giữa
            - Chọn đúng hướng viết ở phần Radio button
            - Sử dụng kích thước bút vừa phải
            """
            
            info_label = Label(info_frame, text=accuracy_info, justify=tk.LEFT, 
                              font=("Arial", 12), bg="#e0e0e0", padx=10, pady=10)
            info_label.pack()
            
            # Load và hiển thị confusion matrix nếu có
            try:
                cm_frame = Frame(accuracy_window, bg="#e0e0e0", bd=2, relief="ridge", padx=20, pady=20)
                cm_frame.pack(fill=tk.X, padx=20, pady=20)
                
                cm_title = Label(cm_frame, text="Confusion Matrix từ đợt huấn luyện gần nhất:", 
                               font=("Arial", 14, "bold"), bg="#e0e0e0")
                cm_title.pack(pady=(0, 10))
                
                # Load và hiển thị confusion matrix từ file ảnh
                if os.path.exists('results/confusion_matrix.png'):
                    cm_image = tk.PhotoImage(file='results/confusion_matrix.png')
                    cm_label = Label(cm_frame, image=cm_image, bg="#e0e0e0")
                    cm_label.image = cm_image  # Keep a reference
                    cm_label.pack(pady=10)
                else:
                    Label(cm_frame, text="Không tìm thấy biểu đồ Confusion Matrix", 
                         font=("Arial", 12), bg="#e0e0e0").pack(pady=10)
            except Exception as e:
                print(f"Không thể hiển thị confusion matrix: {e}")
            
            # Close button
            close_button = Button(accuracy_window, text="Đóng", command=accuracy_window.destroy, 
                                 font=("Arial", 12, "bold"), bg="#F44336", fg="white", width=10)
            close_button.pack(pady=20)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị thông tin: {str(e)}")

def create_drawing_interface():
    try:
        root = tk.Tk()
        app = DigitRecognitionApp(root)
        root.mainloop()
        
        # Clean up temporary files
        if os.path.exists('temp_drawing.png'):
            os.remove('temp_drawing.png')
            
    except Exception as e:
        print(f"Lỗi khi tạo giao diện: {e}")

if __name__ == "__main__":
    create_drawing_interface()