# 🌾 Phân loại Bệnh trên Lá Lúa (Rice Leaf Disease Classification) với Swin Transformer

Dự án này là giải pháp học sâu (Deep Learning) sử dụng kiến trúc **Swin Transformer** để phân loại các loại bệnh phổ biến trên lá lúa. Mã nguồn được tối ưu hóa đặc biệt để chạy trên môi trường Kaggle Notebook, tích hợp nhiều kỹ thuật tiên tiến giúp mô hình chống hiện tượng học vẹt (overfitting) và tối đa hóa độ chính xác.

---

## ✨ Tính năng nổi bật (Features)

* **Kiến trúc State-of-the-Art:** Sử dụng `swin_small_patch4_window7_224` từ thư viện `timm`, mang lại sự cân bằng hoàn hảo giữa hiệu năng và tài nguyên.
* **Chiến lược Huấn luyện Tiên tiến:**
  * **5-Fold Cross Validation:** Đảm bảo mô hình học được toàn bộ dữ liệu, đánh giá khách quan và kết hợp (ensemble) ở bước dự đoán.
  * **Mixup Augmentation:** Trộn các bức ảnh và nhãn lại với nhau trong quá trình huấn luyện, giúp biên giới phân chia giữa các lớp (classes) mềm mại hơn.
  * **Label Smoothing & Balanced Class Weights:** Xử lý triệt để vấn đề mất cân bằng dữ liệu (imbalanced data).
* **Xử lý Dữ liệu/Nhiễu (Albumentations):**
  * Giả lập nhiễu sọc ngang/dọc (`GridDropout`, `CoarseDropout`) để mô hình "miễn nhiễm" với các đường sọc trong tập test.
  * Xử lý nhiễu hạt (`GaussianBlur`, `ISONoise`).
* **Test Time Augmentation (TTA):** Tạo 4 góc nhìn khác nhau cho mỗi ảnh ở pha suy luận (inference), sau đó lấy trung bình dự đoán từ 5 mô hình (Ensemble 5-Fold) để đưa ra kết quả cuối cùng chắc chắn nhất.

---

## 📂 Cấu trúc Mã Nguồn

* **`CFG (Configuration)`**: Lớp chứa toàn bộ các siêu tham số (hyperparameters) như `batch_size`, `lr`, `epochs`, `model_name`, `seed`... Giúp bạn dễ dàng thay đổi cấu hình chỉ tại một nơi duy nhất.
* **`get_transforms`**: Định nghĩa các phép biến đổi ảnh cho tập Train (có augmentations) và tập Val/Test (chỉ resize và normalize).
* **`RiceDataset`**: Lớp quản lý và nạp dữ liệu hình ảnh tùy chỉnh theo chuẩn PyTorch.
* **`create_model`**: Khởi tạo Swin Transformer với `drop_rate` và `drop_path_rate` được điều chỉnh để tăng tính ổn định.
* **`Training Loop`**: Vòng lặp huấn luyện chính với Optimizer `AdamW`, Scheduler `CosineAnnealingLR` và kỹ thuật Gradient Clipping.
* **`Inference & Submission`**: Tải trọng số tốt nhất, thực hiện TTA 4-views và xuất ra file `submission.csv`.

---

## 🚀 Hướng dẫn chạy trên Kaggle

### 1. Chuẩn bị Môi trường
1. Tạo một Notebook mới tại trang chủ cuộc thi: `fptu-can-tho-olympic-ai-2026`.
2. Truy cập **Settings** (ở cột bên phải màn hình) -> Mục **Accelerator**, chọn **GPU P100** hoặc **GPU T4 x2**.
3. Đảm bảo mục **Internet** được bật (On) để tải pre-trained weights.

### 2. Cài đặt Thư viện
Mở một ô code (cell) và cài đặt thư viện PyTorch Image Models (`timm`):
```bash
!pip install timm
