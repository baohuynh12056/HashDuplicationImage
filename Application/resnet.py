import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import random
import cv2
from torch.utils.data import DataLoader
from PIL import Image,ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm

def add_random_noise(img_pil, noise_prob=0.4, noise_std=20):
    """Thêm nhiễu Gaussian"""
    if random.random() < noise_prob:
        np_img = np.array(img_pil).astype(np.float32)
        noise = np.random.normal(0, noise_std, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    return img_pil

def adaptive_denoise(img_pil, var_threshold=3000):
    """Làm mờ nhẹ nếu ảnh nhiễu mạnh"""
    np_img = np.array(img_pil)
    if np_img.var() > var_threshold:
        np_img = cv2.GaussianBlur(np_img, (3, 3), 0)
        return Image.fromarray(np_img)
    return img_pil
def add_gaussian_noise(img, std=10):
    """Thêm nhiễu Gaussian."""
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def denoise_image(img):
    """Lọc nhiễu đơn giản bằng Gaussian blur nhẹ (giả lập denoise)."""
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))

def smooth_image(img):
    """Làm mịn ảnh (reduce sharp edges)."""
    return img.filter(ImageFilter.SMOOTH_MORE)
def advanced_preprocess_pil(
    img_pil,
    size=(224, 224),
    brightness=1.2,
    contrast=1.2,
    noise_prob=0.4,
    noise_std=20,
    var_threshold=3000
):
    """
    Tiền xử lý nâng cao cho ảnh đầu vào (chuẩn bị trước khi trích xuất feature)
    """
    img_pil = add_random_noise(img_pil, noise_prob, noise_std)
    img_pil = adaptive_denoise(img_pil, var_threshold)
    img_pil = img_pil.resize(size)
    img_bright = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img_contrast = ImageEnhance.Contrast(img_bright).enhance(contrast)

    return img_contrast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng device: {device}")

class CustomImageDataset(torch.utils.data.Dataset):
    """
    Dataset tùy chỉnh để đọc ảnh từ một thư mục.
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        valid_extensions = (".jpg", ".jpeg", ".png")
        self.filenames = [f for f in os.listdir(folder)
                          if f.lower().endswith(valid_extensions)]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.filenames[idx])
        try:
            image = Image.open(img_path).convert("L").convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.filenames[idx]
        except Exception as e:
            print(f"Lỗi khi đọc file {img_path}: {e}")
            return torch.zeros((3, 224, 224)), self.filenames[idx]

def mean_extract_image_features(img_dir, feature_file, name_file):
    if os.path.exists(feature_file) and os.path.exists(name_file):
        features = np.load(feature_file)
        names = np.load(name_file)
        return features, names

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)
    transforms_list = [
    # --- Ảnh gốc & xoay ---
    lambda x: x,
    lambda x: x.rotate(90, expand=True),
    lambda x: x.rotate(180, expand=True),
    lambda x: x.rotate(270, expand=True),

    # --- Lật ---
    lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),

    # --- Xoay mạnh ---
    lambda x: x.rotate(15, expand=False, fillcolor="black"),
    lambda x: x.rotate(-15, expand=False, fillcolor="black"),
    lambda x: x.rotate(45, expand=True, fillcolor="black"),
    lambda x: x.rotate(-45, expand=True, fillcolor="black"),

    # # --- Crop / Zoom ---
    lambda x: x.crop((224*0.2, 224*0.2, 224*0.8, 224*0.8)).resize((224, 224)),
    lambda x: x.crop((224*0.1, 224*0.1, 224*0.9, 224*0.9)).resize((224, 224)),

    # --- Sáng / Tối ---
    lambda x: ImageEnhance.Brightness(x).enhance(0.5),
    lambda x: ImageEnhance.Brightness(x).enhance(1.5),

    # --- Tương phản ---
    lambda x: ImageEnhance.Contrast(x).enhance(0.7),
    lambda x: ImageEnhance.Contrast(x).enhance(1.5),

    # --- Độ sắc nét (sharpness) ---
    lambda x: ImageEnhance.Sharpness(x).enhance(0.3),  # mờ bớt
    lambda x: ImageEnhance.Sharpness(x).enhance(2.0),  # sắc nét hơn

    # --- Làm mờ / lọc mờ ---
    lambda x: x.filter(ImageFilter.GaussianBlur(radius=2)),
    lambda x: x.filter(ImageFilter.MedianFilter(size=3)),   # lọc trung vị
    lambda x: x.filter(ImageFilter.BoxBlur(radius=1)),      # làm mờ trung bình

    # --- Lọc nhiễu (denoise) & làm mịn ---
    lambda x: denoise_image(x),
    lambda x: smooth_image(x),

    # --- Thêm nhiễu Gaussian ---
    lambda x: add_gaussian_noise(x, std=5),
    lambda x: add_gaussian_noise(x, std=10),
    ]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])   
    all_features = []
    all_names = []
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, fname in enumerate(tqdm(img_files, desc="Extracting features"), 1):
        try:
            img = Image.open(os.path.join(img_dir, fname)).convert("L").convert("RGB")
        except Exception:
            continue

        feature_list = []
        with torch.no_grad():
            for fn in transforms_list:
                aug_img = fn(img)
                tensor_img = transform(aug_img).unsqueeze(0).to(device)
                feats = resnet(tensor_img)
                feats = feats.view(feats.size(0), -1)
                feature_list.append(feats.cpu().numpy())

        feature_array = np.vstack(feature_list)
        final_feature = np.mean(feature_array, axis=0)
        all_features.append(final_feature)
        all_names.append(fname)

    all_features = np.array(all_features)
    all_names = np.array(all_names)
    np.save(feature_file, all_features)
    np.save(name_file, all_names)
    return all_features, all_names


# def mean_extract_image_features_batch(img_dir, feature_file, name_file, batch_size=32):
#     """
#     Phiên bản tối ưu: Trích đặc trưng ảnh với augment, gom vào batch để chạy ResNet nhanh hơn.
#     """
#     print(f"\nĐang trích xuất đặc trưng theo batch từ thư mục: {img_dir}")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)

#     # --- Load model ---
#     resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#     resnet = nn.Sequential(*list(resnet.children())[:-1])  # Bỏ lớp FC
#     resnet.eval().to(device)

#     # --- Chuẩn hóa ảnh sau augment ---
#     normalize_tf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # --- Danh sách augment ---
#     transforms_list = [
#         lambda x: x,
#         lambda x: x.rotate(90, expand=True),
#         lambda x: x.rotate(180, expand=True),
#         lambda x: x.rotate(270, expand=True),
#         lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
#         lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
#         lambda x: x.rotate(15, expand=False, fillcolor="black"),
#         lambda x: x.rotate(-15, expand=False, fillcolor="black"),
#         lambda x: x.rotate(45, expand=True, fillcolor="black"),
#         lambda x: x.rotate(-45, expand=True, fillcolor="black"),
#         lambda x: x.crop((224*0.2, 224*0.2, 224*0.8, 224*0.8)).resize((224, 224)),
#         lambda x: x.crop((224*0.1, 224*0.1, 224*0.9, 224*0.9)).resize((224, 224)),
#         lambda x: ImageEnhance.Brightness(x).enhance(0.5),
#         lambda x: ImageEnhance.Brightness(x).enhance(1.5),
#         lambda x: ImageEnhance.Contrast(x).enhance(0.7),
#         lambda x: ImageEnhance.Contrast(x).enhance(1.5),
#         lambda x: ImageEnhance.Sharpness(x).enhance(0.3),
#         lambda x: ImageEnhance.Sharpness(x).enhance(2.0),
#         lambda x: x.filter(ImageFilter.GaussianBlur(radius=2)),
#         lambda x: x.filter(ImageFilter.MedianFilter(size=3)),
#         lambda x: x.filter(ImageFilter.BoxBlur(radius=1)),
#         lambda x: denoise_image(x),
#         lambda x: smooth_image(x),
#         lambda x: add_gaussian_noise(x, std=5),
#         lambda x: add_gaussian_noise(x, std=10),
#     ]

#     # --- Chuẩn bị danh sách ảnh ---
#     img_files = [f for f in os.listdir(img_dir)
#                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     if not img_files:
#         print(" Không tìm thấy ảnh hợp lệ trong thư mục.")
#         return np.array([]), np.array([])

#     all_features = []
#     all_names = []

#     print(f" Tổng số ảnh: {len(img_files)}")
#     print(f"  Batch size: {batch_size}, Tổng augment: {len(transforms_list)}")

#     # --- Vòng lặp chính ---
#     with torch.no_grad():
#         for i, fname in enumerate(tqdm(img_files, desc="Extracting features"), 1):
#             try:
#                 img_path = os.path.join(img_dir, fname)
#                 img = Image.open(img_path).convert("L").convert("RGB")
#             except Exception:
#                 print(f" Lỗi khi đọc {fname}, bỏ qua.")
#                 continue

#             # Tạo batch augment cho 1 ảnh
#             batch_tensors = []
#             for fn in transforms_list:
#                 aug_img = fn(img)
#                 tensor_img = normalize_tf(aug_img)
#                 batch_tensors.append(tensor_img)

#             # Gom thành tensor batch
#             batch_tensor = torch.stack(batch_tensors).to(device)

#             # Chia nhỏ batch nếu GPU 8GB không đủ
#             feats_all = []
#             for j in range(0, len(batch_tensor), batch_size):
#                 sub_batch = batch_tensor[j:j + batch_size]
#                 feats = resnet(sub_batch)
#                 feats = feats.view(feats.size(0), -1)  # (b, 2048)
#                 feats_all.append(feats.cpu().numpy())

#             feature_array = np.vstack(feats_all)
#             final_feature = np.mean(feature_array, axis=0)

#             all_features.append(final_feature)
#             all_names.append(fname)

#             if i % 10 == 0 or i == len(img_files):
#                 print(f" Đã xử lý {i}/{len(img_files)} ảnh...")

#     # --- Lưu kết quả ---
#     all_features = np.array(all_features)
#     all_names = np.array(all_names)

#     try:
#         np.save(feature_file, all_features)
#         np.save(name_file, all_names)
#         print(f"\n Đã lưu đặc trưng vào:\n - {feature_file}\n - {name_file}")
#     except Exception as e:
#         print(f"Lỗi khi lưu file: {e}")

#     print("Hoàn tất trích xuất đặc trưng (batch).")
#     return all_features, all_names
