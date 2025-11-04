import os
import random
import time

import cv2
import kornia.augmentation as K
import kornia.filters as KF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader
from tqdm import tqdm


class AddGaussianNoiseGPU:
    """Thêm nhiễu Gaussian trên GPU"""

    def __init__(self, std=0.05, prob=1.0):
        self.std = std
        self.prob = prob

    def __call__(self, tensor):
        if torch.rand(1) < self.prob:
            noise = torch.randn_like(tensor) * self.std
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor


# --- Làm mờ nhẹ (denoise giả lập) ---
class GaussianBlurGPU:
    """Làm mờ nhẹ bằng Gaussian kernel (tương đương ImageFilter.GaussianBlur)"""

    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, tensor):
        # Tạo kernel Gaussian 2D
        device = tensor.device
        ax = torch.arange(
            -self.kernel_size // 2 + 1.0,
            self.kernel_size // 2 + 1.0,
            device=device,
        )
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.to(tensor.dtype)

        # Áp dụng trên từng kênh RGB
        channels = []
        for c in range(tensor.shape[0]):
            ch = tensor[c : c + 1].unsqueeze(0)
            ch = F.conv2d(ch, kernel, padding=self.kernel_size // 2)
            channels.append(ch)
        return torch.cat(channels, dim=1).squeeze(0)


# --- Làm mịn ảnh (smooth edges) ---
class SmoothImageGPU:
    """Làm mịn bằng trung bình kernel (tương tự ImageFilter.SMOOTH_MORE)"""

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, tensor):
        device = tensor.device
        kernel = torch.ones(
            (1, 1, self.kernel_size, self.kernel_size), device=device
        )
        kernel = kernel / kernel.sum()
        kernel = kernel.to(tensor.dtype)

        # Áp dụng trên từng kênh RGB
        channels = []
        for c in range(tensor.shape[0]):
            ch = tensor[c : c + 1].unsqueeze(0)
            ch = F.conv2d(ch, kernel, padding=self.kernel_size // 2)
            channels.append(ch)
        return torch.cat(channels, dim=1).squeeze(0)


# --- Denoise thích nghi (Adaptive Gaussian Blur) ---
class AdaptiveDenoiseGPU:
    """Giảm nhiễu khi phương sai ảnh vượt ngưỡng (tương tự adaptive_denoise)"""

    def __init__(self, var_threshold=0.03, sigma=1.0):
        self.var_threshold = var_threshold
        self.sigma = sigma

    def __call__(self, tensor):
        var = tensor.var()
        if var > self.var_threshold:
            blur = GaussianBlurGPU(kernel_size=3, sigma=self.sigma)
            tensor = blur(tensor)
        return tensor


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
    var_threshold=3000,
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
        self.filenames = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith(valid_extensions)
        ]
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
        lambda x: x.crop((224 * 0.2, 224 * 0.2, 224 * 0.8, 224 * 0.8)).resize(
            (224, 224)
        ),
        lambda x: x.crop((224 * 0.1, 224 * 0.1, 224 * 0.9, 224 * 0.9)).resize(
            (224, 224)
        ),
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
        lambda x: x.filter(ImageFilter.MedianFilter(size=3)),  # lọc trung vị
        lambda x: x.filter(ImageFilter.BoxBlur(radius=1)),  # làm mờ trung bình
        # --- Lọc nhiễu (denoise) & làm mịn ---
        lambda x: denoise_image(x),
        lambda x: smooth_image(x),
        # --- Thêm nhiễu Gaussian ---
        lambda x: add_gaussian_noise(x, std=5),
        lambda x: add_gaussian_noise(x, std=10),
    ]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    all_features = []
    all_names = []
    img_files = [
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for i, fname in enumerate(tqdm(img_files, desc="Extracting features"), 1):
        try:
            img = (
                Image.open(os.path.join(img_dir, fname))
                .convert("L")
                .convert("RGB")
            )
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


def mean_extract_image_features_batch(
    img_dir, feature_file, name_file, batch_size=24
):
    """
    Phiên bản tối ưu: Trích đặc trưng ảnh với augment, gom vào batch để chạy ResNet nhanh hơn.
    """
    print(f"\nĐang trích xuất đặc trưng theo batch từ thư mục: {img_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # --- Load model ---
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Bỏ lớp FC
    resnet.eval().to(device)

    # --- Chuẩn hóa ảnh sau augment ---
    normalize_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            # --- Các augment GPU custom ---
            AddGaussianNoiseGPU(std=0.05, prob=0.5),
            GaussianBlurGPU(kernel_size=3, sigma=1.0),
            SmoothImageGPU(kernel_size=3),
            AdaptiveDenoiseGPU(var_threshold=0.03, sigma=1.0),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # --- Danh sách augment ---
    transforms_list = [
        lambda x: x,
        lambda x: x.rotate(90, expand=True),
        lambda x: x.rotate(180, expand=True),
        lambda x: x.rotate(270, expand=True),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(15, expand=False, fillcolor="black"),
        lambda x: x.rotate(-15, expand=False, fillcolor="black"),
        lambda x: x.rotate(45, expand=True, fillcolor="black"),
        lambda x: x.rotate(-45, expand=True, fillcolor="black"),
        lambda x: x.crop((224 * 0.2, 224 * 0.2, 224 * 0.8, 224 * 0.8)).resize(
            (224, 224)
        ),
        lambda x: x.crop((224 * 0.1, 224 * 0.1, 224 * 0.9, 224 * 0.9)).resize(
            (224, 224)
        ),
        lambda x: ImageEnhance.Brightness(x).enhance(0.5),
        lambda x: ImageEnhance.Brightness(x).enhance(1.5),
        lambda x: ImageEnhance.Contrast(x).enhance(0.7),
        lambda x: ImageEnhance.Contrast(x).enhance(1.5),
        lambda x: ImageEnhance.Sharpness(x).enhance(0.3),
        lambda x: ImageEnhance.Sharpness(x).enhance(2.0),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=2)),
        lambda x: x.filter(ImageFilter.MedianFilter(size=3)),
        lambda x: x.filter(ImageFilter.BoxBlur(radius=1)),
        lambda x: denoise_image(x),
        lambda x: smooth_image(x),
        lambda x: add_gaussian_noise(x, std=5),
        lambda x: add_gaussian_noise(x, std=10),
    ]

    # --- Chuẩn bị danh sách ảnh ---
    img_files = [
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not img_files:
        print(" Không tìm thấy ảnh hợp lệ trong thư mục.")
        return np.array([]), np.array([])

    all_features = []
    all_names = []

    print(f" Tổng số ảnh: {len(img_files)}")
    print(f"  Batch size: {batch_size}, Tổng augment: {len(transforms_list)}")

    # --- Vòng lặp chính ---
    with torch.no_grad():
        for i, fname in enumerate(
            tqdm(img_files, desc="Extracting features"), 1
        ):
            try:
                img_path = os.path.join(img_dir, fname)
                img = Image.open(img_path).convert("L").convert("RGB")
            except Exception:
                print(f" Lỗi khi đọc {fname}, bỏ qua.")
                continue

            # Tạo batch augment cho 1 ảnh
            batch_tensors = []
            t0 = time.time()
            for fn in transforms_list:
                aug_img = fn(img)
                tensor_img = normalize_tf(aug_img)
                batch_tensors.append(tensor_img)
            print("Aug time:", time.time() - t0)
            # Gom thành tensor batch
            start = time.time()
            batch_tensor = torch.stack(batch_tensors).to(device)
            print("→ CPU to GPU:", time.time() - start)

            # Chia nhỏ batch nếu GPU 8GB không đủ
            feats_all = []
            for j in range(0, len(batch_tensor), batch_size):
                sub_batch = batch_tensor[j : j + batch_size]
                feats = resnet(sub_batch)
                feats = feats.view(feats.size(0), -1)  # (b, 2048)
                feats_all.append(feats.cpu().numpy())

            feature_array = np.vstack(feats_all)
            final_feature = np.mean(feature_array, axis=0)

            all_features.append(final_feature)
            all_names.append(fname)

            if i % 10 == 0 or i == len(img_files):
                print(f" Đã xử lý {i}/{len(img_files)} ảnh...")

    # --- Lưu kết quả ---
    all_features = np.array(all_features)
    all_names = np.array(all_names)

    try:
        np.save(feature_file, all_features)
        np.save(name_file, all_names)
        print(f"\n Đã lưu đặc trưng vào:\n - {feature_file}\n - {name_file}")
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")

    print("Hoàn tất trích xuất đặc trưng (batch).")
    return all_features, all_names


def mean_extract_image_features_batch_1(
    img_dir, feature_file, name_file, batch_size=24
):
    """
    Trích xuất đặc trưng ảnh bằng ResNet50 + augment Kornia GPU + gom batch (24 augment mỗi ảnh).
    """
    if os.path.exists(feature_file) and os.path.exists(name_file):
        features = np.load(feature_file)
        names = np.load(name_file)
        return features, names
    print(f"\n Đang trích xuất đặc trưng từ: {img_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)

    # --- ResNet50 (bỏ FC) ---
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)

    # --- Chuẩn hóa theo ImageNet ---
    normalize = K.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]).to(device),
        std=torch.tensor([0.229, 0.224, 0.225]).to(device),
    )

    # --- Các augment GPU tương tự transforms_list ---
    kornia_augments = [
        # --- Rotation cố định ---
        K.RandomRotation(degrees=(0.0, 0.0), p=1.0),  # gốc
        K.RandomRotation(degrees=(90.0, 90.0), p=1.0),
        K.RandomRotation(degrees=(180.0, 180.0), p=1.0),
        K.RandomRotation(degrees=(270.0, 270.0), p=1.0),
        K.RandomRotation(degrees=(15.0, 15.0), p=1.0),
        K.RandomRotation(degrees=(-15.0, -15.0), p=1.0),
        K.RandomRotation(degrees=(45.0, 45.0), p=1.0),
        K.RandomRotation(degrees=(-45.0, -45.0), p=1.0),
        # --- Flip cố định ---
        K.RandomHorizontalFlip(p=1.0),
        K.RandomVerticalFlip(p=1.0),
        # --- Crop / Resize cố định ---
        K.RandomResizedCrop(
            size=(224, 224), scale=(0.8, 0.8), ratio=(1.0, 1.0), p=1.0
        ),
        K.RandomResizedCrop(
            size=(224, 224), scale=(0.9, 0.9), ratio=(1.0, 1.0), p=1.0
        ),
        # --- Brightness / Contrast cố định ---
        K.ColorJitter(brightness=0.5, contrast=0.0, p=1.0),
        K.ColorJitter(brightness=1.5, contrast=0.0, p=1.0),
        K.ColorJitter(brightness=0.0, contrast=0.7, p=1.0),
        K.ColorJitter(brightness=0.0, contrast=1.5, p=1.0),
        # --- Sharpness cố định ---
        K.RandomSharpness(sharpness=0.3, p=1.0),
        K.RandomSharpness(sharpness=2.0, p=1.0),
        # --- Blur / Denoise ---
        lambda x: KF.gaussian_blur2d(x, (3, 3), (2.0, 2.0)),
        lambda x: KF.median_blur(x, (3, 3)),
        lambda x: KF.box_blur(x, (3, 3)),
        lambda x: KF.gaussian_blur2d(x, (3, 3), (0.5, 0.5)),  # denoise nhẹ
        lambda x: KF.gaussian_blur2d(x, (5, 5), (1.0, 1.0)),  # smooth
        # --- Noise ---
        lambda x: x + 0.02 * torch.randn_like(x),
        lambda x: x + 0.05 * torch.randn_like(x),
    ]

    print(f"Tổng augment: {len(kornia_augments)}")

    # --- Danh sách ảnh ---
    img_files = []
    # Duyệt đệ quy qua tất cả các thư mục con bên trong img_dir
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):

                full_path = os.path.join(dirpath, f)

                relative_path = os.path.relpath(full_path, img_dir)

                img_files.append(relative_path)

    if not img_files:
        print(" Không tìm thấy ảnh trong thư mục hoặc các thư mục con.")
        return np.array([]), np.array([])

    all_features, all_names = [], []

    with torch.no_grad():
        for i, fname in enumerate(
            tqdm(img_files, desc="Extracting features"), 1
        ):
            try:
                img_path = os.path.join(img_dir, fname)
                img = Image.open(img_path).convert("L").convert("RGB")
                img_tensor = (
                    transforms.ToTensor()(img).unsqueeze(0).to(device)
                )  # (1,3,H,W)
                img_tensor = transforms.Resize((224, 224))(img_tensor)
            except Exception as e:
                print(f"Lỗi đọc ảnh {fname}: {e}")
                continue

            # --- Tạo batch augment ---
            aug_batch = []
            for aug in kornia_augments:
                out = aug(img_tensor) if callable(aug) else aug(img_tensor)
                aug_batch.append(out)
            aug_batch = torch.cat(aug_batch, dim=0)  # (24,3,224,224)
            aug_batch = normalize(aug_batch)

            # --- Chia batch nhỏ nếu cần ---
            feats_list = []
            for j in range(0, len(aug_batch), batch_size):
                sub_batch = aug_batch[j : j + batch_size]
                feats = resnet(sub_batch)
                feats = feats.view(feats.size(0), -1)
                feats_list.append(feats.cpu())
            feats_all = torch.cat(feats_list, dim=0)

            # --- Lấy mean ---
            mean_feat = feats_all.mean(dim=0).numpy()
            all_features.append(mean_feat)
            all_names.append(fname)

            if i % 10 == 0 or i == len(img_files):
                print(f"Đã xử lý {i}/{len(img_files)} ảnh...")

    # --- Lưu ---
    all_features = np.array(all_features)
    all_names = np.array(all_names)

    np.save(feature_file, all_features)
    np.save(name_file, all_names)

    print(f"\n Lưu xong: {feature_file}, {name_file}")
    print("Hoàn tất trích xuất đặc trưng GPU (batch).")
    return all_features, all_names
