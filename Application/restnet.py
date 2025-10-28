import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


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

def extract_image_features(img_dir, feature_file, name_file, batch_size=64):
    """
    Trích xuất đặc trưng ảnh từ một thư mục, có kiểm tra và sử dụng file
    đã lưu nếu tồn tại.
    """
    
    # if os.path.exists(feature_file) and os.path.exists(name_file):
    #     print(f"Đã tìm thấy file lưu ({os.path.basename(feature_file)}), đang load...")
    #     try:
    #         all_features = np.load(feature_file)
    #         all_names = np.load(name_file)
    #         print("Load thành công.")
    #         return all_features, all_names
    #     except Exception as e:
    #         print(f"Lỗi khi load file, sẽ tiến hành trích xuất lại. Lỗi: {e}")

    print("Chưa có file lưu, đang trích xuất đặc trưng...")

    print("Đang tải model ResNet-50...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Bỏ lớp FC
    resnet.eval()
    resnet = resnet.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"Đang tạo DataLoader từ: {img_dir}")
    dataset = CustomImageDataset(img_dir, transform=transform)
    if len(dataset) == 0:
        print(f"Không tìm thấy ảnh nào trong thư mục: {img_dir}")
        return np.array([]), np.array([])
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []   # Danh sách chứa các vector numpy
    all_names = []      # Danh sách chứa tên file

    with torch.no_grad():
        for batch_imgs, batch_names in loader:
            batch_imgs = batch_imgs.to(device)

            feats = resnet(batch_imgs)              # (batch_size, 2048, 1, 1)
            feats = feats.view(feats.size(0), -1)   # (batch_size, 2048)

            all_features.extend(feats.cpu().numpy())
            all_names.extend(batch_names)

    all_features = np.array(all_features)
    all_names = np.array(all_names)

    try:
        np.save(feature_file, all_features)
        np.save(name_file, all_names)
        print(f"Đã lưu đặc trưng vào {os.path.basename(feature_file)} và {os.path.basename(name_file)}")
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")

    return all_features, all_names

def min_extract_image_features(img_dir, feature_file, name_file):
    # if os.path.exists(feature_file) and os.path.exists(name_file):
    #     features = np.load(feature_file)
    #     names = np.load(name_file)
    #     return features, names

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)
    transforms_list = [
        lambda x: x,
        lambda x: x.rotate(90, expand=True),
        lambda x: x.rotate(180, expand=True),
        lambda x: x.rotate(270, expand=True),
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
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

    for i, fname in enumerate(img_files, 1):
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
        final_feature = np.min(feature_array, axis=0)
        all_features.append(final_feature)
        all_names.append(fname)

    all_features = np.array(all_features)
    all_names = np.array(all_names)
    np.save(feature_file, all_features)
    np.save(name_file, all_names)
    return all_features, all_names

