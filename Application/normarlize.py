import os
import shutil
import numpy as np
import cv2
from collections import defaultdict
import hash_table_py as HashTable
import simhash_py as SimHash
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
NUM_BUCKETS = 10  # số bucket trong hash table
CLUSTER_DIR = "clusters"

def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def normalize(features):
    """Chuẩn hóa vector về [0,1]."""
    min_val = features.min(axis=0)
    max_val = features.max(axis=0)
    norm = (features - min_val) / (max_val - min_val + 1e-10)
    return norm

def build_clusters(features, filenames, img_folder):
    """Xây dựng cluster từ feature vectors và lưu ảnh theo bucket."""
    hashtable = defaultdict(list)
    ht = SimHash.SimHash(16)
    normalized_features = l2_normalize(features)
    print("[INFO] Bắt đầu thêm vector vào HashTable...")
    for i, vec in enumerate(normalized_features):
         # ép sang list nếu C++ binding yêu cầu
        hash_key = ht.hashFunction(vec.tolist())
        print(hash_key,filenames[i])
        hashtable[hash_key].append(filenames[i])

    # Xóa thư mục cũ nếu có
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR)

    print("Bắt đầu lưu ảnh vào các cụm...")
    for bucket_id, img_list in hashtable.items():
        cluster_path = os.path.join(CLUSTER_DIR, f"bucket_{bucket_id}")
        os.makedirs(cluster_path, exist_ok=True)
        if not img_list:
            continue   
        for fname in img_list:
            src = os.path.join(img_folder, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(cluster_path, fname))

    print(f"Đã tạo {len(hashtable)} bucket trong thư mục '{CLUSTER_DIR}/'")
    return hashtable

def evaluate_sharpness(image_path):
    """
    Đánh giá độ sắc nét (Laplacian Variance). Giá trị càng cao càng nét.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 0.0

def evaluate_colorfulness(image_path):
    """
    Đánh giá mức độ màu sắc (Mean Saturation). Giá trị càng cao càng nhiều màu.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Lấy giá trị trung bình của kênh Saturation (Độ bão hòa)
        return np.mean(hsv[:, :, 1])
    except Exception:
        return 0.0

def select_best_image_in_group(image_path1, image_path2, sharp_thresh=100.0, color_thresh=20.0):
    """
    So sánh 2 ảnh và trả về ảnh tốt hơn.
    - Ảnh "tốt" là ảnh có độ sắc nét cao và màu sắc tốt.
    - Nếu cả 2 đều không đạt ngưỡng, chọn ảnh sắc nét hơn.
    """
    sharp1 = evaluate_sharpness(image_path1)
    color1 = evaluate_colorfulness(image_path1)

    sharp2 = evaluate_sharpness(image_path2)
    color2 = evaluate_colorfulness(image_path2)

    good1 = (sharp1 > sharp_thresh and color1 > color_thresh)
    good2 = (sharp2 > sharp_thresh and color2 > color_thresh)

    if good1 and good2:
        return (image_path1, "Giữ ảnh 1") if sharp1 >= sharp2 else (image_path2, "Giữ ảnh 2")
    elif good1:
        return image_path1, "Giữ ảnh 1 (ảnh 2 kém màu/sắc)"
    elif good2:
        return image_path2, "Giữ ảnh 2 (ảnh 1 kém màu/sắc)"
    else:
        # Cả 2 đều không đạt ngưỡng → chọn ảnh sắc nét hơn
        return (image_path1, "Giữ ảnh 1 (fallback)") if sharp1 >= sharp2 else (image_path2, "Giữ ảnh 2 (fallback)")
        
def build_clusters_best(features, filenames, img_folder):
    """Xây dựng cluster từ feature vectors và lưu ảnh theo bucket."""
    hashtable = defaultdict(list)
    ht = HashTable.HashTable(NUM_BUCKETS, 8, features.shape[1])

    print("Bắt đầu thêm vector vào HashTable...",flush=True)
    for i, vec in enumerate(features): 
        hash_key = ht.hashFunction(vec.tolist()) 
        print(hash_key)
        img_path = os.path.join(img_folder, filenames[i])
        if not os.path.exists(img_path):
            continue

        # Đánh giá ảnh mới
        sharpness = evaluate_sharpness(img_path)
        color_score = evaluate_colorfulness(img_path)

        if hash_key not in hashtable:
            # Chưa có ảnh nào trong bucket này
            hashtable[hash_key] = {
                'path': img_path,
                'sharpness': sharpness,
                'color': color_score
            }
        else:
            # So sánh với ảnh hiện có
            current = hashtable[hash_key]
            better_path, _ = select_best_image_in_group(
                current['path'], img_path
            )
            # Nếu ảnh mới tốt hơn → thay thế
            if better_path == img_path:
                hashtable[hash_key] = {
                    'path': img_path,
                    'sharpness': sharpness,
                    'color': color_score
                }

    # Xóa thư mục cũ và tạo mới
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR, exist_ok=True)

    print("Bắt đầu lưu ảnh tốt nhất vào từng bucket...")
    for bucket_id, info in hashtable.items():
        cluster_path = os.path.join(CLUSTER_DIR, f"bucket_{bucket_id}")
        os.makedirs(cluster_path, exist_ok=True)
        shutil.copy(info['path'], cluster_path)
        print(info['path'])

    print(f" Đã tạo {len(hashtable)} bucket, mỗi bucket chứa 1 ảnh tốt nhất.")
    return hashtable
        
if __name__ == "__main__":
    features = np.load("features.npy")
    filenames = np.load("filenames.npy")

    IMG_DIR = "img"  # thư mục chứa ảnh gốc

    print("Số ảnh:", len(filenames))
    hashtable = build_clusters(features, filenames, IMG_DIR)
