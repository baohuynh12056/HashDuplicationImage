import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

# Đảm bảo import đúng class HashTable đã sửa lỗi (dùng map)
try:
    from MyHash import HashTable
except ImportError:
    print("LỖI: Không tìm thấy module MyHash. Hãy đảm bảo đã biên dịch C++.")
    exit()


def evaluate_sharpness(image_path):
    """
    Đánh giá độ sắc nét (Laplacian Variance). Giá trị càng cao càng nét.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.0
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
        if img is None:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 1])
    except Exception:
        return 0.0


def select_best_image_in_group(
    image_path1, image_path2, sharp_thresh=100.0, color_thresh=20.0
):
    """
    So sánh 2 ảnh và trả về ảnh tốt hơn.
    """
    sharp1 = evaluate_sharpness(image_path1)
    color1 = evaluate_colorfulness(image_path1)

    sharp2 = evaluate_sharpness(image_path2)
    color2 = evaluate_colorfulness(image_path2)

    good1 = sharp1 > sharp_thresh and color1 > color_thresh
    good2 = sharp2 > sharp_thresh and color2 > color_thresh

    if good1 and not good2:
        return image_path1, "Giữ ảnh 1 (ảnh 2 kém)"
    if good2 and not good1:
        return image_path2, "Giữ ảnh 2 (ảnh 1 kém)"

    # Cả 2 đều tốt HOẶC cả 2 đều tệ -> chọn ảnh nét hơn
    if sharp1 >= sharp2:
        return image_path1, "Giữ ảnh 1 (nét hơn)"
    else:
        return image_path2, "Giữ ảnh 2 (nét hơn)"


def build_clusters_simple(features, filenames, img_folder, num_planes=12):
    """
    Gom cụm (Clustering) đơn giản.
    Chỉ gom các ảnh có CHÍNH XÁC cùng 1 hash key.
    Không gom lân cận (No Multi-Probe).
    """

    CLUSTER_DIR = "clusters_simple"  # Thư mục output mới
    dimension = features.shape[1]

    if dimension != 512:
        print(f"CẢNH BÁO: Kích thước vector là {dimension}, không phải 512!")

    try:
        # Gọi hàm C++ đã sửa lỗi (2 tham số)
        ht = HashTable(num_planes, dimension)
    except Exception as e:
        print(f"LỖI KHỞI TẠO HashTable: {e}")
        return None

    # === BƯỚC 1: HASHING ===
    # hashtable = { bucket_id: [list of filenames] }
    hashtable = defaultdict(list)
    print(
        f"Bước 1: Bắt đầu thêm {len(features)} vector vào LSH (k={num_planes} planes)..."
    )

    for i, vec in enumerate(features):
        hash_key = ht.hashFunction(vec.tolist())
        img_path = os.path.join(img_folder, filenames[i])

        if os.path.exists(img_path):
            hashtable[hash_key].append(filenames[i])

    print(f"Hashing hoàn tất. Tìm thấy {len(hashtable)} bucket.")

    # === BƯỚC 2: LƯU ẢNH (Không có bước DSU) ===
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR, exist_ok=True)

    print(f"Bước 2: Bắt đầu lưu ảnh vào '{CLUSTER_DIR}/'...")

    cluster_count = 0
    for bucket_id, file_list in hashtable.items():
        if len(file_list) < 1:
            continue

        # Đặt tên thư mục theo bucket_id
        cluster_name = f"bucket_{bucket_id}"
        bucket_path = os.path.join(CLUSTER_DIR, cluster_name)
        os.makedirs(bucket_path, exist_ok=True)

        # Copy TẤT CẢ ảnh trong bucket
        for fname in file_list:
            src_path = os.path.join(img_folder, fname)
            if os.path.exists(src_path):
                shutil.copy(src_path, bucket_path)

        cluster_count += 1

    print(f"Đã lưu {cluster_count} cụm vào: '{CLUSTER_DIR}/'")
    return hashtable
