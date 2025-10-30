import os

import numpy as np

from Application.feature_extract import extract_features
from Application.normalize import build_clusters_simple  # Import hàm mới

# =============== Cấu hình cơ bản ===============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "dataset")  # Quét toàn bộ dataset

FEATURE_FILE = os.path.join(BASE_DIR, "features.npy")
NAME_FILE = os.path.join(BASE_DIR, "names.npy")
BATCH_SIZE = 64

# =============== Main chạy chính ===============
if __name__ == "__main__":

    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        print(f"Đã tạo thư mục {IMG_DIR}. Vui lòng thêm ảnh vào đó.")
        exit()

    # Bước 1: Trích xuất đặc trưng (Giữ nguyên)
    if not os.path.exists(FEATURE_FILE) or not os.path.exists(NAME_FILE):
        print(
            f"Không tìm thấy file đặc trưng, bắt đầu trích xuất từ {IMG_DIR}..."
        )
        features, filenames = extract_features(
            img_dir=IMG_DIR,
            feature_file=FEATURE_FILE,
            name_file=NAME_FILE,
            batch_size=BATCH_SIZE,
        )
    else:
        print(f"Đã tìm thấy file đặc trưng, đang tải {FEATURE_FILE}...")
        features = np.load(FEATURE_FILE)
        filenames = np.load(NAME_FILE)

    print("\n--- Hoàn tất trích xuất/tải đặc trưng ---")
    if len(features) == 0:
        print("Không có đặc trưng nào được xử lý.")
        exit()

    print(f"Tổng số đặc trưng đã load: {len(features)}")

    print("\n[INFO] Bắt đầu gom cụm (Clustering) với Multi-Probe LSH...")

    # Gọi hàm LSH (build_clusters_multiprobe)
    hashtable = build_clusters_simple(
        features,
        filenames,
        IMG_DIR,
        num_planes=12,
    )

    if hashtable:
        print("[INFO] Hoàn tất gom cụm.")
        print(f"Tổng số cụm (sau khi hợp nhất) được tìm thấy: {len(hashtable)}")
    else:
        print("[INFO] Lỗi trong quá trình gom cụm.")
