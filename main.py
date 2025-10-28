import os
import numpy as np
from Application.restnet import min_extract_image_features  # <-- import trực tiếp từ file bạn có
from Application.normarlize import build_clusters, build_clusters_best  # <-- import trực tiếp từ file bạn có
from Application.simhash import build_clusters_simhash
# =============== Cấu hình cơ bản ===============


# =============== Main chạy chính ===============
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(BASE_DIR, "img")
    FEATURE_FILE = os.path.join(BASE_DIR, "features.npy")
    NAME_FILE = os.path.join(BASE_DIR, "filenames.npy")
    BATCH_SIZE = 8

    # Kiểm tra và tạo thư mục IMG_DIR nếu chưa có
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        print(f"Đã tạo thư mục {IMG_DIR}. Vui lòng thêm ảnh vào đó và chạy lại.")
    
    # Gọi hàm chính để thực thi
    features, filenames = min_extract_image_features(
        img_dir=IMG_DIR,
        feature_file=FEATURE_FILE,
        name_file=NAME_FILE,
    )

    print("\n--- Hoàn tất ---")
    if len(features) > 0:
        print(f"Tổng số đặc trưng đã load/trích xuất: {len(features)}")
        print(f"Kích thước vector đặc trưng đầu tiên: {features[0].shape}")
        print(f"Tên file đầu tiên: {filenames[0]}")
    else:
        print("Không có đặc trưng nào được xử lý.")

    print("[INFO] Đang tải dữ liệu feature và filename...")
    features = np.load("features.npy")
    filenames = np.load("filenames.npy")

    print("Số ảnh:", len(filenames))

    # Gọi hàm bạn muốn test:
    # hashtable = build_clusters(features, filenames, IMG_DIR)
    hashtable = build_clusters_best(features, filenames, IMG_DIR)
    for i in range(min(2, len(features))):
        print(f"Feature vector {i}:", features[i])

    print("[INFO] Hoàn tất clustering.")
    print(f"Tổng số bucket được tạo: {len(hashtable)}")
