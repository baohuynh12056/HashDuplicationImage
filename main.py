import os
import numpy as np
from Application.restnet import mean_extract_image_features # <-- import trực tiếp từ file bạn có
from Application.normarlize import build_clusters,build_cluster_faiss  # <-- import trực tiếp từ file bạn có
# =============== Cấu hình cơ bản ===============
import hash_table_py as HashTable
import simhash_py as SimHash
import minhash_py as MinHash
import bloom_filter_py as BloomFilter
import os
from collections import defaultdict, Counter
import time

def evaluate_by_image(base_dir):
    label_to_clusters = defaultdict(lambda: defaultdict(int))
    total_images = 0

    # B1. Duyệt qua từng nhóm (cluster)
    for group_name in sorted(os.listdir(base_dir)):
        group_path = os.path.join(base_dir, group_name)
        if not os.path.isdir(group_path):
            continue

        for filename in os.listdir(group_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                total_images += 1
                label = filename.split("_")[0]
                label_to_clusters[label][group_name] += 1

    # B2. Tìm cluster chuẩn (cluster có nhiều ảnh nhất cho mỗi label)
    label_best_cluster = {
        label: max(cluster_counts.items(), key=lambda x: x[1])[0]
        for label, cluster_counts in label_to_clusters.items()
    }

    # B3. Đếm số ảnh đúng / sai
    correct = 0
    wrong = 0
    per_image_detail = []

    for group_name in sorted(os.listdir(base_dir)):
        group_path = os.path.join(base_dir, group_name)
        if not os.path.isdir(group_path):
            continue

        for filename in os.listdir(group_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                label = filename.split("_")[0]
                total_images += 0  # (đã tính ở trên)
                correct_cluster = label_best_cluster[label]

                if group_name == correct_cluster:
                    correct += 1
                    is_correct = True
                else:
                    wrong += 1
                    is_correct = False

                per_image_detail.append({
                    "filename": filename,
                    "label": label,
                    "group": group_name,
                    "correct_group": correct_cluster,
                    "is_correct": is_correct
                })

    accuracy = (correct / total_images * 100) if total_images else 0

    print("📊 Kết quả đánh giá clustering theo ẢNH:")
    print(f"   🖼️ Tổng số ảnh: {total_images}")
    print(f"   ✅ Ảnh đúng nhóm: {correct}")
    print(f"   ❌ Ảnh sai nhóm: {wrong}")
    print(f"   🎯 Độ chính xác: {accuracy:.2f}%\n")

    return per_image_detail, accuracy

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(BASE_DIR, "img")
    FEATURE_FILE = os.path.join(BASE_DIR, "features.npy")
    NAME_FILE = os.path.join(BASE_DIR, "filenames.npy")

    # Kiểm tra và tạo thư mục IMG_DIR nếu chưa có
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        print(f"Đã tạo thư mục {IMG_DIR}. Vui lòng thêm ảnh vào đó và chạy lại.")
    
    # Gọi hàm chính để thực thi
    features, filenames = mean_extract_image_features(
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

    print("Tải dữ liệu feature và filename...\n")
    features = np.load("features.npy")
    filenames = np.load("filenames.npy")

    print("Số ảnh:", len(filenames))
    
    start = time.time()
    build_cluster_faiss(features, filenames, IMG_DIR, "clusters5", threshold=0.75, K=10)  
    end = time.time()

    print(f"[FAISS] Thời gian chạy: {end - start:.2f}s\n")
    evaluate_by_image("clusters5")
    # Gọi hàm bạn muốn test:
    ht = HashTable.HashTable(32, features.shape[1])
    ht1 = BloomFilter.BloomFilter(36, features.shape[1],9)
    ht2 = SimHash.SimHash(128) 
    ht3 = MinHash.MinHash(128)

    start = time.time()
    hashtable = build_clusters(ht, features, filenames, IMG_DIR,5,"clusters")
    end = time.time()
    print(f"[HashTable] Thời gian chạy: {end - start:.10f}s\n")
    evaluate_by_image("clusters")

    start = time.time()
    bloomfilter = build_clusters(ht1, features, filenames, IMG_DIR,7,"clusters2")
    end = time.time()
    print(f"[BloomFilter] Thời gian chạy: {end - start:.10f}s\n")
    evaluate_by_image("clusters2")

    start = time.time()
    simhash = build_clusters(ht2,features, filenames, IMG_DIR,13,"clusters3")
    end = time.time()
    print(f"[SimHash] Thời gian chạy: {end - start:.2f}s\n")
    evaluate_by_image("clusters3")

    start = time.time()
    minhash = build_clusters(ht3, features, filenames, IMG_DIR,580,"clusters4")
    end = time.time()
    print(f"[MinHash] Thời gian chạy: {end - start:.2f}s\n")    
    evaluate_by_image("clusters4")
#     NUM_RUNS = 100

# acc_hash = []
# acc_bloom = []
# acc_simhash = []
# acc_minhash = []

# for i in range(NUM_RUNS):
#     print(f"\n========== 🔁 Lần chạy thứ {i+1}/{NUM_RUNS} ==========")

#     # --- 1. Khởi tạo lại từng cấu trúc hash ---
#     ht = HashTable.HashTable(32, features.shape[1])
#     ht1 = BloomFilter.BloomFilter(36, features.shape[1], 9)
#     ht2 = SimHash.SimHash(128)
#     ht3 = MinHash.MinHash(128)

#     # --- 2. Hashtable ---
#     hashtable = build_clusters(ht, features, filenames, IMG_DIR, 5, "clusters")
#     print("[INFO] Hashtable hoàn tất clustering.")
#     print(f"Tổng số bucket: {len(hashtable)}")
#     _, acc = evaluate_by_image("clusters")
#     acc_hash.append(acc)

#     # --- 3. Bloom Filter ---
#     bloomfilter = build_clusters(ht1, features, filenames, IMG_DIR, 7, "clusters2")
#     print("[INFO] Bloom Filter hoàn tất clustering.")
#     print(f"Tổng số bucket: {len(bloomfilter)}")
#     _, acc = evaluate_by_image("clusters2")
#     acc_bloom.append(acc)

#     # --- 4. SimHash ---
#     simhash = build_clusters(ht2, features, filenames, IMG_DIR, 13, "clusters3")
#     print("[INFO] SimHash hoàn tất clustering.")
#     print(f"Tổng số bucket: {len(simhash)}")
#     _, acc = evaluate_by_image("clusters3")
#     acc_simhash.append(acc)

#     # --- 5. MinHash ---
#     minhash = build_clusters(ht3, features, filenames, IMG_DIR, 580, "clusters4")
#     print("[INFO] MinHash hoàn tất clustering.")
#     print(f"Tổng số bucket: {len(minhash)}")
#     _, acc = evaluate_by_image("clusters4")
#     acc_minhash.append(acc)

# # --- 6. Tính trung bình accuracy ---
#     print("\n====================== 📊 KẾT QUẢ TRUNG BÌNH ======================")
#     print(f"Hashtable trung bình: {np.mean(acc_hash):.2f}% ± {np.std(acc_hash):.2f}")
#     print(f"Bloom Filter trung bình: {np.mean(acc_bloom):.2f}% ± {np.std(acc_bloom):.2f}")
#     print(f"SimHash trung bình: {np.mean(acc_simhash):.2f}% ± {np.std(acc_simhash):.2f}")
#     print(f"MinHash trung bình: {np.mean(acc_minhash):.2f}% ± {np.std(acc_minhash):.2f}")
