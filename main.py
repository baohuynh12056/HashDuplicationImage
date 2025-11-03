import os
import numpy as np
from Application.resnet import mean_extract_image_features,mean_extract_image_features_batch_1 # <-- import trực tiếp từ file bạn có
from Application.cluster import build_clusters,build_cluster_faiss  # <-- import trực tiếp từ file bạn có
# =============== Cấu hình cơ bản ===============
import hash_table_py as HashTable
import simhash_py as SimHash
import minhash_py as MinHash
import bloom_filter_py as BloomFilter
import os
from collections import defaultdict, Counter
import time
from statistics import mean
from tqdm import tqdm


def evaluate_by_image(base_folder):
    # === Bước 1: Duyệt toàn bộ thư mục group ===
    group_classes = defaultdict(lambda: defaultdict(int))
    all_classes = set()

    for group_name in sorted(os.listdir(base_folder)):
        group_path = os.path.join(base_folder, group_name)
        if not os.path.isdir(group_path):
            continue

        for fname in os.listdir(group_path):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            class_name = fname.split('_')[0]
            group_classes[group_name][class_name] += 1
            all_classes.add(class_name)

    # === Bước 2: Tìm group có số ảnh lớn nhất cho mỗi class ===
        class_group_candidates = {}
        for cls in all_classes:
            max_count = 0
            candidates = []

            for group_name, cls_count_dict in group_classes.items():
                count = cls_count_dict.get(cls, 0)
                if count == 0:
                    continue

                # ✅ Kiểm tra xem class này có phải class có nhiều ảnh nhất trong group không
                group_max_count = max(cls_count_dict.values())
                if count < group_max_count:
                    continue  # class này không thống trị group, bỏ qua group này

                # ✅ Nếu là class mạnh nhất group, xét bình thường
                if count > max_count:
                    max_count = count
                    candidates = [group_name]
                elif count == max_count:
                    candidates.append(group_name)

            class_group_candidates[cls] = sorted(candidates) if max_count > 0 else []


    # === Bước 3: Gán tạm group đầu tiên cho mỗi class ===
    group_max = {}
    group_owners = defaultdict(list)

    for cls, groups in class_group_candidates.items():
        if groups:
            group_max[cls] = groups[0]
            group_owners[groups[0]].append(cls)
        else:
            group_max[cls] = None

    # === Bước 4: Xử lý xung đột ===
    changed = True
    while changed:
        changed = False
        for group_name, cls_list in list(group_owners.items()):
            if len(cls_list) <= 1:
                continue

            cls_list.sort(key=lambda c: (-group_classes[group_name][c], c))
            winner = cls_list[0]
            losers = cls_list[1:]
            group_owners[group_name] = [winner]

            for loser in losers:
                changed = True
                old_groups = class_group_candidates[loser]
                new_group = None

                for g in old_groups:
                    if g == group_name:
                        continue
                    count_in_g = group_classes[g].get(loser, 0)
                    max_count_for_loser = max(cls_count.get(loser, 0) for cls_count in group_classes.values())
                    if count_in_g < max_count_for_loser:
                        continue
                    if g in group_owners and len(group_owners[g]) == 1:
                        current_owner = group_owners[g][0]
                        c1 = group_classes[g][loser]
                        c2 = group_classes[g][current_owner]
                        if c1 == c2 and loser > current_owner:
                            continue

                    if loser not in group_owners[g]:
                        new_group = g
                        break

                if new_group:
                    group_max[loser] = new_group
                    group_owners[new_group].append(loser)
                else:
                    group_max[loser] = None  # ❌ Không có group hợp lệ

    # === Bước 5: Tính độ chính xác ===
    total_images = 0
    correct = 0
    wrong = 0

    for group_name in sorted(os.listdir(base_folder)):
        group_path = os.path.join(base_folder, group_name)
        if not os.path.isdir(group_path):
            continue

        for fname in os.listdir(group_path):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            total_images += 1
            class_name = fname.split('_')[0]
            expected_group = group_max.get(class_name, None)

            if expected_group is None:
                wrong += 1
            elif expected_group == group_name:
                correct += 1
            else:
                wrong += 1

    accuracy = (correct / total_images) * 100 if total_images > 0 else 0

    print("=== 📊 Kết quả đánh giá ===")
    print(f"🖼️ Tổng số ảnh: {total_images}")
    print(f"✅ Ảnh đúng nhóm: {correct}")
    print(f"❌ Ảnh sai nhóm: {wrong}")
    print(f"🎯 Độ chính xác: {accuracy:.2f}%\n")


    return accuracy


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(BASE_DIR, "img10")
    FEATURE_FILE = os.path.join(BASE_DIR, "features10.npy")
    NAME_FILE = os.path.join(BASE_DIR, "filenames10.npy")

    # Kiểm tra và tạo thư mục IMG_DIR nếu chưa có
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        print(f"Đã tạo thư mục {IMG_DIR}. Vui lòng thêm ảnh vào đó và chạy lại.")
    
    # Gọi hàm chính để thực thi
    features, filenames = mean_extract_image_features_batch_1(
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
    features = np.load("features10.npy")
    filenames = np.load("filenames10.npy")

    print("Số ảnh:", len(filenames))
    
    start = time.time()
    build_cluster_faiss(features, filenames, IMG_DIR, "clusters1", threshold=0.8, K=10)  
    end = time.time()

    print(f"[FAISS] Thời gian chạy: {end - start:.2f}s\n")
    evaluate_by_image("clusters1")

    ht = HashTable.HashTable(36, features.shape[1])
    ht1 = BloomFilter.BloomFilter(108, features.shape[1],9)
    ht2 = SimHash.SimHash(73) 
    ht3 = MinHash.MinHash(64)

    start = time.time()
    hashtable = build_clusters(ht, features, filenames, IMG_DIR,5,"clusters2")
    end = time.time()
    print(f"[HashTable] Thời gian chạy: {end - start:.10f}s\n")
    evaluate_by_image("clusters2")

    start = time.time()
    bloomfilter = build_clusters(ht1, features, filenames, IMG_DIR,21,"clusters3")
    end = time.time()
    print(f"[BloomFilter] Thời gian chạy: {end - start:.10f}s\n")
    evaluate_by_image("clusters3")

    start = time.time()
    simhash = build_clusters(ht2,features, filenames, IMG_DIR,13,"clusters4")
    end = time.time()
    print(f"[SimHash] Thời gian chạy: {end - start:.2f}s\n")
    evaluate_by_image("clusters4")

    start = time.time()
    minhash = build_clusters(ht3, features, filenames, IMG_DIR,316,"clusters5")
    end = time.time()
    print(f"[MinHash] Thời gian chạy: {end - start:.2f}s\n")    
    evaluate_by_image("clusters5")


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
# def evaluate_by_image(base_dir):
#     label_to_clusters = defaultdict(lambda: defaultdict(int))
#     total_images = 0

#     for group_name in sorted(os.listdir(base_dir)):
#         group_path = os.path.join(base_dir, group_name)
#         if not os.path.isdir(group_path):
#             continue
#         for filename in os.listdir(group_path):
#             if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#                 total_images += 1
#                 label = filename.split("_")[0]
#                 label_to_clusters[label][group_name] += 1

#     label_best_cluster = {
#         label: max(cluster_counts.items(), key=lambda x: x[1])[0]
#         for label, cluster_counts in label_to_clusters.items()
#     }

#     correct = 0
#     for group_name in sorted(os.listdir(base_dir)):
#         group_path = os.path.join(base_dir, group_name)
#         if not os.path.isdir(group_path):
#             continue
#         for filename in os.listdir(group_path):
#             if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#                 label = filename.split("_")[0]
#                 correct_cluster = label_best_cluster[label]
#                 if group_name == correct_cluster:
#                     correct += 1

#     return (correct / total_images * 100) if total_images else 0


# # === Hàm chính: Tuning từng hash + FAISS trên từng dataset ===
# def run_tuning():
#     dataset_ids = [1, 2, 3, 5, 10]
#     all_results = {}
#     early_stop_count = 120  # dừng nếu acc không cải thiện 20 lần liên tiếp

#     for did in enumerate(tqdm(dataset_ids)) :
#         feature_file = f"features{did}.npy"
#         name_file = f"filenames{did}.npy"
#         IMG_DIR = f"img{did}"

#         # Kiểm tra dữ liệu
#         if not (os.path.exists(feature_file) and os.path.exists(name_file) and os.path.exists(IMG_DIR)):
#             print(f"\n⚠️ Bỏ qua dataset {did} — thiếu file hoặc thư mục ảnh.")
#             continue

#         print(f"\n📂 Dataset {did}: {feature_file}")
#         features = np.load(feature_file)
#         filenames = np.load(name_file)
#         print(f" → {len(filenames)} ảnh, vector kích thước {features.shape[1]}")

#         dataset_results = {}

#         # === 🟠 FAISS ===
#         print("\n🔶 [FAISS] Tuning tham số...")
#         faiss_best = {"acc": 0, "param": None}
#         no_improve = 0
#         for threshold in np.arange(0.7, 0.91, 0.1):
#             for K in range(10, 58, 4):
#                 cluster_dir = f"clusters_faiss"
#                 start = time.time()
#                 build_cluster_faiss(features, filenames, IMG_DIR, cluster_dir, threshold, K)
#                 end = time.time()
#                 acc = evaluate_by_image(cluster_dir)
#                 print(f" threshold={threshold:.2f}, K={K:<3d} → acc={acc:.2f}% ({end - start:.2f}s)")

#                 if acc > faiss_best["acc"]:
#                     faiss_best = {"acc": acc, "param": (threshold, K)}
#                     no_improve = 0
#                 else:
#                     no_improve += 1
#                     if no_improve >= early_stop_count:
#                         print(f" ⚠️ Early stop: acc không cải thiện {early_stop_count} lần liên tiếp.")
#                         break
#             if no_improve >= early_stop_count:
#                 break
#         dataset_results["FAISS"] = faiss_best

#         # === 🔷 HashTable ===
#         print("\n🔷 [HashTable] Tuning tham số...")
#         ht_best = {"acc": 0, "param": None}
#         no_improve = 0
#         for buckets in range(32, 2048, 4):
#             for threshold in range(1, round(buckets/2), 1):
#                 cluster_dir = f"clusters_ht"
#                 ht = HashTable.HashTable(buckets, features.shape[1])
#                 build_clusters(ht, features, filenames, IMG_DIR, threshold, cluster_dir)
#                 acc = evaluate_by_image(cluster_dir)
#                 print(f" buckets={buckets:<4d}, threshold={threshold:<4d} → acc={acc:.2f}%")

#                 if acc > ht_best["acc"]:
#                     ht_best = {"acc": acc, "param": (buckets, threshold)}
#                     no_improve = 0
#                 else:
#                     no_improve += 1
#                     if no_improve >= early_stop_count:
#                         print(f" ⚠️ Early stop: acc không cải thiện {early_stop_count} lần liên tiếp.")
#                         break
#             if no_improve >= early_stop_count:
#                 break
#         dataset_results["HashTable"] = ht_best

#         # === 🟢 BloomFilter ===
#         print("\n🟢 [BloomFilter] Tuning tham số...")
#         bf_best = {"acc": 0, "param": None}
#         no_improve = 0
#         for k_hash in [3, 5, 7, 9, 11]:
#             for bit_size in range(36, 129, 12):
#                 if bit_size % k_hash != 0:
#                     continue
#                 for threshold in range(1, round(bit_size/2), 4):
#                     cluster_dir = f"clusters_bf"
#                     bf = BloomFilter.BloomFilter(bit_size, features.shape[1], k_hash)
#                     build_clusters(bf, features, filenames, IMG_DIR, threshold, cluster_dir)
#                     acc = evaluate_by_image(cluster_dir)
#                     print(f" bits={bit_size:<3d}, k={k_hash:<2d}, threshold={threshold:<4d} → acc={acc:.2f}%")

#                     if acc > bf_best["acc"]:
#                         bf_best = {"acc": acc, "param": (bit_size, k_hash, threshold)}
#                         no_improve = 0
#                     else:
#                         no_improve += 1
#                         if no_improve >= early_stop_count:
#                             print(f" ⚠️ Early stop: acc không cải thiện {early_stop_count} lần liên tiếp.")
#                             break
#                 if no_improve >= early_stop_count:
#                     break
#             if no_improve >= early_stop_count:
#                 break
#         dataset_results["BloomFilter"] = bf_best

#         # === 🟣 SimHash ===
#         print("\n🟣 [SimHash] Tuning tham số...")
#         sim_best = {"acc": 0, "param": None}
#         no_improve = 0
#         for bits in range(64, 129, 1):
#             for threshold in range(1, 51, 4):
#                 cluster_dir = f"clusters_sim"
#                 sim = SimHash.SimHash(bits)
#                 build_clusters(sim, features, filenames, IMG_DIR, threshold, cluster_dir)
#                 acc = evaluate_by_image(cluster_dir)
#                 print(f" bits={bits:<3d}, threshold={threshold:<3d} → acc={acc:.2f}%")

#                 if acc > sim_best["acc"]:
#                     sim_best = {"acc": acc, "param": (bits, threshold)}
#                     no_improve = 0
#                 else:
#                     no_improve += 1
#                     if no_improve >= early_stop_count:
#                         print(f" ⚠️ Early stop: acc không cải thiện {early_stop_count} lần liên tiếp.")
#                         break
#             if no_improve >= early_stop_count:
#                 break
#         dataset_results["SimHash"] = sim_best

#         # === 🟡 MinHash ===
#         print("\n🟡 [MinHash] Tuning tham số...")
#         min_best = {"acc": 0, "param": None}
#         no_improve = 0
#         for sig_size in range(64, 513, 1):
#             for threshold in range(1, 1001, 4):
#                 cluster_dir = f"clusters_min"
#                 mh = MinHash.MinHash(sig_size)
#                 build_clusters(mh, features, filenames, IMG_DIR, threshold, cluster_dir)
#                 acc = evaluate_by_image(cluster_dir)
#                 print(f" sig_size={sig_size:<3d}, threshold={threshold:<3d} → acc={acc:.2f}%")

#                 if acc > min_best["acc"]:
#                     min_best = {"acc": acc, "param": (sig_size, threshold)}
#                     no_improve = 0
#                 else:
#                     no_improve += 1
#                     if no_improve >= early_stop_count:
#                         print(f" ⚠️ Early stop: acc không cải thiện {early_stop_count} lần liên tiếp.")
#                         break
#             if no_improve >= early_stop_count:
#                 break
#         dataset_results["MinHash"] = min_best

#         all_results[f"dataset_{did}"] = dataset_results

#     # === Tổng kết ===
#     print("\n===============================")
#     print("🏆 KẾT QUẢ TỐI ƯU TOÀN BỘ")
#     print("===============================")
#     for ds, results in all_results.items():
#         print(f"\n📁 {ds}:")
#         for method, res in results.items():
#             print(f" {method:<12} → best={res['param']} acc={res['acc']:.2f}%")

#     return all_results
# if __name__ == "__main__":
#     results = run_tuning()