import os
import numpy as np
from Application.resnet import mean_extract_image_features_batch_1 # <-- import trực tiếp từ file bạn có
from Application.cluster import build_clusters,build_cluster_faiss,analyze_and_plot_distances # <-- import trực tiếp từ file bạn có
# =============== Cấu hình cơ bản ===============
import hash_table_py as HashTable
import simhash_py as SimHash
import minhash_py as MinHash
import bloom_filter_py as BloomFilter
import os
from collections import defaultdict
import time



def evaluate_precision_recall(base_folder):
    """
    Tính Precision, Recall, F1-score từng class và macro-average.
    """
    group_classes = defaultdict(lambda: defaultdict(int))
    all_classes = set()

    # Bước 1: Duyệt thư mục nhóm
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

    # Bước 2: Xác định “chủ sở hữu” từng class
    class_group_candidates = {}
    for cls in all_classes:
        max_count = 0
        candidates = []
        for group_name, cls_count_dict in group_classes.items():
            count = cls_count_dict.get(cls, 0)
            if count == 0:
                continue
            group_max_count = max(cls_count_dict.values())
            if count < group_max_count:
                continue
            if count > max_count:
                max_count = count
                candidates = [group_name]
            elif count == max_count:
                candidates.append(group_name)
        class_group_candidates[cls] = sorted(candidates) if max_count > 0 else []

    group_max = {}
    group_owners = defaultdict(list)
    for cls, groups in class_group_candidates.items():
        if groups:
            group_max[cls] = groups[0]
            group_owners[groups[0]].append(cls)
        else:
            group_max[cls] = None

    # Xử lý xung đột
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
                    new_group = g
                    break
                group_max[loser] = new_group

    # Tính Precision, Recall, F1-score
    class_counts_true = defaultdict(int)     # số ảnh thực sự của class
    class_counts_pred = defaultdict(int)     # số ảnh dự đoán vào group
    class_correct = defaultdict(int)         # số ảnh đúng

    for group_name in sorted(os.listdir(base_folder)):
        group_path = os.path.join(base_folder, group_name)
        if not os.path.isdir(group_path):
            continue
        for fname in os.listdir(group_path):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            class_name = fname.split('_')[0]
            class_counts_true[class_name] += 1
            predicted_group = group_name
            expected_group = group_max.get(class_name, None)
            if expected_group == predicted_group:
                class_correct[class_name] += 1
            class_counts_pred[predicted_group] += 1

    precisions = {}
    recalls = {}
    f1s = {}

    for cls in all_classes:
        tp = class_correct[cls]
        fp = class_counts_pred.get(group_max.get(cls,""),0) - tp
        fn = class_counts_true[cls] - tp
        precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precisions[cls] + recalls[cls] > 0:
            f1s[cls] = 2 * precisions[cls] * recalls[cls] / (precisions[cls] + recalls[cls])
        else:
            f1s[cls] = 0.0

    # Macro-average
    macro_precision = sum(precisions.values()) / len(all_classes) if all_classes else 0
    macro_recall = sum(recalls.values()) / len(all_classes) if all_classes else 0
    macro_f1 = sum(f1s.values()) / len(all_classes) if all_classes else 0

    # print("=== 📊 Metrics từng class ===")
    # for cls in all_classes:
    #     print(f"{cls}: Precision={precisions[cls]:.2f}, Recall={recalls[cls]:.2f}, F1={f1s[cls]:.2f}")
    print(f"\n🎯 Macro Precision={macro_precision:.2f}, Macro Recall={macro_recall:.2f}, Macro F1={macro_f1:.2f}\n")

    return precisions, recalls, f1s, macro_precision, macro_recall, macro_f1
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
    IMG_DIR = os.path.join(BASE_DIR, "img")
    FEATURE_FILE = os.path.join(BASE_DIR, "features.npy")
    NAME_FILE = os.path.join(BASE_DIR, "filenames.npy")

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
    features = np.load("features_test.npy")
    filenames = np.load("filenames_test.npy")

    print("Số ảnh:", len(filenames))
    
    start = time.time()
    build_cluster_faiss(features, filenames, IMG_DIR, "clusters1", threshold=0.8, K=10)  
    end = time.time()
    ht = SimHash.SimHash(73)
    # ht2 = HashTable.HashTable(36, features.shape[1])
    # ht3 = BloomFilter.BloomFilter(108, features.shape[1],9)
    # ht4 = MinHash.MinHash(64)
    start = time.time()
    best_threshold = analyze_and_plot_distances(ht, features)
    build_clusters(ht, features, filenames, IMG_DIR, best_threshold, "clusters_simhash")
    end = time.time()    
    evaluate_precision_recall("clusters_simhash")
    evaluate_by_image("clusters_simhash")
    print(f"Thời gian chạy: {end - start:.2f}s\n")
