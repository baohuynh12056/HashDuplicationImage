import os
import shutil
from collections import defaultdict, deque

import cv2
import faiss
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
CLUSTER_DIR = "clusters"


def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def merge_similar_buckets(hashtable, threshold=1):
    """
    Gom các bucket có hash_value gần nhau (Hamming distance <= threshold).
    Trả về list các nhóm (mỗi nhóm là list tên ảnh).
    """
    hash_keys = list(hashtable.keys())
    merged = set()
    groups = []

    for i, h1 in enumerate(hash_keys):
        if h1 in merged:
            continue

        # Nhóm mới bắt đầu từ bucket hiện tại
        group = list(hashtable[h1])
        merged.add(h1)

        # So sánh với các bucket khác
        for j, h2 in enumerate(hash_keys):
            if h2 in merged:
                continue
            if hamming_distance(h1, h2) <= threshold:
                group.extend(hashtable[h2])
                merged.add(h2)

        groups.append(group)

    return groups


def normalize(features):
    """Chuẩn hóa vector về [0,1]."""
    min_val = features.min(axis=0)
    max_val = features.max(axis=0)
    norm = (features - min_val) / (max_val - min_val + 1e-10)
    return norm


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


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
        # Lấy giá trị trung bình của kênh Saturation (Độ bão hòa)
        return np.mean(hsv[:, :, 1])
    except Exception:
        return 0.0


def select_best_image_in_group(
    image_path1, image_path2, sharp_thresh=100.0, color_thresh=20.0
):
    """
    So sánh 2 ảnh và trả về ảnh tốt hơn.
    - Ảnh "tốt" là ảnh có độ sắc nét cao và màu sắc tốt.
    - Nếu cả 2 đều không đạt ngưỡng, chọn ảnh sắc nét hơn.
    """
    sharp1 = evaluate_sharpness(image_path1)
    color1 = evaluate_colorfulness(image_path1)

    sharp2 = evaluate_sharpness(image_path2)
    color2 = evaluate_colorfulness(image_path2)

    good1 = sharp1 > sharp_thresh and color1 > color_thresh
    good2 = sharp2 > sharp_thresh and color2 > color_thresh

    if good1 and good2:
        return (
            (image_path1, "Giữ ảnh 1")
            if sharp1 >= sharp2
            else (image_path2, "Giữ ảnh 2")
        )
    elif good1:
        return image_path1, "Giữ ảnh 1 (ảnh 2 kém màu/sắc)"
    elif good2:
        return image_path2, "Giữ ảnh 2 (ảnh 1 kém màu/sắc)"
    else:
        # Cả 2 đều không đạt ngưỡng → chọn ảnh sắc nét hơn
        return (
            (image_path1, "Giữ ảnh 1 (fallback)")
            if sharp1 >= sharp2
            else (image_path2, "Giữ ảnh 2 (fallback)")
        )


def build_clusters(
    ht,
    features,
    filenames,
    img_folder=IMG_DIR,
    threshold=5,
    cluster_dir=CLUSTER_DIR,
):
    """
    Gom nhóm ảnh dựa trên loại hash được truyền vào (SimHash, MinHash, BloomFilter, HashTable,...)
    ht: đối tượng hash đã khởi tạo
    """
    normalized_features = l2_normalize(features)
    n_samples = len(filenames)

    # kiểm tra thuộc tính của ht
    if type(ht).__name__ == "SimHash":
        hashtable = defaultdict(list)
        print(f"[SimHash] Bắt đầu hashing cho {n_samples} ảnh...")
        ht.IDF(normalized_features.tolist())
        for i, vec in enumerate(normalized_features):
            h = ht.hashFunction(vec.tolist())
            hashtable[h].append(filenames[i])

    elif type(ht).__name__ == "MinHash":
        hashtable = defaultdict(list)
        print(f"[MinHash] Bắt đầu hashing cho {n_samples} ảnh...")
        signatures = ht.computeSignatures(
            normalized_features.tolist(), useMedianThreshold=False
        )
        for i, sig in enumerate(signatures):
            key = int("".join(map(str, sig)), 2)
            hashtable[key].append(filenames[i])

    elif type(ht).__name__ == "BloomFilter":
        hashtable = defaultdict(set)
        print(f"[BloomFilter] Bắt đầu hashing cho {n_samples} ảnh...")
        pairs = defaultdict(list)
        for i, vec in enumerate(normalized_features):
            hash_values = ht.hashFunction(vec.tolist())
            pairs[filenames[i]] = hash_values
            for hv in hash_values:
                hashtable[hv].add(filenames[i])

        grouped = set()
        groups = []
        for fname_i in filenames:
            if fname_i in grouped:
                continue
            hash_i = pairs[fname_i]
            group = [fname_i]
            grouped.add(fname_i)
            for hv in hash_i:
                for f in hashtable[hv]:
                    if f not in grouped and f != fname_i:
                        hash_j = pairs[f]
                        total_dist = sum(
                            hamming_distance(hash_i[t], hash_j[t])
                            for t in range(len(hash_i))
                        )
                        if total_dist <= threshold:
                            group.append(f)
                            grouped.add(f)
            groups.append(group)
        final_groups = groups

    elif type(ht).__name__ == "HashTable":
        hashtable = defaultdict(list)
        print(f"[HashTable] Bắt đầu hashing cho {n_samples} ảnh...")
        for i, vec in enumerate(normalized_features):
            h = ht.hashFunction(vec.tolist())
            hashtable[h].append(filenames[i])
        final_groups = merge_similar_buckets(hashtable, threshold)

    else:
        raise ValueError("Không nhận diện được loại hash được truyền vào!")

    if not "final_groups" in locals():  # trường hợp SimHash, MinHash
        final_groups = merge_similar_buckets(hashtable, threshold)

    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir, exist_ok=True)

    print(f"Lưu {len(final_groups)} cụm ảnh vào thư mục '{cluster_dir}'...")
    for gid, group in enumerate(final_groups):
        cluster_path = os.path.join(cluster_dir, f"group_{gid:03d}")
        os.makedirs(cluster_path, exist_ok=True)
        for fname in group:
            src = os.path.join(img_folder, fname)
            if os.path.exists(src):
                shutil.copy(
                    src, os.path.join(cluster_path, os.path.basename(fname))
                )

    print(f"Hoàn tất! {len(hashtable)} bucket , {len(final_groups)} cụm.")
    return hashtable, final_groups


def build_cluster_faiss(
    features,
    filenames,
    img_folder=IMG_DIR,
    cluster_dir=CLUSTER_DIR,
    threshold=0.9,
    K=10,
):
    """
    Gom nhóm ảnh trùng hoặc gần trùng bằng FAISS và lưu ra thư mục.
    Mỗi ảnh chỉ thuộc 1 nhóm duy nhất (sử dụng lan truyền theo đồ thị).
    """

    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features = l2_normalize(features)
    d = features.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(features)
    similarities, neighbors = index.search(features, K)

    adj = defaultdict(set)
    N = len(features)
    for i in range(N):
        for j in range(1, K):
            nb = neighbors[i][j]
            sim = similarities[i][j]
            if sim >= threshold:
                adj[i].add(nb)
                adj[nb].add(i)  # đối xứng (undirected)

    visited = set()
    final_groups = []
    hashtable = {}

    for i in range(N):
        if i in visited:
            continue
        # BFS để gom nhóm
        queue = deque([i])
        component = set([i])
        visited.add(i)
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    component.add(nb)
                    queue.append(nb)
        # Lưu nhóm
        group_filenames = [filenames[idx] for idx in component]
        final_groups.append(group_filenames)
        hashtable[tuple(group_filenames)] = group_filenames

    final_groups.sort(key=len, reverse=True)

    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir, exist_ok=True)
    print(f"Lưu {len(final_groups)} cụm ảnh vào thư mục '{cluster_dir}'...")

    for gid, group in enumerate(final_groups):
        cluster_path = os.path.join(cluster_dir, f"group_{gid:03d}")
        os.makedirs(cluster_path, exist_ok=True)
        for fname in group:
            src = os.path.join(img_folder, fname)
            if os.path.exists(src):
                shutil.copy(
                    src, os.path.join(cluster_path, os.path.basename(fname))
                )

    print(f"Có {len(hashtable)} bucket, {len(final_groups)} cụm.")
    return hashtable, final_groups
