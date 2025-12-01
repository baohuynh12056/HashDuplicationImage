import os
import shutil
import numpy as np
import cv2
from collections import defaultdict,deque
import faiss
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # <--- Thêm thư viện này để xử lý trục số nguyên
import random
from scipy.signal import argrelextrema
import scipy.ndimage.filters as filters
from itertools import combinations
from tqdm import tqdm
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
    return bin(a ^ b).count('1')
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

def build_clusters(ht, features, filenames, img_folder = IMG_DIR, threshold=5, cluster_dir = CLUSTER_DIR):
    """
    Gom nhóm ảnh dựa trên loại hash được truyền vào (SimHash, MinHash, BloomFilter, HashTable,...)
    ht: đối tượng hash đã khởi tạo 
    """
    normalized_features = l2_normalize(features)
    n_samples = len(filenames)

    #kiểm tra thuộc tính của ht
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
        signatures = ht.computeSignatures(normalized_features.tolist(), useMedianThreshold=False)
        for i, sig in enumerate(signatures):
            key = int(''.join(map(str, sig)), 2)
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
                        total_dist = sum(hamming_distance(hash_i[t], hash_j[t]) for t in range(len(hash_i)))
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

    if not 'final_groups' in locals():  # trường hợp SimHash, MinHash
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
                shutil.copy(src, os.path.join(cluster_path, os.path.basename(fname)))

    print(f"Hoàn tất! {len(hashtable)} bucket , {len(final_groups)} cụm.")
    return hashtable, final_groups

def build_cluster_faiss(features, filenames, img_folder = IMG_DIR, cluster_dir = CLUSTER_DIR, threshold=0.9, K=10):
    """
    Gom nhóm ảnh trùng hoặc gần trùng bằng FAISS và lưu ra thư mục.
    Mỗi ảnh chỉ thuộc 1 nhóm duy nhất (sử dụng lan truyền theo đồ thị).
    """

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
                shutil.copy(src, os.path.join(cluster_path, os.path.basename(fname)))

    print(f"Có {len(hashtable)} bucket, {len(final_groups)} cụm.")
    return hashtable, final_groups

def find_valley_threshold(distances, smooth_sigma=1.0):
    """
    Tìm threshold dựa trên vị trí thung lũng (valley) giữa 2 đỉnh của biểu đồ.
    Sử dụng làm mượt Gaussian để tránh nhiễu răng cưa.
    """
    dist_arr = np.array(distances, dtype=int)
    min_val, max_val = int(np.min(dist_arr)), int(np.max(dist_arr))
    
    # 1. Tính Histogram
    bins = range(min_val, max_val + 2)
    hist, bin_edges = np.histogram(dist_arr, bins=bins)
    
    # 2. Làm mượt biểu đồ (Gaussian Smoothing)
    # Bước này cực quan trọng để loại bỏ các trồi sụt nhỏ (răng cưa)
    # sigma càng lớn thì càng mượt
    smooth_hist = filters.gaussian_filter1d(hist, sigma=smooth_sigma)
    
    local_min_indices = argrelextrema(smooth_hist, np.less, order=2)[0]
    
    
    valid_threshold = None
    
    if len(local_min_indices) > 0:
        # Lấy cực tiểu đầu tiên tìm thấy
        # + min_val để bù lại offset nếu bin không bắt đầu từ 0
        valid_threshold = local_min_indices[0] + min_val
        
        # Kiểm tra an toàn: Threshold không nên quá nhỏ (sát 0) hoặc quá lớn
        # Nếu threshold tìm được < min + 2, có thể do nhiễu ở đầu, ta tìm cái tiếp theo
        if valid_threshold < min_val + 2 and len(local_min_indices) > 1:
             valid_threshold = local_min_indices[1] + min_val
    else:
        # Fallback: Nếu không tìm thấy thung lũng (biểu đồ chỉ có 1 dốc)
        # Ta dùng percentile 5% hoặc 10% làm ngưỡng an toàn
        print("Không tìm thấy thung lũng rõ ràng. Dùng Percentile an toàn.")
        valid_threshold = int(np.percentile(dist_arr, 10))

    return valid_threshold, smooth_hist

def analyze_and_plot_distances(ht, features, sample_size=None):
    """
    Phân tích, vẽ histogram và TỰ ĐỘNG TÌM THRESHOLD TỐT NHẤT.
    """
    normalized_features = l2_normalize(features)
    n_samples = len(normalized_features)
    
    # --- BƯỚC 1: Sinh Hash ---
    hashes = []
    hash_type = type(ht).__name__
    print(f"[{hash_type}] Đang sinh mã hash ({n_samples} mẫu)...")
    
    # (Phần sinh hash giữ nguyên như code cũ...)
    if hash_type == "SimHash":
        ht.IDF(normalized_features.tolist())
        for vec in normalized_features:
            hashes.append(ht.hashFunction(vec.tolist()))
    elif hash_type == "MinHash":
        hashes = ht.computeSignatures(normalized_features.tolist(), useMedianThreshold=False)
    elif hash_type == "BloomFilter" or hash_type == "HashTable":
        for vec in normalized_features:
            h = ht.hashFunction(vec.tolist())
            hashes.append(h.tolist() if isinstance(h, np.ndarray) else h)
    else:
         raise ValueError(f"Loại hash '{hash_type}' chưa được hỗ trợ!")

    # --- BƯỚC 2: Tính khoảng cách ---
    if sample_size and n_samples > sample_size:
        print(f"Lấy mẫu ngẫu nhiên {sample_size} ảnh...")
        hashes = random.sample(hashes, sample_size)
    
    distances = []
    pairs = list(combinations(hashes, 2))
    print("Đang tính toán khoảng cách...")
    
    for h1, h2 in tqdm(pairs):
        try:
            distances.append(hamming_distance(h1, h2))
        except ValueError: continue

    if not distances: return []
    # --- CHỈ SỬA PHẦN TÍNH THRESHOLD VÀ VẼ ---
    dist_arr = np.array(distances, dtype=int)
    min_val, max_val = int(np.min(dist_arr)), int(np.max(dist_arr))
    
    # TÍNH THRESHOLD THEO CÁCH MỚI (VALLEY)
    best_threshold, smooth_hist = find_valley_threshold(dist_arr, smooth_sigma=1.5)
    
    print(f"-> Thuật toán tìm thung lũng đề xuất: {best_threshold}")

    # # Vẽ biểu đồ
    # plt.figure(figsize=(12, 6))
    # bins = range(min_val, max_val + 2)
    
    # # Vẽ Histogram gốc (cột xanh)
    # plt.hist(dist_arr, bins=bins, color='#3498db', alpha=0.5, edgecolor='black', align='left', label='Dữ liệu thực')
    
    # # Vẽ đường cong đã làm mượt (để minh họa cách máy tính nhìn thấy thung lũng)
    # # Lưu ý: smooth_hist có len = len(bins)-1, cần vẽ khớp trục x
    # x_smooth = np.arange(min_val, max_val + 1)
    # plt.plot(x_smooth, smooth_hist, color='orange', linewidth=2, linestyle='-', label='Đường xu hướng (Smoothed)')
    
    # # Vẽ Threshold
    # plt.axvline(best_threshold, color='red', linestyle='-', linewidth=2, label=f'Best Threshold (Valley): {best_threshold}')
    
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # if (max_val - min_val) < 40:
    #     plt.xticks(range(min_val, max_val + 1))
        
    # plt.title(f'Phân tích điểm cắt thung lũng (Valley Detection)', fontsize=14)
    # plt.legend()
    # plt.show()
    
    return best_threshold