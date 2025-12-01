import os
import shutil
import random
import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Any, List
import scipy.ndimage as filters
from scipy.signal import argrelextrema
from tqdm import tqdm  # Cần cài đặt: pip install tqdm

# --- CÁC HÀM TIỆN ÍCH CƠ BẢN ---

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector L2."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-7
    return vectors / norm

def hamming_distance(h1, h2) -> int:
    """
    Tính khoảng cách Hamming.
    Hỗ trợ cả số nguyên (SimHash/HashTable), chuỗi bit và List/Array.
    """
    # Trường hợp số nguyên
    if isinstance(h1, (int, np.integer)) and isinstance(h2, (int, np.integer)):
        return bin(int(h1) ^ int(h2)).count("1")

    # Trường hợp mảng/list/tuple
    if hasattr(h1, "__len__") and hasattr(h2, "__len__"):
        if len(h1) != len(h2):
            return 9999
        arr1 = np.array(h1)
        arr2 = np.array(h2)
        return np.sum(arr1 != arr2)

    return 9999

# --- PHẦN ANALYZE & THRESHOLD (BẠN YÊU CẦU SỬA) ---

def find_valley_threshold(distances: List[int], smooth_sigma=1.0):
    """
    Tìm threshold dựa trên vị trí thung lũng (valley) giữa 2 đỉnh của biểu đồ.
    Sử dụng làm mượt Gaussian để tránh nhiễu răng cưa.
    """
    if not distances:
        return 0, []

    dist_arr = np.array(distances, dtype=int)
    min_val, max_val = int(np.min(dist_arr)), int(np.max(dist_arr))
    
    # Nếu tất cả khoảng cách giống nhau hoặc quá ít biến thiên
    if max_val - min_val < 2:
        return min_val, []

    # 1. Tính Histogram
    bins = range(min_val, max_val + 2)
    hist, bin_edges = np.histogram(dist_arr, bins=bins)
    
    # 2. Làm mượt biểu đồ (Gaussian Smoothing)
    # Bước này cực quan trọng để loại bỏ các trồi sụt nhỏ (răng cưa)
    smooth_hist = filters.gaussian_filter1d(hist, sigma=smooth_sigma)
    
    # 3. Tìm các điểm cực tiểu cục bộ (Local Minima)
    # order=2 so sánh với 2 điểm bên trái và 2 điểm bên phải
    local_min_indices = argrelextrema(smooth_hist, np.less, order=2)[0]
    
    # 4. Logic chọn thung lũng:
    valid_threshold = None
    
    if len(local_min_indices) > 0:
        # Lấy cực tiểu đầu tiên tìm thấy
        valid_threshold = local_min_indices[0] + min_val
        
        # Kiểm tra an toàn: Threshold không nên quá nhỏ (sát 0)
        # Nếu threshold tìm được < min + 2, có thể do nhiễu ở đầu, ta tìm cái tiếp theo
        if valid_threshold < min_val + 2 and len(local_min_indices) > 1:
             valid_threshold = local_min_indices[1] + min_val
    else:
        # Fallback: Nếu không tìm thấy thung lũng (biểu đồ chỉ có 1 dốc)
        print("⚠️ Không tìm thấy thung lũng rõ ràng. Dùng Percentile 10% an toàn.")
        valid_threshold = int(np.percentile(dist_arr, 10))

    return int(valid_threshold), smooth_hist

def analyze_and_plot_distances(ht, features, sample_size=None):
    """
    Hàm độc lập để phân tích và tìm threshold (dùng cho EDA hoặc test).
    """
    normalized_features = l2_normalize(features)
    n_samples = len(normalized_features)
    
    # --- BƯỚC 1: Sinh Hash ---
    hashes = []
    hash_type = type(ht).__name__
    print(f"[{hash_type}] Đang sinh mã hash ({n_samples} mẫu)...")
    
    # (Phần sinh hash)
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
        # Fallback cho các loại hash tự chế khác
        if hasattr(ht, "hashFunction"):
             for vec in normalized_features:
                h = ht.hashFunction(vec.tolist())
                hashes.append(h)
        else:
             raise ValueError(f"Loại hash '{hash_type}' chưa được hỗ trợ!")

    # --- BƯỚC 2: Tính khoảng cách ---
    if sample_size and n_samples > sample_size:
        print(f"Lấy mẫu ngẫu nhiên {sample_size} ảnh...")
        hashes = random.sample(hashes, sample_size)
    
    distances = []
    # Sử dụng combinations để tính cặp
    pairs = list(combinations(hashes, 2))
    print(f"Đang tính toán khoảng cách cho {len(pairs)} cặp...")
    
    # Dùng tqdm để hiện thanh tiến trình
    for h1, h2 in tqdm(pairs):
        try:
            distances.append(hamming_distance(h1, h2))
        except ValueError: continue

    if not distances: return 0

    # --- BƯỚC 3: TÍNH THRESHOLD THEO CÁCH MỚI (VALLEY) ---
    best_threshold, smooth_hist = find_valley_threshold(distances, smooth_sigma=1.5)
    
    print(f"-> Thuật toán tìm thung lũng đề xuất: {best_threshold}")
    
    return best_threshold

def merge_similar_buckets(hashtable, threshold=1):
    """
    Gom các bucket có hash_value gần nhau (Hamming distance <= threshold).
    Trả về list các nhóm (mỗi nhóm là list tên ảnh).
    """
    hash_keys = list(hashtable.keys())
    merged = set()
    groups = []
    
    print(f"   -> Đang merge {len(hash_keys)} buckets với threshold={threshold}...")

    # Duyệt O(N^2) trên số lượng bucket (thường nhỏ hơn số ảnh nhiều)
    for i, h1 in enumerate(hash_keys):
        if h1 in merged:
            continue

        # Nhóm mới bắt đầu từ bucket hiện tại
        group = list(hashtable[h1])
        merged.add(h1)

        for j, h2 in enumerate(hash_keys):
            if h2 in merged:
                continue
            if hamming_distance(h1, h2) <= threshold:
                group.extend(hashtable[h2])
                merged.add(h2)

        if len(group) > 1:
            groups.append(group)

    return groups

# --- CLASS CLUSTERING ĐÃ CHỈNH SỬA ---

class HierarchicalClustering:
    def __init__(self, hash_type: str = "SimHash"):
        self.hash_type = hash_type

    def cluster(
        self, 
        hash_obj: Any, 
        features: np.ndarray, 
        filenames: List[str], 
    ) -> List[List[str]]:
        """
        Quy trình chính: Hash -> Tính Distance -> Tìm Threshold (Valley) -> Merge Bucket -> Lưu file.
        """
        n_samples = len(filenames)
        if n_samples == 0:
            return []

        # 1. Chuẩn hóa
        normalized_features = l2_normalize(features)
        
        # Biến lưu trữ hash để tính threshold và hashtable để merge
        # `all_hashes_for_threshold`: list các hash (để tính distance cặp)
        all_hashes_for_threshold = [] 
        
        # `hashtable`: dict map hash -> list filenames (để merge bucket)
        hashtable = defaultdict(list)
        
        # BloomFilter cần cấu trúc đặc biệt hơn một chút
        bloom_pairs = defaultdict(list) 

        print(f"[{self.hash_type}] Bắt đầu hashing cho {n_samples} ảnh...")

        # --- GIAI ĐOẠN 1: HASHING ---
        try:
            if self.hash_type == "SimHash":
                if hasattr(hash_obj, "IDF"):
                    hash_obj.IDF(normalized_features.tolist())
                    for i, vec in enumerate(normalized_features):
                        h = hash_obj.hashFunction(vec.tolist())
                        hashtable[h].append(filenames[i])
                        all_hashes_for_threshold.append(h)
                else:
                    return []

            elif self.hash_type == "MinHash":
                signatures = hash_obj.computeSignatures(normalized_features.tolist(), useMedianThreshold=False)
                for i, sig in enumerate(signatures):
                    try:
                        key = tuple(sig) # Tuple hóa để làm key dict
                    except:
                        key = i
                    hashtable[key].append(filenames[i])
                    all_hashes_for_threshold.append(sig)

            elif self.hash_type == "HashTable" or hasattr(hash_obj, "hashFunction"):
                # Dùng cho HashTable hoặc các class custom có hashFunction
                for i, vec in enumerate(normalized_features):
                    h = hash_obj.hashFunction(vec.tolist())
                    # Nếu h là array, chuyển thành tuple hoặc giữ nguyên tùy loại
                    h_key = h
                    if isinstance(h, np.ndarray):
                        h_key = tuple(h.tolist()) # Tuple để làm key dict
                        
                    if self.hash_type == "BloomFilter":
                        # BloomFilter trả về list các hash, xử lý riêng bên dưới
                        bloom_pairs[filenames[i]] = h
                        all_hashes_for_threshold.append(h) # Lưu list hash để tính distance sau
                        for sub_h in h:
                            hashtable[sub_h].append(filenames[i]) # Bloom dùng list thay vì set ở đây cho đồng nhất logic
                    else:
                        hashtable[h_key].append(filenames[i])
                        all_hashes_for_threshold.append(h)
            
            else:
                print(f"❌ Unknown hash type: {self.hash_type}")
                return []

        except Exception as e:
            print(f"❌ Error during hashing: {e}")
            import traceback
            traceback.print_exc()
            return []

        # --- GIAI ĐOẠN 2: TÍNH THRESHOLD TỰ ĐỘNG (VALLEY) ---
        print(f"[{self.hash_type}] Đang tính toán khoảng cách để tìm Threshold...")
        distances = []
        
        # Lấy tất cả các cặp (Full Scan) hoặc lấy mẫu nếu quá lớn (để tránh treo máy với N>5000)
        # Nếu muốn Full Scan tuyệt đối như bạn yêu cầu trước đó:
        pairs = list(combinations(all_hashes_for_threshold, 2))
        
        # Giới hạn hiển thị tqdm nếu quá nhiều
        limit_desc = "Full Distance Scan"
        if len(pairs) > 1000000:
             print("⚠️ Số lượng cặp quá lớn (>1M), việc tính toán có thể mất thời gian.")
        
        for h1, h2 in tqdm(pairs, desc=limit_desc):
            distances.append(hamming_distance(h1, h2))
            
        # Gọi hàm find_valley_threshold đã sửa đổi
        threshold, _ = find_valley_threshold(distances, smooth_sigma=1.5)
        print(f"✅ [Auto-Threshold] Ngưỡng cắt đề xuất: {threshold}")

        # --- GIAI ĐOẠN 3: GOM NHÓM (MERGE BUCKETS) ---
        print(f"[{self.hash_type}] Đang gom nhóm (Merging)...")

        final_groups = []

        if self.hash_type == "BloomFilter":
            # Logic riêng cho BloomFilter
            grouped_files = set()
            groups = []
            
            # Chuyển hashtable values thành set để lookup nhanh hơn
            hashtable_set = {k: set(v) for k, v in hashtable.items()}

            for fname_i in filenames:
                if fname_i in grouped_files:
                    continue
                
                hash_i = bloom_pairs[fname_i]
                group = [fname_i]
                grouped_files.add(fname_i)
                
                candidates = set()
                for hv in hash_i:
                    if hv in hashtable_set:
                        candidates.update(hashtable_set[hv])
                
                for f in candidates:
                    if f not in grouped_files and f != fname_i:
                        hash_j = bloom_pairs[f]
                        # BloomFilter: distance = tổng hamming các thành phần
                        total_dist = sum(hamming_distance(hash_i[t], hash_j[t]) for t in range(len(hash_i)))
                        
                        if total_dist <= threshold:
                            group.append(f)
                            grouped_files.add(f)
                
                if len(group) > 1:
                    groups.append(group)
            final_groups = groups

        else:
            # Logic chung cho SimHash, MinHash, HashTable
            final_groups = merge_similar_buckets(hashtable, threshold)

        return final_groups