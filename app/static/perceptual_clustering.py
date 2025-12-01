import random
from collections import defaultdict
from itertools import combinations
from typing import Any, List

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector L2."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-7
    return vectors / norm


def hamming_distance(h1, h2) -> int:
    """
    Tính khoảng cách Hamming.
    Hỗ trợ cả số nguyên (SimHash/HashTable) và List/Array (MinHash signatures).
    """
    # Trường hợp số nguyên (SimHash, HashTable)
    if isinstance(h1, (int, np.integer)) and isinstance(h2, (int, np.integer)):
        return bin(int(h1) ^ int(h2)).count("1")

    # Trường hợp mảng/list (MinHash signatures)
    if hasattr(h1, "__len__") and hasattr(h2, "__len__"):
        if len(h1) != len(h2):
            return 9999
        arr1 = np.array(h1)
        arr2 = np.array(h2)
        return np.sum(arr1 != arr2)

    return 9999


def find_valley_threshold(distances: List[int], smooth_sigma=1.5) -> int:
    """
    Tìm threshold dựa trên vị trí thung lũng (valley).
    """
    if not distances:
        return 0

    dist_arr = np.array(distances, dtype=int)
    min_val, max_val = int(np.min(dist_arr)), int(np.max(dist_arr))

    # Nếu khoảng cách quá ít biến thiên
    if max_val - min_val < 2:
        return min_val

    # 1. Tính Histogram
    bins = range(min_val, max_val + 2)
    hist, _ = np.histogram(dist_arr, bins=bins)

    # 2. Làm mượt biểu đồ (Gaussian Smoothing)
    try:
        # Sử dụng hàm chuẩn từ scipy.ndimage
        smooth_hist = gaussian_filter1d(hist, sigma=smooth_sigma)
    except Exception as e:
        print(f"⚠️ Warning: Gaussian filter failed ({e}), using raw histogram.")
        smooth_hist = hist

    # 3. Tìm cực tiểu cục bộ
    local_min_indices = argrelextrema(smooth_hist, np.less, order=2)[0]

    valid_threshold = None

    if len(local_min_indices) > 0:
        valid_threshold = local_min_indices[0] + min_val
        if valid_threshold < min_val + 2 and len(local_min_indices) > 1:
            valid_threshold = local_min_indices[1] + min_val
    else:
        # Fallback
        valid_threshold = int(np.percentile(dist_arr, 15))

    return int(valid_threshold)


# --- PYTHON FALLBACK ---
class PythonHashTable:
    def __init__(self, input_dim=2048, hash_bits=64):
        np.random.seed(42)
        self.planes = np.random.randn(input_dim, hash_bits)

    def hash(self, vector: np.ndarray) -> int:
        projections = np.dot(vector, self.planes)
        bits = (projections > 0).astype(int)
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | bit
        return hash_val


# --- CLASS CLUSTERING CHÍNH ---


class HierarchicalClustering:
    def __init__(self, hash_type: str = "SimHash"):
        self.hash_type = hash_type

    def cluster(
        self, hash_obj: Any, features: np.ndarray, filenames: List[str]
    ) -> List[List[str]]:
        n_samples = len(filenames)
        if n_samples == 0:
            return []

        # 1. Chuẩn hóa
        normalized_features = l2_normalize(features)
        hashes = []

        print(f"[{self.hash_type}] Sinh mã hash cho {n_samples} ảnh...")

        try:
            # --- SINH HASH ---
            if self.hash_type == "SimHash":
                if hasattr(hash_obj, "IDF"):
                    hash_obj.IDF(normalized_features.tolist())
                    for vec in normalized_features:
                        hashes.append(hash_obj.hashFunction(vec.tolist()))
                else:
                    return []

            elif self.hash_type == "MinHash":
                hashes = hash_obj.computeSignatures(
                    normalized_features.tolist(), useMedianThreshold=False
                )

            elif self.hash_type == "HashTable":
                if hash_obj and hasattr(hash_obj, "hashFunction"):
                    for vec in normalized_features:
                        hashes.append(hash_obj.hashFunction(vec.tolist()))
                else:
                    print("⚠️ Using Python HashTable fallback")
                    py_hasher = PythonHashTable(
                        input_dim=normalized_features.shape[1], hash_bits=64
                    )
                    for vec in normalized_features:
                        hashes.append(py_hasher.hash(vec))

            elif self.hash_type == "BloomFilter":
                if hash_obj:
                    for vec in normalized_features:
                        h = hash_obj.hashFunction(vec.tolist())
                        hashes.append(tuple(h) if isinstance(h, list) else h)
        except Exception as e:
            print(f"❌ Error during hashing: {e}")
            return []

        # 2. Tính toán khoảng cách mẫu
        sample_size = min(2000, len(hashes))
        sample_hashes = (
            random.sample(hashes, sample_size)
            if len(hashes) > sample_size
            else hashes
        )

        distances = []
        sample_indices = list(combinations(range(len(sample_hashes)), 2))
        random.shuffle(sample_indices)

        # Lấy mẫu tối đa 30k cặp để tính nhanh
        for i, j in sample_indices[:30000]:
            distances.append(
                hamming_distance(sample_hashes[i], sample_hashes[j])
            )

        # 3. Tự động tìm Threshold
        threshold = find_valley_threshold(distances, smooth_sigma=1.5)
        print(f"✅ [Auto-Threshold] Ngưỡng cắt: {threshold}")

        # 4. Gom nhóm (Union-Find)
        parent = list(range(n_samples))

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        print(f"[{self.hash_type}] Đang gom nhóm...")

        # So khớp
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if hamming_distance(hashes[i], hashes[j]) <= threshold:
                    union(i, j)

        groups_map = defaultdict(list)
        for i in range(n_samples):
            root = find(i)
            groups_map[root].append(filenames[i])

        result_groups = [g for g in groups_map.values() if len(g) > 1]

        print(f"✓ Hoàn tất: {len(result_groups)} nhóm trùng.")
        return result_groups
