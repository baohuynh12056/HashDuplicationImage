import os
import shutil
from collections import defaultdict
import numpy as np
from simhash_py import SimHash  # hoặc: from .SimHash import SimHash
def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# =======================================================
# 🧩 Các hàm hỗ trợ (có thể để trong cùng file hoặc import)
# =======================================================
def hamming_distance(a, b):
    return bin(a ^ b).count("1")

def split_hash(hash_val, bands=4, bits=16):
    """Chia hash thành nhiều band (ví dụ: 16 bits → 4 band, mỗi band 4 bits)"""
    band_size = bits // bands
    masks = [(1 << band_size) - 1 for _ in range(bands)]
    bands_list = []
    for i in range(bands):
        shift = (bands - i - 1) * band_size
        band_val = (hash_val >> shift) & masks[i]
        bands_list.append(band_val)
    return bands_list

def build_lsh_table(hash_list, bands=4, bits=16):
    """Tạo nhiều bảng hash (LSH)"""
    tables = [defaultdict(list) for _ in range(bands)]
    for idx, hv in enumerate(hash_list):
        band_vals = split_hash(hv, bands=bands, bits=bits)
        for b, band_val in enumerate(band_vals):
            tables[b][band_val].append(idx)
    return tables

def find_similar_groups(hash_list, tables, threshold=3, bands=4, bits=16):
    """Tìm nhóm ảnh trùng nhau (theo Hamming distance)"""
    visited = set()
    groups = []

    for i in range(len(hash_list)):
        if i in visited:
            continue
        group = {i}
        candidates = set()

        # tìm ảnh cùng bucket ở ít nhất 1 band
        band_vals = split_hash(hash_list[i], bands=bands, bits=bits)
        for b, band_val in enumerate(band_vals):
            candidates.update(tables[b][band_val])

        for j in candidates:
            if j != i and hamming_distance(hash_list[i], hash_list[j]) <= threshold:
                group.add(j)
                visited.add(j)

        visited.add(i)
        if len(group) > 1:
            groups.append(group)
    return groups

# =======================================================
# 🚀 Hàm chính: Build cluster dùng SimHash + LSH
# =======================================================
def build_clusters_simhash(features, filenames, img_folder, cluster_dir="clusters_lsh",
                           bits=32, bands=4, threshold=10):
    ht = SimHash(bits)
    normalized_features = l2_normalize(features)

    # 1️⃣ Tính SimHash cho từng ảnh
    print("[INFO] Bắt đầu tính SimHash...")
    img_hashes = [ht.hashFunction(vec.tolist()) for vec in normalized_features]

    # 2️⃣ Tạo bảng LSH (chia nhóm band)
    print("[INFO] Xây bảng LSH...")
    tables = build_lsh_table(img_hashes, bands=bands, bits=bits)

    # 3️⃣ Tìm nhóm ảnh trùng nhau
    print("[INFO] Tìm nhóm ảnh tương tự...")
    groups = find_similar_groups(img_hashes, tables, threshold=threshold, bands=bands, bits=bits)

    # 4️⃣ Tạo thư mục kết quả
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)

    # 5️⃣ Lưu ảnh theo nhóm
    print(f"[INFO] Bắt đầu lưu {len(groups)} nhóm ảnh...")
    for idx, group in enumerate(groups, 1):
        folder = os.path.join(cluster_dir, f"group_{idx}")
        os.makedirs(folder, exist_ok=True)
        for i in group:
            src = os.path.join(img_folder, filenames[i])
            if os.path.exists(src):
                shutil.copy(src, os.path.join(folder, filenames[i]))

    print(f"[INFO] ✅ Hoàn tất: {len(groups)} nhóm ảnh đã được lưu vào '{cluster_dir}/'")
    return groups
