import os
import shutil
import numpy as np
import cv2
from collections import defaultdict
import hash_table_py as HashTable
import simhash_py as SimHash
import minhash_py as MinHash
import bloom_filter_py as BloomFilter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
NUM_BUCKETS = 10  # số bucket trong hash table
CLUSTER_DIR = "clusters"
NUM_PLANES = 128

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
def build_clusters_hash_table(features, filenames, img_folder):
    """Xây dựng cluster từ feature vectors và lưu ảnh theo bucket."""
    hashtable = defaultdict(list)
    ht = HashTable.HashTable(32, features.shape[1])
    normalized_features = l2_normalize(features)
    print("[INFO] Bắt đầu thêm vector vào HashTable...")
    for i, vec in enumerate(normalized_features):
         # ép sang list nếu C++ binding yêu cầu
        hash_key = ht.hashFunction(vec.tolist())
        print(hash_key,filenames[i])
        hashtable[hash_key].append(filenames[i])

    # Xóa thư mục cũ nếu có
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR)

    # print("Bắt đầu lưu ảnh vào các cụm...")
    # for bucket_id, img_list in hashtable.items():
    #     cluster_path = os.path.join(CLUSTER_DIR, f"bucket_{bucket_id}")
    #     os.makedirs(cluster_path, exist_ok=True)
    #     if not img_list:
    #         continue   
    #     for fname in img_list:
    #         src = os.path.join(img_folder, fname)
    #         if os.path.exists(src):
    #             shutil.copy(src, os.path.join(cluster_path, fname))
    print("🔍 Gom nhóm ảnh theo độ tương tự hash...")
    groups = merge_similar_buckets(hashtable, threshold=5)
    print(f"✅ Tạo {len(groups)} cụm ảnh (threshold={5}).")

    print("💾 Bắt đầu lưu ảnh vào các cụm...")
    os.makedirs(CLUSTER_DIR, exist_ok=True)

    for group_id, group in enumerate(groups):
        cluster_path = os.path.join(CLUSTER_DIR, f"group_{group_id:03d}")
        os.makedirs(cluster_path, exist_ok=True)

        for fname in group:
            src = os.path.join(img_folder, fname)
            dst = os.path.join(cluster_path, os.path.basename(fname))
            if os.path.exists(src):
                shutil.copy(src, dst)    
    print(f"Đã tạo {len(hashtable)} bucket trong thư mục '{CLUSTER_DIR}/'")
    return hashtable, groups

  
def build_clusters_best(features, filenames, img_folder):
    """Xây dựng cluster từ feature vectors và lưu ảnh theo bucket."""
    hashtable = defaultdict(list)
    ht = HashTable.HashTable(NUM_BUCKETS, 64, features.shape[1])

    print("Bắt đầu thêm vector vào HashTable...",flush=True)
    for i, vec in enumerate(features): 
        hash_key = ht.hashFunction(vec.tolist()) 
        print(hash_key)
        img_path = os.path.join(img_folder, filenames[i])
        if not os.path.exists(img_path):
            continue

        # Đánh giá ảnh mới
        sharpness = evaluate_sharpness(img_path)
        color_score = evaluate_colorfulness(img_path)

        if hash_key not in hashtable:
            # Chưa có ảnh nào trong bucket này
            hashtable[hash_key] = {
                'path': img_path,
                'sharpness': sharpness,
                'color': color_score
            }
        else:
            # So sánh với ảnh hiện có
            current = hashtable[hash_key]
            better_path, _ = select_best_image_in_group(
                current['path'], img_path
            )
            # Nếu ảnh mới tốt hơn → thay thế
            if better_path == img_path:
                hashtable[hash_key] = {
                    'path': img_path,
                    'sharpness': sharpness,
                    'color': color_score
                }

    # Xóa thư mục cũ và tạo mới
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR, exist_ok=True)

    print("Bắt đầu lưu ảnh tốt nhất vào từng bucket...")
    for bucket_id, info in hashtable.items():
        cluster_path = os.path.join(CLUSTER_DIR, f"bucket_{bucket_id}")
        os.makedirs(cluster_path, exist_ok=True)
        shutil.copy(info['path'], cluster_path)
        print(info['path'])

    print(f" Đã tạo {len(hashtable)} bucket, mỗi bucket chứa 1 ảnh tốt nhất.")
    return hashtable
        
def build_clusters_min_hash(features, filenames, img_folder):
    """
    Xây dựng cluster bằng MinHash cho tập ảnh.
    -----------------------------------------
    features: np.ndarray (n_samples, 2048)
        Vector đặc trưng trích từ ResNet.
    filenames: List[str]
        Tên file ảnh tương ứng với từng vector.
    img_folder: str
        Đường dẫn chứa ảnh gốc.
    """

    print("⚙️  Khởi tạo MinHash...")
    mh = MinHash.MinHash(NUM_PLANES)

    print("📊 Chuẩn hóa đặc trưng (L2)...")
    normalized_features = l2_normalize(features)
    n_samples = normalized_features.shape[0]

    print(f"[INFO] Bắt đầu tính MinHash signatures cho {n_samples} ảnh...")
    signatures = mh.computeSignatures(normalized_features.tolist(), useMedianThreshold=False)

    # ---- Tạo HashTable ----
    hashtable = defaultdict(list)
    for i, sig in enumerate(signatures):
        # Dùng signature dạng bit list -> chuyển sang tuple để làm key hashable
        key = int(''.join(map(str, sig)), 2)
        hashtable[key].append(filenames[i])
        print(filenames[i],i)
    print(f"🔍 Tổng số bucket tạo ra: {len(hashtable)}")

    # ---- Merge các bucket tương tự nhau ----
    print("🔗 Gom nhóm ảnh tương tự theo Hamming distance...")
    groups = merge_similar_buckets(hashtable, threshold=580)
    print(f"✅ Đã tạo {len(groups)} cụm ảnh (threshold = 5).")

    # ---- Lưu ảnh theo nhóm ----
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR)

    print("💾 Bắt đầu lưu ảnh vào thư mục cụm...")
    for group_id, group in enumerate(groups):
        cluster_path = os.path.join(CLUSTER_DIR, f"group_{group_id:03d}")
        os.makedirs(cluster_path, exist_ok=True)
        for fname in group:
            src = os.path.join(img_folder, fname)
            dst = os.path.join(cluster_path, os.path.basename(fname))
            if os.path.exists(src):
                shutil.copy(src, dst)

    print(f"🎉 Hoàn tất! Đã tạo {len(groups)} cụm ảnh trong thư mục '{CLUSTER_DIR}/'.")
    return hashtable

def build_clusters_sim_hash(features, filenames, img_folder, threshold=13):
    """
    features: np.ndarray shape (N, D)
    filenames: list of filenames (len N)
    img_folder: folder path where images are stored
    threshold: hamming-distance threshold to merge buckets
    use_normalize_for_training: nếu True thì train IDF trên vector đã L2-normalized
    """
    print("🔹 Chuẩn bị dữ liệu...")

    # Bạn có thể train IDF trên dữ liệu gốc hoặc trên bản chuẩn hóa — tuỳ chiến lược.
    train_features = l2_normalize(features)

    # Chuyển sang list để binding C++ chấp nhận
    all_features_list = train_features.tolist()

    # Khởi tạo SimHash (C++ binding)
    ht = SimHash.SimHash(128) 
    # ----- BƯỚC QUAN TRỌNG: train IDF -----
    print("[INFO] Huấn luyện IDF trên toàn bộ tập feature vectors...")
    ht.IDF(all_features_list)   # bắt buộc gọi trước khi hash nếu C++ dùng TF-IDF

    # Nếu bạn muốn chuẩn hóa khi băm, dùng L2-normalize
    normalized_features = l2_normalize(features)

    hashtable = defaultdict(list)
    print("[INFO] Bắt đầu tính SimHash cho từng vector...")
    for i, vec in enumerate(normalized_features):
        # chuyển sang list cho binding
        feature_list = vec.tolist()
        hash_key = ht.hashFunction(feature_list)
        hashtable[hash_key].append(filenames[i])
        if (i + 1) % 200 == 0:
            print(f"  → Đã xử lý {i+1}/{len(filenames)} ảnh")

    # Xóa và tạo folder kết quả
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR, exist_ok=True)

    print("🔍 Gom nhóm bucket gần nhau theo Hamming distance...")
    groups = merge_similar_buckets(hashtable, threshold=threshold)
    print(f"✅ Đã tạo {len(groups)} cụm ảnh (threshold={threshold}).")

    print("💾 Lưu ảnh vào thư mục cụm...")
    for gid, group in enumerate(groups):
        cluster_path = os.path.join(CLUSTER_DIR, f"group_{gid:03d}")
        os.makedirs(cluster_path, exist_ok=True)
        for fname in group:
            src = os.path.join(img_folder, fname)
            dst = os.path.join(cluster_path, os.path.basename(fname))
            if os.path.exists(src):
                shutil.copy(src, dst)

    print(f"🎯 Hoàn tất: {len(hashtable)} hash bucket → {len(groups)} cụm.")
    return hashtable, groups

def build_bloom_clusters(features, filenames, img_folder, threshold =7, cluster_dir=CLUSTER_DIR):
    """
    Gom nhóm ảnh dựa trên Bloom Filter hash.
    - Mỗi ảnh có m hashValues (BloomFilter)
    - Với mỗi hashValue, thêm filename vào hashtable[hashValue]
    - Gom nhóm theo Hamming distance ±threshold trên từng hashValue
    """
    ht = BloomFilter.BloomFilter(36, features.shape[1],9)
    normalized_features = l2_normalize(features)

    # 2️⃣ Tính hashValues và lưu vào hashtable
    hashtable = defaultdict(set)  # key = hashValue, value = list filename
    pairs = defaultdict(list)

    print("[INFO] Tính hashValues từ BloomFilter...")
    for i, vec in enumerate(normalized_features):
        hash_values = ht.hashFunction(vec.tolist())  # list<size_t> với m hash
        for h in hash_values:
            hashtable[h].add(filenames[i])  # Bloom Filter: đánh dấu từng hashValue riêng
        pairs[filenames[i]] = hash_values
    print(f"✅ Đã đánh dấu {len(pairs)} ảnh trong BloomFilter.")

    # 3️⃣ Gom nhóm ảnh gần giống  
    grouped = set()
    groups = []

    print(f"[INFO] Gom nhóm theo Hamming distance ≤ {threshold}...")
    for fname_i in filenames:
        if fname_i in grouped:
            continue  # ảnh đã vào nhóm nào rồi thì bỏ qua
        hash_i = pairs[fname_i]
        group = [fname_i]
        grouped.add(fname_i)

        # Kiểm tra các ảnh khác dựa trên hashtable từng hashValue
        for h in hash_i:
            for f in hashtable[h]:
                if f not in grouped and f != fname_i:
                    hash_j = pairs[f]
                    total_distance = 0
                    for i in range(9):  # từ 0 đến 8
                        total_distance += hamming_distance(hash_i[i], hash_j[i])
                    if total_distance <= threshold:
                        group.append(f)
                        grouped.add(f)


        groups.append(group)

    print(f"✅ Tạo {len(groups)} cụm ảnh (theo threshold={threshold}).")

    # 4️⃣ Lưu ảnh ra thư mục
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR, exist_ok=True)

    print("[INFO] Lưu ảnh vào thư mục cụm...")
    for gid, group in enumerate(groups):
        cluster_path = os.path.join(cluster_dir, f"group_{gid:03d}")
        os.makedirs(cluster_path, exist_ok=True)
        for fname in group:
            src = os.path.join(img_folder, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(cluster_path, fname))

    print(f"💾 Hoàn tất — {len(groups)} cụm được lưu trong '{cluster_dir}/'")

    return hashtable, groups

def build_clusters(ht, features, filenames, img_folder, threshold=5, cluster_dir= CLUSTER_DIR):
    """
    Gom nhóm ảnh dựa trên loại hash được truyền vào (SimHash, MinHash, BloomFilter, HashTable,...)
    ht: đối tượng hash đã khởi tạo (vd: SimHash.SimHash(128), MinHash.MinHash(32), ...)
    features: np.ndarray (n_samples, feature_dim)
    filenames: list[str] tên file tương ứng
    img_folder: thư mục chứa ảnh gốc
    threshold: ngưỡng Hamming distance để gộp nhóm
    cluster_dir: nơi lưu kết quả các nhóm
    """
    normalized_features = l2_normalize(features)
    n_samples = len(filenames)

    print(f"Bắt đầu hashing cho {n_samples} ảnh...")

    #kiểm tra thuộc tính của ht
    if type(ht).__name__ == "SimHash": 
        hashtable = defaultdict(list)
        print("SimHash")
        ht.IDF(normalized_features.tolist())
        for i, vec in enumerate(normalized_features):
            h = ht.hashFunction(vec.tolist())
            hashtable[h].append(filenames[i])

    elif type(ht).__name__ == "MinHash":
        hashtable = defaultdict(list)
        print("MinHash")
        signatures = ht.computeSignatures(normalized_features.tolist(), useMedianThreshold=False)
        for i, sig in enumerate(signatures):
            key = int(''.join(map(str, sig)), 2)
            hashtable[key].append(filenames[i])

    elif type(ht).__name__ == "BloomFilter":
        hashtable = defaultdict(set)
        print("BloomFilter")
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
        print("HashTable")
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