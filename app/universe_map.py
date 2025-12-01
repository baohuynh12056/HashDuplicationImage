from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_universe_map(
    features: np.ndarray,
    filenames: List[str],
    clustering_results: Dict[str, Any],
    quality_scores: Dict[str, Dict] = None,
) -> List[Dict]:
    """
    Generate Universe Map with separate Global vs Local scaling logic.
    """
    if features.shape[0] == 0:
        return []

    # 1. Normalize & PCA
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(features_normalized)

    # --- NEW LOGIC: SEPARATE CLUSTER SCALE & POINT SPREAD ---
    # Cấu hình khoảng cách
    GLOBAL_SPREAD = 10.0  # Khoảng cách giữa các cụm
    LOCAL_EXPLOSION = 30.0  # Độ bung lụa của các điểm trong cùng 1 cụm

    groups = clustering_results.get("groups", {})

    # Tạo map index để truy cập nhanh
    filename_to_idx = {
        name.split("/")[-1]: i for i, name in enumerate(filenames)
    }
    final_coords = np.zeros_like(coords_3d)
    processed_indices = set()

    # Xử lý từng cụm
    for cluster_name, paths in groups.items():
        # Lấy index của các ảnh trong cụm này
        indices = []
        for p in paths:
            fname = p.split("/")[-1]
            if fname in filename_to_idx:
                indices.append(filename_to_idx[fname])

        if not indices:
            continue

        # Tính tâm của cụm (Centroid)
        cluster_points = coords_3d[indices]
        centroid = np.mean(cluster_points, axis=0)

        # Di chuyển từng điểm
        for idx in indices:
            original_pos = coords_3d[idx]
            # Vector từ tâm đến điểm
            offset = original_pos - centroid

            # Công thức mới:
            # Vị trí Mới = (Tâm Cụm * Global) + (Độ Lệch * Local)
            final_coords[idx] = (centroid * GLOBAL_SPREAD) + (
                offset * LOCAL_EXPLOSION
            )
            processed_indices.add(idx)

    # Xử lý các điểm nhiễu (không thuộc cụm nào)
    for i in range(len(coords_3d)):
        if i not in processed_indices:
            # Điểm lẻ loi thì cứ giãn nở đều
            final_coords[i] = coords_3d[i] * (GLOBAL_SPREAD * 1.5)

    # --- END NEW LOGIC ---

    # 4. Create filename to cluster mapping
    filename_to_cluster = {}
    for cluster_name, file_list in groups.items():
        for filepath in file_list:
            base_name = filepath.split("/")[-1]
            filename_to_cluster[base_name] = cluster_name
            filename_to_cluster[filepath] = cluster_name
            name_no_ext = (
                base_name.rsplit(".", 1)[0] if "." in base_name else base_name
            )
            filename_to_cluster[name_no_ext] = cluster_name

    # 5. Create quality lookup
    quality_lookup = {}
    if quality_scores:
        for cluster_name, cluster_data in quality_scores.items():
            for img_info in cluster_data.get("images", []):
                path = img_info["path"]
                base_name = path.split("/")[-1]
                quality_lookup[base_name] = img_info["scores"]["total"]

    # 6. Build result list
    result = []

    for i, (x, y, z) in enumerate(final_coords):
        if i >= len(filenames):
            break

        original_filename = filenames[i]
        base_name = original_filename.split("/")[-1]
        name_no_ext = (
            base_name.rsplit(".", 1)[0] if "." in base_name else base_name
        )

        cluster = None
        if base_name in filename_to_cluster:
            cluster = filename_to_cluster[base_name]
        elif original_filename in filename_to_cluster:
            cluster = filename_to_cluster[original_filename]
        elif name_no_ext in filename_to_cluster:
            cluster = filename_to_cluster[name_no_ext]

        if cluster is None:
            cluster = "Noise/Unique"
            final_path = original_filename
        else:
            final_path = f"{cluster}/{base_name}"

        quality = quality_lookup.get(base_name, 50)

        result.append(
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "filename": base_name,
                "path": final_path,
                "cluster": cluster,
                "quality": float(quality),
            }
        )

    return result
