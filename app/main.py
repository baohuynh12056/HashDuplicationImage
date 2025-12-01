import io
import os
import shutil
import time
import uuid
import zipfile
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# --- IMPORT MODULES ---
from feature_extractors import ResNetExtractor
from perceptual_clustering import HierarchicalClustering
from pydantic import BaseModel
from quality_scorer import ImageQualityScorer
from universe_map import generate_universe_map

# --- THỬ IMPORT MyHash C++ ---
try:
    import MyHash

    print("✓ Đã tìm thấy module C++ MyHash.")
except ImportError:
    MyHash = None
    print("⚠️ Không tìm thấy module MyHash. Sẽ sử dụng Python Fallback.")

# --- CẤU HÌNH APP ---
app = FastAPI(title="Image Deduplicator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- KHỞI TẠO GLOBAL MODELS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang khởi tạo ResNet50 trên thiết bị: {DEVICE}...")
extractor = ResNetExtractor(DEVICE)
scorer = ImageQualityScorer()

# In-memory Session Storage
SESSIONS: Dict[str, Dict[str, Any]] = {}
TEMP_DIR = "temp_data"
os.makedirs(TEMP_DIR, exist_ok=True)


# --- PYDANTIC MODELS ---
class DeleteRequest(BaseModel):
    session_id: str
    image_paths: List[str]


class SmartCleanupRequest(BaseModel):
    session_id: str
    cluster_name: str
    image_to_keep: str


class DeleteGroupRequest(BaseModel):
    session_id: str
    cluster_name: str


class MoveRequest(BaseModel):
    session_id: str
    image_paths: List[str]
    target_cluster: str


# --- ROUTES ---


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.post("/api/upload-session")
async def upload_session_files(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    saved_paths = []
    for file in files:
        if not file.filename:
            continue
        safe_filename = file.filename.replace("..", "").replace("/", "_")
        file_path = os.path.join(session_dir, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)

    SESSIONS[session_id] = {
        "files": saved_paths,
        "results": None,
        "features": None,
        "filenames": [os.path.basename(p) for p in saved_paths],
    }
    return {
        "session_id": session_id,
        "file_count": len(saved_paths),
        "message": "Ready.",
    }


@app.post("/api/run-clustering")
async def process_clustering(
    session_id: str = Form(...), algorithm: str = Form(...)
):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")

    session_data = SESSIONS[session_id]
    image_paths = session_data["files"]
    filenames = session_data["filenames"]

    if not image_paths:
        raise HTTPException(400, "Session empty")

    start_time = time.time()
    perf_metrics = {}

    # 1. Feature Extraction (ResNet50)
    print(f"[{session_id}] Extracting features...")
    t0 = time.time()

    # Check cache
    if session_data.get("features") is None:
        features, valid_filenames = extractor.extract(image_paths, filenames)
        session_data["features"] = features
        session_data["valid_filenames"] = valid_filenames
    else:
        features = session_data["features"]
        valid_filenames = session_data["valid_filenames"]

    valid_paths = [
        p for p in image_paths if os.path.basename(p) in valid_filenames
    ]
    perf_metrics["extraction_time"] = time.time() - t0

    # 2. Hashing & Clustering
    print(f"[{session_id}] Running {algorithm} Clustering...")
    t0 = time.time()

    clusterer = HierarchicalClustering(hash_type=algorithm)
    hash_obj = None

    # Khởi tạo MyHash Object (Nếu có module C++)
    if MyHash:
        try:
            if algorithm == "SimHash":
                # ResNet50 (2048 dims) -> 128 bit hash
                # C++ Signature: SimHash(int bits)
                hash_obj = MyHash.SimHash(128)

            elif algorithm == "MinHash":
                # C++ Signature: MinHash(int input_dim, int num_hashes)
                hash_obj = MyHash.MinHash(2048, 128)

            elif algorithm == "HashTable":
                # C++ Signature: HashTable(int bits) hoặc tương tự
                # Giả định dùng 64 bit cho HashTable bucket
                if hasattr(MyHash, "HashTable"):
                    hash_obj = MyHash.HashTable(64)
                else:
                    print(
                        "⚠️ MyHash.HashTable not found, using Python fallback."
                    )

            elif algorithm == "BloomFilter":
                # BloomFilter(int input_dim, int size, float error_rate)
                if hasattr(MyHash, "BloomFilter"):
                    hash_obj = MyHash.BloomFilter(2048, 20000, 0.01)

        except Exception as e:
            print(f"❌ Error initializing MyHash algorithm: {e}")
            print("➡️ Falling back to Python implementation if available.")
            hash_obj = None  # Để perceptual_clustering dùng Python fallback

    # Chạy Clustering (tự động tìm threshold valley)
    groups = clusterer.cluster(hash_obj, features, valid_paths)
    perf_metrics["clustering_time"] = time.time() - t0

    groups_dict = {f"cluster_{i+1:03d}": g for i, g in enumerate(groups)}

    # 3. Quality Scoring
    print(f"[{session_id}] Scoring quality...")
    t0 = time.time()
    quality_scores = {}
    for name, paths in groups_dict.items():
        scored_imgs = scorer.score_cluster(paths)
        quality_scores[name] = {
            "images": [
                {
                    "path": p,
                    "scores": s,
                    "is_best": i == 0,
                    "quality_color": scorer.get_quality_color(s["total"]),
                }
                for i, (p, s) in enumerate(scored_imgs)
            ],
            "best_image": scored_imgs[0][0] if scored_imgs else None,
        }
    perf_metrics["scoring_time"] = time.time() - t0

    # 4. Generate Map Data
    map_data = []
    try:
        clustering_results = {
            "groups": groups_dict,
            "total_images": len(valid_paths),
        }
        print(f"[{session_id}] Generating Universe Map...")
        map_data = generate_universe_map(
            features, valid_paths, clustering_results, quality_scores
        )
    except Exception as e:
        print(f"⚠️ Universe Map Generation Failed: {e}")
        map_data = []

    final_results = {
        "session_id": session_id,
        "results": {
            "groups": groups_dict,
            "total_images": len(valid_paths),
            "duplicate_count": sum(len(g) for g in groups),
        },
        "quality_scores": quality_scores,
        "universe_map": map_data,
        "performance": perf_metrics,
    }

    SESSIONS[session_id]["results"] = final_results
    print(f"[{session_id}] Done in {time.time() - start_time:.2f}s")
    return final_results


# --- HELPER ENDPOINTS ---


@app.get("/api/results/{session_id}/clusters/{file_path:path}")
async def get_image(session_id: str, file_path: str):
    if ".." in file_path:
        raise HTTPException(400)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    full_path = os.path.join(TEMP_DIR, session_id, os.path.basename(file_path))
    if os.path.exists(full_path):
        return FileResponse(full_path)
    raise HTTPException(404)


@app.post("/api/delete-images")
async def delete_images(req: DeleteRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404)
    deleted = []
    for p in req.image_paths:
        if os.path.exists(p):
            try:
                os.remove(p)
                deleted.append(p)
            except:
                pass

    groups = SESSIONS[req.session_id]["results"]["results"]["groups"]
    for k, v in groups.items():
        groups[k] = [f for f in v if f not in deleted]
    SESSIONS[req.session_id]["results"]["results"]["groups"] = {
        k: v for k, v in groups.items() if v
    }
    return {"status": "success", "deleted": deleted}


@app.post("/api/smart-cleanup")
async def smart_cleanup(req: SmartCleanupRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404)
    groups = SESSIONS[req.session_id]["results"]["results"]["groups"]
    if req.cluster_name not in groups:
        raise HTTPException(404)

    current = groups[req.cluster_name]
    to_del = [f for f in current if f != req.image_to_keep]
    for p in to_del:
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

    groups[req.cluster_name] = [req.image_to_keep]
    return {"status": "success", "image_kept": req.image_to_keep}


@app.post("/api/delete-group")
async def delete_group(req: DeleteGroupRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404)
    groups = SESSIONS[req.session_id]["results"]["results"]["groups"]
    if req.cluster_name in groups:
        for p in groups[req.cluster_name]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
        del groups[req.cluster_name]
    return {"status": "success"}


@app.post("/api/move-images")
async def move_images(req: MoveRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404)
    groups = SESSIONS[req.session_id]["results"]["results"]["groups"]

    for k, v in groups.items():
        groups[k] = [f for f in v if f not in req.image_paths]

    if req.target_cluster not in groups:
        groups[req.target_cluster] = []
    groups[req.target_cluster].extend(req.image_paths)

    SESSIONS[req.session_id]["results"]["results"]["groups"] = {
        k: v for k, v in groups.items() if v
    }
    return {"status": "success"}


@app.get("/api/download-results/{session_id}")
async def download_results(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404)
    session_dir = os.path.join(TEMP_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(404)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(session_dir):
            for file in files:
                if os.path.exists(os.path.join(root, file)):
                    zf.write(os.path.join(root, file), arcname=file)
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={session_id}.zip"
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
