
# Deduplicate image based on feature hashing techniques
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
> **Dự án thuộc khuôn khổ bài tập lớn (Assignment) môn Cấu trúc Dữ liệu và Giải thuật (Data Structures and Algorithms) - Học phần mở rộng cho Chương trình Cử nhân Tài năng.**

## 👥 Thành viên thực hiện
Dự án được phát triển bởi nhóm 4 thành viên:
* **Huỳnh Gia Bảo**
* **Nguyễn Gia An**
* **Lại Trần Trí**
* **Nguyễn Hữu Phước**

**Giảng viên hướng dẫn:** TS. Lê Thành Sách
# 📖 Giới thiệu dự án

Dự án này là một hệ thống gom nhóm và loại bỏ ảnh trùng lặp (Image Deduplication) hiệu năng cao. Hệ thống kết hợp sức mạnh của **Deep Learning (ResNet50)** để trích xuất đặc trưng ảnh và các thuật toán **Hashing (C++)** để tìm kiếm tương đồng cực nhanh.



## 🚀 Tính năng nổi bật

* **Robust Feature Extraction:** Sử dụng **ResNet50** (đã loại bỏ lớp FC) kết hợp với kỹ thuật **Test-Time Augmentation (TTA)** phong phú (xoay, lật, nhiễu, làm mờ...) bằng thư viện `Kornia` (trên GPU) và `PIL`. Điều này giúp vector đặc trưng không bị ảnh hưởng bởi ánh sáng, góc chụp hay nhiễu.
* **High Performance Hashing (C++):** Các thuật toán băm (Hashing) được viết bằng C++ và bind qua Python bằng `pybind11`:
    * **SimHash:** Tìm kiếm tương đồng cosine.
    * **MinHash:** Ước lượng độ tương đồng Jaccard.
    * **BloomFilter:** Kiểm tra thành viên tập hợp xác suất.
    * **HashTable:** Gom nhóm chính xác.
* **Auto-Thresholding:** Tự động tìm ngưỡng cắt (threshold) tối ưu dựa trên phân tích "thung lũng" (Valley Detection) của biểu đồ khoảng cách Hamming. (Lưu ý: Chỉ dùng cho HashTable và SimHash)
* **Best Image Selection:** Trong mỗi nhóm ảnh trùng, hệ thống tự động chọn ra ảnh tốt nhất dựa trên độ sắc nét (Laplacian) và độ rực màu (Saturation).
* **FAISS Integration:** Hỗ trợ thư viện FAISS của Facebook để tìm kiếm vector tốc độ cao.

## 🛠 Yêu cầu hệ thống

* **Python:** 3.8 trở lên.
* **Compiler:** C++17 compatible compiler (GCC, Clang, hoặc MSVC trên Windows).
* **GPU:** Khuyến nghị có NVIDIA GPU (CUDA) để trích xuất đặc trưng nhanh hơn với `torch` và `kornia`.

## 📦 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/baohuynh12056/HashDuplicationImage
cd HashDuplicationImage
```
### 2. Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```
### 3. Biên dịch module C++
Dự án sử dụng pybind11 để biên dịch mã nguồn C++ trong thư mục Hash Structure.
```bash
python setup.py build_ext --inplace
```
Sau lệnh này, các file .so (Linux/Mac) hoặc .pyd (Windows) sẽ được tạo ra, cho phép import các thuật toán hash như simhash_py, minhash_py...

### 4. Chạy thử nghiệm
Hệ thống đã tích hợp sẵn file main.py mẫu. Bạn có thể chạy ngay để kiểm tra quá trình hoạt động:
```
python main.py
```
Lưu ý: Chương trình sẽ tự động tạo thư mục img nếu chưa có. Hãy bỏ ảnh vào đó và chạy lại.

Datasets mẫu: https://drive.google.com/drive/folders/1ZninkrJztjI2grmj6bY9__xPBA9UV3wP?usp=sharing
## 🚀 Run with Docker

Ứng dụng hỗ trợ chạy hoàn toàn bằng Docker, không cần cài thêm môi trường Python hay thư viện.

### 🏗️ Build Docker Image

Chạy lệnh sau trong thư mục chứa `Dockerfile`:

```bash
docker build -t hash-duplication-app .
```
### ▶️ Run Container
Sau khi build xong, chạy container:
```bash
docker run -p 8000:8000 hash-duplication-app
```
Copy-paste vào trình duyệt:
```bash
http://localhost:8000
```
## 📂 Cấu trúc dự án
```text
Project/
├── Application/              # Python package chứa logic xử lý chính
├── Hash Structure/           # Mã nguồn C++ (Core Hashing)
│   ├── Header/               # File .h
│   └── Source/               # File .cpp
├── app/                      # Web
├── .gitattributes            # Cấu hình thuộc tính Git
├── .gitignore                # Danh sách file cần bỏ qua khi commit
├── CMakeLists.txt            # File cấu hình CMake cho phần C++
├── Dockerfile                # Build và chạy ứng dụng bằng Docker
├── LICENSE                   # Giấy phép dự án
├── main.py                   # File chạy chính của ứng dụng
├── requirement.txt           # Danh sách thư viện Python cần cài đặt
├── setup.py                  # Script build C++ extension
└── README.md                 # Tài liệu hướng dẫn
```
## ⚡ Hướng dẫn sử dụng
### 1. Trích xuất đặc trưng (Feature Extraction)
Hàm: `mean_extract_image_features_batch_1` (trong Application.resnet)

Chức năng: Đọc ảnh từ thư mục, sử dụng ResNet50 + GPU Augmentation để biến ảnh thành vector đặc trưng.

Tương quan: Kết quả được lưu xuống file .npy (ví dụ features.npy). Các bước sau sẽ đọc từ file này để không phải chạy lại resnet (tiết kiệm rất nhiều thời gian).
```python
# Trích xuất và lưu vào file .npy
features, names = mean_extract_image_features_batch_1(
    img_dir="img",
    feature_file="features.npy",
    name_file="names.npy",
    batch_size=24
)

```
### 2. Phân tích & Tìm ngưỡng (Threshold Analysis)
Trước khi gom nhóm, bạn cần biết "cắt" ở đâu là hợp lý.

- **Hàm:** `analyze_and_plot_distances` (trong Application.cluster)

- **Chức năng:** Tính toán khoảng cách giữa các vector và tìm ra "thung lũng" (valley) trên biểu đồ phân phối để đề xuất ngưỡng cắt (threshold) tối ưu.
Tương quan: Hàm này nhận features từ bước 1 và đối tượng Hash (ví dụ SimHash), trả về con số best_threshold để dùng cho bước 3.
```python
import simhash_py as SimHash
ht = SimHash.SimHash(64) # Cấu hình 64 bit
best_threshold = analyze_and_plot_distances(ht, features)
```
**Lưu ý:** Chỉ dùng cho HashTable và SimHash
### 3. Phân tích & Tìm ngưỡng (Threshold Analysis)
#### Nhánh A: Sử dụng Hash (C++ Backend)
Sử dụng các thuật toán SimHash, MinHash, BloomFilter, HashTable.

- **Hàm:** `build_clusters` (trong Application.cluster)

- **Tương quan**: Nhận features, filenames, đối tượng ht (đã khởi tạo) và threshold (từ bước 2). Hệ thống sẽ copy ảnh vào các thư mục clusters/group_xxx.
```python
build_clusters(ht, features, filenames, img_folder="img", threshold=best_threshold, cluster_dir="clusters_type_hash")
```
#### Nhánh B: Sử dụng FAISS (Vector Search)
Sử dụng thư viện FAISS của Facebook.

- **Hàm:** `build_cluster_faiss`

- **Tương quan:** Chạy độc lập, không cần đối tượng C++ Hash. Phù hợp khi muốn so sánh hiệu năng với Hash truyền thống.
```python
build_cluster_faiss(features, filenames, img_folder="img", cluster_dir="clusters_faiss", threshold=0.8)
```
### 4. Đánh giá hiệu năng (Evaluation)
Sau khi có các thư mục nhóm, các hàm này sẽ chấm điểm độ chính xác.
* **Hàm sử dụng:**
    * `evaluate_precision_recall`: Tính chỉ số chuyên sâu (F1-Score, Precision, Recall) cho từng class.
    * `evaluate_by_image`: Tính Accuracy đơn giản (tỷ lệ ảnh được xếp đúng vào nhóm chủ đạo).
* **Tương quan:** Các hàm này chỉ đọc thư mục kết quả (`cluster_dir`) để tính toán, không tham gia vào quá trình xử lý ảnh.

```python
evaluate_precision_recall("clusters_hash_type")
evaluate_by_image("clusters_hash_type")
```
## 🏁 Lời kết

Dự án **HashDuplicationImage** là sự kết hợp giữa sức mạnh trích xuất đặc trưng của **Deep Learning (ResNet50)** và tốc độ xử lý của các thuật toán **Hashing (C++)**. Mục tiêu của dự án là giải quyết bài toán loại bỏ dữ liệu trùng lặp (Deduplication) trên tập dữ liệu lớn với độ chính xác cao và chi phí tính toán hợp lý.
## Lời cảm ơn

Chúng em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến **TS. Lê Thành Sách**.

Trong suốt quá trình học tập và thực hiện đề tài, thầy đã luôn tận tình hướng dẫn, định hướng tư duy và cung cấp những kiến thức nền tảng quý báu về Cấu trúc Dữ liệu & Giải thuật nâng cao. Những nhận xét, góp ý chuyên môn của thầy đã giúp nhóm tháo gỡ nhiều vướng mắc về mặt kỹ thuật, đồng thời học hỏi thêm được cách tiếp cận vấn đề một cách khoa học và tối ưu hơn.

Dự án này là cơ hội tuyệt vời để chúng em áp dụng những lý thuyết trên lớp vào thực tế. Một lần nữa, nhóm xin chân thành cảm ơn thầy!

## 📚 Tài liệu & Nguồn tham khảo

Dự án được xây dựng dựa trên việc nghiên cứu các tài liệu, bài báo khoa học quốc tế và các dự án mã nguồn mở. Xin gửi lời cảm ơn đến các tác giả:

### 📄 Bài báo khoa học (Scientific Papers)

**Cơ sở lý thuyết & Thuật toán nền tảng**
* **Space/time trade-offs in hash coding with allowable errors** - *Burton H. Bloom* (Communications of the ACM, 1970).
* **On the resemblance and containment of documents** - *Andrei Z. Broder* (IEEE, 1997).
* **Similarity Estimation Techniques from Rounding Algorithms** - *Moses S. Charikar* (Princeton University, 2002).
* **The Automatic Creation of Literature Abstracts** - *H. P. Luhn* (IBM Journal of Research and Development, 1958).

**Trích xuất đặc trưng**
* **Deep Residual Learning for Image Recognition** - *Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun* (CVPR 2016).

**Hashing**
* **Detecting near-duplicates for web crawling** - *Gurmeet Singh Manku, Arvind Jain, Anish Das Sarma* (Google, WWW 2007).
* **Simhash for large scale image retrieval** - *Qin-Zhen Guo et al.* (Applied Mechanics and Materials, 2014).
* **Bloom Filters and Compact Hash Codes for Efficient and Distributed Image Retrieval** - *Andrea Salvi, Simone Ercoli, Marco Bertini, Alberto Del Bimbo* (IEEE ISM, 2016).
* **Large-Scale Query-by-Image Video Retrieval Using Bloom Filters** - *André Araujo et al.* (Stanford University, 2016).
* **Advanced Bloom Filter Based Algorithms for Efficient Approximate Data De-Duplication in Streams** - *Suman K. Bera et al.* (arXiv, 2012).

### 💻 Mã nguồn mở & Thư viện (Open Source)

* **MurmurHash3: A Non-Cryptographic Hash Function** - *Austin Appleby* (2008).
    * Source: [SMHasher Repository](https://github.com/aappleby/smhasher)

---







