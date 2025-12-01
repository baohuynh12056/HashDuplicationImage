# STAGE 1: Môi trường Build C++
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install pybind11

# Copy file CMakeLists.txt
COPY CMakeLists.txt .
# Copy thư mục src
COPY src/ ./src/

RUN cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYBIND11_TEST=OFF \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
RUN cmake --build build --config Release --parallel

# STAGE 2: Môi trường Runtime Python
FROM python:3.10-slim

WORKDIR /app

ENV TORCH_HOME=/app/models
ENV XDG_CACHE_HOME=/app/cache

# Copy requirements.txt (nằm ở root)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY app/preload_models.py .
RUN python preload_models.py

COPY app/ ./app/

# Copy file .so đã build từ stage 1
COPY --from=builder /app/build/MyHash.*.so ./app/

# Chuyển vào thư mục làm việc
WORKDIR /app/app

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]