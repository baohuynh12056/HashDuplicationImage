# Yêu cầu phiên bản CMake tối thiểu
cmake_minimum_required(VERSION 3.15)

# Tên dự án C++
project(MyHash)

# Đặt tiêu chuẩn C++ (khuyến nghị dùng C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Tìm Python phiên bản 3.10 (phải khớp với Dockerfile)
find_package(Python 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)

# 2. Tìm pybind11
# Dockerfile sẽ cung cấp đường dẫn cho lệnh này
find_package(pybind11 REQUIRED)

# 3. Tạo module Python
# Tên 'MyHash' phải khớp với PYBIND11_MODULE trong binding.cpp
pybind11_add_module(MyHash
    # File binding
    src/binding.cpp

    # Các file C++ của bạn
    src/bloom_filter.cpp
    src/hash_table.cpp
    src/min_hash.cpp
    src/MurmurHash3.cpp
    src/sim_hash.cpp
)

# Chỉ định thư mục chứa header để .cpp tìm thấy .h
target_include_directories(MyHash PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 4. (Tùy chọn) Phần này để 'cài đặt' nếu bạn build local
# Trong Dockerfile, chúng ta chỉ cần copy file build
install(TARGETS MyHash
    LIBRARY DESTINATION ${Python_SITELIB}
    COMPONENT MyHash
)