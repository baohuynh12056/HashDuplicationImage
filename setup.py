from setuptools import setup, Extension
import pybind11
import os

# ==== Common include paths ====
include_dirs = [
    pybind11.get_include(),
    os.path.join("Hash Structure", "Header"),
]

# ==== Helper function to build clean paths ====
def src(*path_parts):
    return os.path.join("Hash Structure", "Source", *path_parts)

# ==== Common compile flags ====
compile_args = ["-O3", "-Wall", "-std=c++17", "-fPIC"]

# ==== Module 1: HashTable ====
hash_table_ext = Extension(
    "hash_table_py",
    sources=[
        src("hash_table.cpp"),
        src("hash_table_binding.cpp"),
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=compile_args,
)

# ==== Module 2: SimHash ====
simhash_ext = Extension(
    "simhash_py",
    sources=[
        src("sim_hash.cpp"),
        src("sim_hash_binding.cpp"),
        src("MurmurHash3.cpp"),
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=compile_args,
)

# ==== Module 3: MinHash ====
minhash_ext = Extension(
    "minhash_py",
    sources=[
        src("min_hash.cpp"),
        src("min_hash_binding.cpp"),
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=compile_args,
)

# ==== Module 4: BloomFilter ====
bloom_ext = Extension(
    "bloom_filter_py",
    sources=[
        src("bloom_filter.cpp"),
        src("bloom_filter_binding.cpp"),
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_compile_args=compile_args,
)

# ==== Setup ====
setup(
    name="hashduplicationimage",
    version="0.3",
    author="Huynh Gia Bao",
    description="Hash-based image deduplication with SimHash, MinHash, BloomFilter, and HashTable",
    ext_modules=[hash_table_ext, simhash_ext, minhash_ext, bloom_ext],
    setup_requires=["pybind11"],
)
