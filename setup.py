from setuptools import setup
from setuptools.extension import Extension
import pybind11

# ---- Module 1: HashTable ----
hash_table_ext = Extension(
    "hash_table_py",
    sources=[
        "Hash Structure/Source/hash_table.cpp",
        "Hash Structure/Source/hash_table_binding.cpp",
    ],
    include_dirs=[
        pybind11.get_include(),
        "Hash Structure/Header",
    ],
    language="c++",
    extra_compile_args=["-std=c++17"],
)

# ---- Module 2: SimHash ----
simhash_ext = Extension(
    "simhash_py",
    sources=[
        "Hash Structure/Source/sim_hash.cpp",
        "Hash Structure/Source/sim_hash_binding.cpp",
    ],
    include_dirs=[
        pybind11.get_include(),
        "Hash Structure/Header",
    ],
    language="c++",
    extra_compile_args=["-std=c++17"],
)

# ---- Setup ----
setup(
    name="hash_table_py",
    version="0.1",
    ext_modules=[hash_table_ext, simhash_ext],
    setup_requires=["pybind11"],
)
