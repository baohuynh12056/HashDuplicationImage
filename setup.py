import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "MyHash",
        sources=[
            "Hash Structure/Source/binding.cpp",
            "Hash Structure/Source/hash_table.cpp",
            "Hash Structure/Source/min_hash.cpp",
            "Hash Structure/Source/bloom_filter.cpp",
            "Hash Structure/Source/sim_hash.cpp",
            "Hash Structure/Source/MurmurHash3.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            "Hash Structure/Header",
        ],
        language="c++",
    )
]

setup(
    name="MyHash",
    version="0.0",
    ext_modules=ext_modules,
)
