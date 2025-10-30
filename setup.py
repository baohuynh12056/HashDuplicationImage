import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "MyHash",
        sources=[
            "Hash Structure/Source/binding.cpp",
            "Hash Structure/Source/hash_table.cpp",
            "Hash Structure/Source/min_hash.cpp"
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
    version="0.1",
    ext_modules=ext_modules,
)
