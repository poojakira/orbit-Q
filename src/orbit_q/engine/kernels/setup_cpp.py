from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "orbit_q.engine.kernels.orbit_q_cpp",
        ["src/orbit_q/engine/kernels/fusion_kernel.cpp"],
        # Example: cxx_std=11
    ),
]

setup(
    name="orbit_q_cpp",
    author="Pooja Kiran",
    description="Orbit-Q C++ Optimized Kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
