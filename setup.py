from setuptools import setup, find_packages

setup(
    name="fofana",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pyzed",  # ZED SDK Python API
        "torch",  # For CUDA support
        "opencv-python",
    ],
    python_requires=">=3.8",
    author="Tolga Yarkin Usta",
    description="Autonomous Surface Vehicle control software for RoboBoat 2025 competition",
)
