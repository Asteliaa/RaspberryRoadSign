"""Setup configuration for RaspberryRoadSign package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="RaspberryRoadSign",
    version="0.1.0",
    description="Traffic sign detection system using YOLOv11 for Russian/Belarusian roads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RaspberryYolo Team",
    url="https://github.com/Asteliaa/RaspberryRoadSign",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "ultralytics>=8.3.0",
        "opencv-python>=4.8.0",
        "numpy>=2.1.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "mypy>=1.5.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
