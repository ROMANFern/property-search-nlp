from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="property-search-nlp",
    version="0.1.0",
    author="Manusha Fernando",
    author_email="manusha@romanfern.com",
    description="NLP-based property search query parser",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ROMANFern/property-search-nlp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires = [
        "openai>=2.9.0",
        "pandas>=2.3.3",
        "numpy>=2.3.5",
        "tabulate>=0.9.0",
    ],
    extras_require = {
        "dev": [
            "pytest>=9.0.1",
            "matplotlib>=3.10.7",
            "seaborn>=0.13.2",
        ],
    },
)