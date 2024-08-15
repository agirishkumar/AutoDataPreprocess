from setuptools import setup, find_packages

setup(
    name="AutoDataPreprocess",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "category-encoders",
        "ydata-profiling",
        "SQLAlchemy",
        "umap-learn",
        "requests",
        "statsmodels",
        "scipy",
        "imbalanced-learn"

    ],
    author="Girish Kumar Adari",
    author_email="adari.girishkumar@gmail.com",
    description="A high-level library for automatic preprocessing of tabular data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/agirishkumar/AutoDataPreprocess",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)