from setuptools import setup, find_packages

setup(
    name="eda_toolkit",
    version="0.0.10",
    author="Leonid Shpaner, Oscar Gil",
    author_email="lshpaner@ucla.edu",
    description="A Python library for EDA, including visualizations, directory management, data preprocessing, reporting, and more.",
    long_description=open("README_min.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",  # Type of the long description
    package_dir={"": "src"},  # Directory where your package files are located
    # Automatically find packages in the specified directory
    packages=find_packages(where="src"),
    project_urls={  # Optional
        "Leonid Shpaner's Website": "https://www.leonshpaner.com",
        "Oscar Gil's Website": "https://www.oscargildata.com",
        "Documentation": "https://lshpaner.github.io/eda_toolkit/",
        "Zenodo Archive": "https://zenodo.org/records/13162633",
        "Source Code": "https://github.com/lshpaner/eda_toolkit/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # Classifiers for the package
    python_requires=">=3.7.4",  # Minimum version of Python required
    install_requires=[
        "jinja2>=3.1.4",  # Minimum version of jinja2 required
        "matplotlib>=3.5.3",  # Minimum version of matplotlib required
        "nbformat>=4.2.0",  # Minimum version of nbformat required
        "numpy>=1.21.6",  # Minimum version of numpy required
        "pandas>=1.3.5",  # Minimum version of pandas required
        "plotly>=5.18.0",  # Minimum version of plotly required
        "scikit-learn>=1.0.2",  # Minimum version of scikit-learn required
        "seaborn>=0.12.2",  # Minimum version of seaborn required
        "xlsxwriter>=3.2.0",  # Minimum version of xlsxwriter required
    ],
)
