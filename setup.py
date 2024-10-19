from setuptools import setup, find_packages

setup(
    name="eda_toolkit",
    version="0.0.11a",
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
    python_requires=">=3.7.4",  # Python versions 3.7.4 to 3.11
    install_requires=[
        "jinja2==3.1.4",  # Exact version of Jinja2 required
        "matplotlib>=3.5.3,<=3.9.2",  # Matplotlib capped at 3.9.2
        "nbformat>=4.2.0,<=5.10.4",  # Nbformat capped at 5.10.4
        "numpy>=1.21.6,<=2.1.0",  # Numpy capped at 2.1.0
        "pandas>=1.3.5,<=2.2.2",  # Pandas capped at 2.2.2
        "plotly>=5.18.0,<=5.24.0",  # Plotly capped at 5.24.0
        "scikit-learn>=1.0.2,<=1.5.2",  # Scikit-learn capped at 1.5.2
        "seaborn>=0.12.2,<0.13.0",  # Seaborn capped below 0.13.0
        "xlsxwriter>=3.2.0,<=3.2.0",  # XlsxWriter capped at 3.2.0
        "scipy>=1.7.3,<=1.11.1",  # SciPy capped at 1.11.1
    ],
)
