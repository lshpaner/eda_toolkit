.. _getting_started:   

.. KFRE Python Library Documentation documentation master file, created by
   sphinx-quickstart on Thu May 2 15:44:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: ../assets/eda_toolkit_logo.svg
   :alt: EDA Toolkit Logo
   :align: left
   :width: 300px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 100px;"></div>

\


Welcome to the EDA Toolkit Python Library Documentation!
========================================================
.. note::
   This documentation is for ``eda_toolkit`` version ``0.0.6``.


The ``eda_toolkit`` is a comprehensive library designed to streamline and 
enhance the process of Exploratory Data Analysis (EDA) for data scientists, 
analysts, and researchers. This toolkit provides a suite of functions and 
utilities that facilitate the initial investigation of datasets, enabling users 
to quickly gain insights, identify patterns, and uncover underlying structures 
in their data.

Project Links
---------------

1. `PyPI Page <https://pypi.org/project/eda_toolkit/>`_  

2. `GitHub Repository <https://github.com/lshpaner/eda_toolkit>`_


What is EDA?
-------------

Exploratory Data Analysis (EDA) is a crucial step in the data science workflow. 
It involves various techniques to summarize the main characteristics of the data, 
often with visual methods. EDA helps in understanding the data better, identifying 
anomalies, discovering patterns, and forming hypotheses. This process is essential 
before applying any machine learning models, as it ensures the quality and relevance 
of the data.


Purpose of EDA Toolkit
-----------------------
The ``eda_toolkit`` library is a comprehensive suite of tools designed to 
streamline and automate many of the tasks associated with Exploratory Data 
Analysis (EDA). It offers a broad range of functionalities, including:

- **Data Management:** Tools for managing directories, generating unique IDs, 
  standardizing dates, and handling common DataFrame manipulations.
- **Data Cleaning:** Functions to address missing values, remove outliers, and 
  correct formatting issues, ensuring data is ready for analysis.
- **Data Visualization:** A variety of plotting functions, including KDE 
  distribution plots, stacked bar plots, scatter plots with optional best fit 
  lines, and box/violin plots, to visually explore data distributions, 
  relationships, and trends.
- **Descriptive and Summary Statistics:** Methods to generate comprehensive 
  reports on data types, summary statistics (mean, median, standard deviation, 
  etc.), and to summarize all possible combinations of specified variables.
- **Reporting and Export:** Features to save DataFrames to Excel with 
  customizable formatting, create contingency tables, and export generated 
  plots in multiple formats.
 


Key Features
-------------

- **Ease of Use:** The toolkit is designed with simplicity in mind, offering intuitive and easy-to-use functions.  
- **Customizable:** Users can customize various aspects of the toolkit to fit their specific needs.  
- **Integration:** Seamlessly integrates with popular data science libraries such as ``Pandas``, ``NumPy``, ``Matplotlib``, and ``Seaborn``.  
- **Documentation and Examples:** Comprehensive documentation and examples to help users get started quickly and effectively.  

.. _prerequisites:   

Prerequisites
-------------
Before you install ``eda_toolkit``, ensure your system meets the following requirements:

- **Python**: version ``3.7.4`` or higher is required to run ``eda_toolkit``.

Additionally, ``eda_toolkit`` depends on the following packages, which will be automatically installed when you install ``eda_toolkit``:

- ``numpy``: version ``1.21.6`` or higher
- ``pandas``: version ``1.3.5`` or higher
- ``matplotlib``: version ``3.5.3`` or higher
- ``seaborn``: version ``0.12.2`` or higher
- ``jinja2``: version ``3.1.4`` or higher
- ``xlsxwriter``: version ``3.2.0`` or higher

.. _installation:

Installation
-------------

You can install ``eda_toolkit`` directly from PyPI:

.. code-block:: bash

    pip install eda_toolkit


