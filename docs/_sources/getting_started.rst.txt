.. _getting_started:   

.. KFRE Python Library Documentation documentation master file, created by
   sphinx-quickstart on Thu May  2 15:44:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: ../assets/eda_toolkit_logo.svg
   :alt: EDA Toolkit Logo
   :align: left
   :width: 250px

.. raw:: htmlN

   </div>

.. raw:: html
   
   <div style="height: 106px;"></div>


Welcome to the EDA Toolkit Python Library Documentation!
========================================================
.. note::
   This documentation is for ``eda_toolkit`` version ``0.0.1b0``.


The ``eda_toolkit`` is a comprehensive library designed to streamline and 
enhance the process of Exploratory Data Analysis (EDA) for data scientists, 
analysts, and researchers. This toolkit provides a suite of functions and 
utilities that facilitate the initial investigation of datasets, enabling users 
to quickly gain insights, identify patterns, and uncover underlying structures 
in their data.

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
The ``eda_toolkit`` library is designed to simplify and automate many of the 
repetitive and time-consuming tasks associated with EDA. It provides a wide 
range of functionalities, including:

- **Data Cleaning:** Functions to handle missing values, outliers, and data type conversions.  
- **Data Visualization:** Tools to create various types of plots and charts that help in visualizing the data distributions, relationships, and trends.  
- **Descriptive Statistics:** Methods to compute summary statistics, such as mean, median, standard deviation, and quantiles.  


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

Additionally, ``kfre`` depends on the following packages, which will be automatically installed when you install ``eda_toolkit``:

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


