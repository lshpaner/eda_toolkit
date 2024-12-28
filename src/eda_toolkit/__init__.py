# Import package modules
from .main import *
from .ml_eda import *
from .art import *

import os
import sys
import builtins

# Detailed Documentation
detailed_doc = """
Welcome to EDA Toolkit, a collection of utility functions designed to streamline 
your exploratory data analysis (EDA) tasks. This repository offers tools for 
directory management, some data preprocessing, reporting, visualizations, and 
more, helping you efficiently handle various aspects of data manipulation and 
analysis.

PyPI: https://pypi.org/project/eda-toolkit/
Documentation: https://lshpaner.github.io/eda_toolkit

Authors: Leonid Shpaner, Oscar Gil

Acknowledgements

We would like to express our deepest gratitude to Dr. Ebrahim Tarshizi, our 
mentor during our time in the University of San Diego M.S. Applied Data Science 
Program. His unwavering dedication and mentorship played a pivotal role in our 
academic journey, guiding us to successfully graduate from the program and 
pursue successful careers as data scientists.

We also extend our thanks to the Shiley-Marcos School of Engineering at the 
University of San Diego for providing an exceptional learning environment and 
supporting our educational endeavors.


Version: 0.0.15
"""

# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc

# Metadata
__version__ = "0.0.15"
__author__ = "Leonid Shpaner, Oscar Gil"
__email__ = "lshpaner@ucla.edu; info@oscargildata.com"


# Define the custom help function
def custom_help(obj=None):
    """
    Custom help function to dynamically include ASCII art in help() output.
    """
    if (
        obj is None or obj is sys.modules[__name__]
    ):  # When `help()` is called for this module
        print(eda_toolkit_logo)  # Print ASCII art first
        print(detailed_doc)  # Print the detailed documentation
    else:
        original_help(obj)  # Use the original help for other objects


# Backup the original help function
original_help = builtins.help

# Override the global help function in builtins
builtins.help = custom_help
