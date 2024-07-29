.. _usage_guide:   

.. _target-link:

.. raw:: html

   <div class="no-click">

.. image:: ../assets/eda_toolkit_logo.svg
   :alt: EDA Toolkit Logo
   :align: left
   :width: 250px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 106px;"></div>

\

Description
===========

This guide provides detailed instructions and examples for using the functions 
provided in the ``eda_toolkit`` library and how to use them effectively in your projects.

For the ensuing examples, we will leverage the Census Income Data (1994) from
the UCI Machine Learning Repository. This dataset provides a rich source of
information for demonstrating the functionalities of the ``eda_toolkit``. You can
find more details and download the dataset from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Census+Income>`_.


Path Directories
=================

.. function:: ensure_directory(path)

    Ensure that the directory exists. If not, create it.

    :param path: The path to the directory that needs to be ensured.
    :type path: str

    :returns: None


The ``ensure_directory`` function is a utility designed to facilitate the 
management of directory paths within your project. When working with data 
science projects, it is common to save and load data, images, and other 
artifacts from specific directories. This function helps in making sure that 
these directories exist before any read/write operations are performed. If 
the specified directory does not exist, the function creates it. If it 
already exists, it does nothing, thus preventing any errors related to 
missing directories.


**Example Usage**

In the example below, we demonstrate how to use the ``ensure_directory`` function 
to verify and create directories as needed. This example sets up paths for data and 
image directories, ensuring they exist before performing any operations that depend on them.

First, we define the base path as the parent directory of the current directory. 
The ``os.pardir`` constant, equivalent to ``..``, is used to navigate up one 
directory level. Then, we define paths for the data directory and data output 
directory, both located one level up from the current directory. Next, we set 
paths for the PNG and SVG image directories, located within an ``images`` folder 
in the parent directory. Using the `ensure_directory` function, we then verify 
that these directories exist. If any of the specified directories do not exist, 
the function creates them.

.. code-block:: python


    import eda_toolkit # import main library
    import os # import operating system for dir

.. code-block:: python

    base_path = os.path.join(os.pardir)

    # Go up one level from 'notebooks' to the parent directory, 
    # then into the 'data' folder
    data_path = os.path.join(os.pardir, "data")
    data_output = os.path.join(os.pardir, "data_output")

    # create image paths
    image_path_png = os.path.join(base_path, "images", "png_images")
    image_path_svg = os.path.join(base_path, "images", "svg_images")

    # Use the function to ensure the 'data' directory exists
    eda_toolkit.ensure_directory(data_path)
    eda_toolkit.ensure_directory(data_output)
    eda_toolkit.ensure_directory(image_path_png)
    eda_toolkit.ensure_directory(image_path_svg)

**Output**

.. code-block:: python

    Created directory: ../data
    Created directory: ../data_output
    Created directory: ../images/png_images
    Created directory: ../images/svg_images


.. function:: add_ids(df, column_name="Patient_ID", seed=None)

    Add a column of unique, 9-digit IDs to the dataframe.

    This function sets a random seed and then generates a 9-digit ID for
    each row in the dataframe. The new IDs are added as a new column with
    the specified column name, which is placed as the first column in the dataframe.

    :param df: The dataframe to add IDs to.
    :type df: pd.DataFrame
    :param column_name: The name of the new column for the IDs.
    :type column_name: str
    :param seed: The seed for the random number generator. Defaults to None.
    :type seed: int, optional

    :returns: The updated dataframe with the new ID column.
    :rtype: pd.DataFrame

    **Description**

    The `add_ids` function is used to append a column of unique 9-digit identifiers to a given dataframe. This is particularly useful for creating unique patient or record IDs in datasets. The function allows you to specify a custom column name for the IDs and optionally set a seed for the random number generator to ensure reproducibility.

    **Example Usage**

    In the example below, we demonstrate how to use the `add_ids` function to add a column of unique IDs to a dataframe. We start by importing the necessary libraries and creating a sample dataframe. We then use the `add_ids` function to generate and append a column of 9-digit IDs to the dataframe.

    .. code-block:: python

        import pandas as pd
        import random
        from eda_toolkit import add_ids

        # Create a sample dataframe
        data = {
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        df = pd.DataFrame(data)

        # Add a column of unique IDs
        df_with_ids = add_ids(df, column_name="Patient_ID", seed=42)

        print(df_with_ids)

    This code creates a dataframe with a new column `Patient_ID` containing unique 9-digit identifiers for each row. The `seed` parameter ensures that the IDs are reproducible if the function is run again with the same seed value.





