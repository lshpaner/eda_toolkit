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

Description
===========

This guide provides detailed instructions and examples for using the functions 
provided in the ``eda_toolkit`` library and how to use them effectively in your projects.

For most of the ensuing examples, we will leverage the Census Income Data (1994) from
the UCI Machine Learning Repository [1]_. This dataset provides a rich source of
information for demonstrating the functionalities of the ``eda_toolkit``.


Data Preparation and Management
===============================

Path directories
----------------

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
directory, both located one level up from the current directory. 


Next, we set paths for the PNG and SVG image directories, located within an 
``images`` folder in the parent directory. Using the ``ensure_directory`` 
function, we then verify that these directories exist. If any of the specified 
directories do not exist, the function creates them.

.. code-block:: python

    # import function from library
    from eda_toolkit import ensure_directory 
    import os # import operating system for dir
    

    base_path = os.path.join(os.pardir)

    # Go up one level from 'notebooks' to parent directory, 
    # then into the 'data' folder
    data_path = os.path.join(os.pardir, "data")
    data_output = os.path.join(os.pardir, "data_output")

    # create image paths
    image_path_png = os.path.join(base_path, "images", "png_images")
    image_path_svg = os.path.join(base_path, "images", "svg_images")

    # Use the function to ensure'data' directory exists
    ensure_directory(data_path)
    ensure_directory(data_output)
    ensure_directory(image_path_png)
    ensure_directory(image_path_svg)

**Output**

.. code-block:: python

    Created directory: ../data
    Created directory: ../data_output
    Created directory: ../images/png_images
    Created directory: ../images/svg_images


Adding Unique Identifiers
--------------------------

.. function:: add_ids(df, id_colname="ID", num_digits=9, seed=None, set_as_index=True)

    Add a column of unique IDs with a specified number of digits to the dataframe.

    :param df: The dataframe to add IDs to.
    :type df: pd.DataFrame
    :param id_colname: The name of the new column for the IDs.
    :type id_colname: str
    :param num_digits: The number of digits for the unique IDs.
    :type num_digits: int
    :param seed: The seed for the random number generator. Defaults to ``None``.
    :type seed: int, optional
    :param set_as_index: Whether to set the new ID column as the index. Defaults to ``False``.
    :type set_as_index: bool, optional

    :returns: The updated dataframe with the new ID column.
    :rtype: pd.DataFrame

The ``add_ids`` function is used to append a column of unique identifiers with a 
specified number of digits to a given dataframe. This is particularly useful for 
creating unique patient or record IDs in datasets. The function allows you to 
specify a custom column name for the IDs, the number of digits for each ID, and 
optionally set a seed for the random number generator to ensure reproducibility. 
Additionally, you can choose whether to set the new ID column as the index of the dataframe.

**Example Usage**

In the example below, we demonstrate how to use the ``add_ids`` function to add a 
column of unique IDs to a dataframe. We start by importing the necessary libraries 
and creating a sample dataframe. We then use the ``add_ids`` function to generate 
and append a column of unique IDs with a specified number of digits to the dataframe.

First, we import the pandas library and the ``add_ids`` function from the ``eda_toolkit``. 
Then, we create a sample dataframe with some data. We call the ``add_ids`` function, 
specifying the dataframe, the column name for the IDs, the number of digits for 
each ID, a seed for reproducibility, and whether to set the new ID column as the 
index. The function generates unique IDs for each row and adds them as the first 
column in the dataframe.

.. code-block:: python

    import pandas as pd
    import random
    from eda_toolkit import add_ids

    # Add a column of unique IDs with 9 digits and call it "census_id"
    df = add_ids(
        df=df,
        id_colname="census_id",
        num_digits=9,
        seed=111,
        set_as_index=True, 
    )

**Output**

`First 5 Rows of Census Income Data (Adapted from Kohavi, 1996, UCI Machine Learning Repository)` [1]_

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
    .tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
    .tg .tg-aw21{border-color:#ffffff;font-weight:bold;text-align:center;vertical-align:top}
    </style>
    <table class="tg"><thead>
    <tr>
        <th class="tg-zv4m"></th>
        <th class="tg-aw21">age</th>
        <th class="tg-aw21">workclass</th>
        <th class="tg-aw21">fnlwgt</th>
        <th class="tg-aw21">education</th>
        <th class="tg-aw21">education-num</th>
        <th class="tg-aw21">marital-status</th>
        <th class="tg-aw21">occupation</th>
        <th class="tg-aw21">relationship</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-aw21">census_id</td>
        <td class="tg-8jgo"></td>
        <td class="tg-8jgo"></td>
        <td class="tg-8jgo"></td>
        <td class="tg-8jgo"></td>
        <td class="tg-8jgo"></td>
        <td class="tg-8jgo"></td>
        <td class="tg-8jgo"></td>
        <td class="tg-8jgo"></td>
    </tr>
    <tr>
        <td class="tg-zv4m">82943611</td>
        <td class="tg-8jgo">39</td>
        <td class="tg-8jgo">State-gov</td>
        <td class="tg-8jgo">77516</td>
        <td class="tg-8jgo">Bachelors</td>
        <td class="tg-8jgo">13</td>
        <td class="tg-8jgo">Never-married</td>
        <td class="tg-8jgo">Adm-clerical</td>
        <td class="tg-8jgo">Not-in-family</td>
    </tr>
    <tr>
        <td class="tg-zv4m">42643227</td>
        <td class="tg-8jgo">50</td>
        <td class="tg-8jgo">Self-emp-not-inc</td>
        <td class="tg-8jgo">83311</td>
        <td class="tg-8jgo">Bachelors</td>
        <td class="tg-8jgo">13</td>
        <td class="tg-8jgo">Married-civ-spouse</td>
        <td class="tg-8jgo">Exec-managerial</td>
        <td class="tg-8jgo">Husband</td>
    </tr>
    <tr>
        <td class="tg-zv4m">93837254</td>
        <td class="tg-8jgo">38</td>
        <td class="tg-8jgo">Private</td>
        <td class="tg-8jgo">215646</td>
        <td class="tg-8jgo">HS-grad</td>
        <td class="tg-8jgo">9</td>
        <td class="tg-8jgo">Divorced</td>
        <td class="tg-8jgo">Handlers-cleaners</td>
        <td class="tg-8jgo">Not-in-family</td>
    </tr>
    <tr>
        <td class="tg-zv4m">87104229</td>
        <td class="tg-8jgo">53</td>
        <td class="tg-8jgo">Private</td>
        <td class="tg-8jgo">234721</td>
        <td class="tg-8jgo">11th</td>
        <td class="tg-8jgo">7</td>
        <td class="tg-8jgo">Married-civ-spouse</td>
        <td class="tg-8jgo">Handlers-cleaners</td>
        <td class="tg-8jgo">Husband</td>
    </tr>
    <tr>
        <td class="tg-zv4m">90069867</td>
        <td class="tg-8jgo">28</td>
        <td class="tg-8jgo">Private</td>
        <td class="tg-8jgo">338409</td>
        <td class="tg-8jgo">Bachelors</td>
        <td class="tg-8jgo">13</td>
        <td class="tg-8jgo">Married-civ-spouse</td>
        <td class="tg-8jgo">Prof-specialty</td>
        <td class="tg-8jgo">Wife</td>
    </tr>
    </tbody></table>

\


Trailing Period Removal
-----------------------

.. function:: strip_trailing_period(df, column_name)

    Strip the trailing period from floats in a specified column of a DataFrame, if present.

    :param df: The DataFrame containing the column to be processed.
    :type df: pd.DataFrame
    :param column_name: The name of the column containing floats with potential trailing periods.
    :type column_name: str

    :returns: The updated DataFrame with the trailing periods removed from the specified column.
    :rtype: pd.DataFrame

    The ``strip_trailing_period`` function is designed to remove trailing periods 
    from float values in a specified column of a DataFrame. This can be particularly 
    useful when dealing with data that has been inconsistently formatted, ensuring 
    that all float values are correctly represented.

**Example Usage**

In the example below, we demonstrate how to use the ``strip_trailing_period`` function to clean a column in a DataFrame. We start by importing the necessary libraries and creating a sample DataFrame. We then use the ``strip_trailing_period`` function to remove any trailing periods from the specified column.

.. code-block:: python

    import pandas as pd
    from eda_toolkit import strip_trailing_period

    # Create a sample dataframe with trailing periods in some values
    data = {
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.],
    }
    df = pd.DataFrame(data)

    # Remove trailing periods from the 'values' column
    df = strip_trailing_period(df=df, column_name="values")


**Output**

`First 6 Rows of Data Before and After Removing Trailing Periods (Adapted from Example)`

.. raw:: html

    <table>
        <tr>
            <td style="padding-right: 10px;">

                <strong>Before:</strong>

                <table border="1" style="width: 150px; text-align: center;">
                    <tr>
                        <th>Index</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>0</td>
                        <td>1.0</td>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td>2.0</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>3.0</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>4.0</td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>5.0</td>
                    </tr>
                    <tr style="background-color: #FFCCCC;">
                        <td>5</td>
                        <td>6.</td>
                    </tr>
                </table>

            </td>
            <td style="padding-left: 10px;">

                <strong>After:</strong>

                <table border="1" style="width: 150px; text-align: center;">
                    <tr>
                        <th>Index</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>0</td>
                        <td>1.0</td>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td>2.0</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>3.0</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>4.0</td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>5.0</td>
                    </tr>
                    <tr style="background-color: #FFCCCC;">
                        <td>5</td>
                        <td>6.0</td>
                    </tr>
                </table>

            </td>
        </tr>
    </table>

\

`Note:` The last row shows 6 as an `int` with a trailing period with its conversion to `float`.


\

Standardized Dates
==================

.. function:: parse_date_with_rule(date_str)

    Parse and standardize date strings based on the provided rule.

    This function takes a date string and standardizes it to the ISO 8601 format
    (YYYY-MM-DD). It assumes dates are provided in either day/month/year or
    month/day/year format. The function first checks if the first part of the
    date string (day or month) is greater than 12, which unambiguously indicates
    a day/month/year format. If the first part is 12 or less, the function
    attempts to parse the date as month/day/year, falling back to day/month/year
    if the former raises a ValueError due to an impossible date (e.g., month
    being greater than 12).

    :param date_str: A date string to be standardized.
    :type date_str: str

    :returns: A standardized date string in the format YYYY-MM-DD.
    :rtype: str

    :raises ValueError: If date_str is in an unrecognized format or if the function
                        cannot parse the date.

**Example Usage**

In the example below, we demonstrate how to use the ``parse_date_with_rule`` 
function to standardize date strings. We start by importing the necessary library 
and creating a sample list of date strings. We then use the ``parse_date_with_rule`` 
function to parse and standardize each date string to the ISO 8601 format.

.. code-block:: python

    from eda_toolkit import parse_date_with_rule

    # Sample date strings
    date_strings = ["15/04/2021", "04/15/2021", "01/12/2020", "12/01/2020"]

    # Standardize the date strings
    standardized_dates = [parse_date_with_rule(date) for date in date_strings]

    print(standardized_dates)

**Output**

.. code-block:: python

    ['2021-04-15', '2021-04-15', '2020-12-01', '2020-01-12']


Binning Numerical Columns
==========================

If your DataFrame (e.g., the census data [1]_) 
does not have age or any other numerical column of interest binned, you can 
apply the following binning logic to categorize the data. Below, we use the age 
column from the UCI Machine Learning Repository as an example:

.. code-block:: python

    # Create age bins so that the ages can be categorized
    bin_ages = [
        0,
        18,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        float("inf"),
    ]

    # Create labels for the bins
    label_ages = [
        "< 18",
        "18-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80-89",
        "90-99",
        "100 +",
    ]

    # Categorize the ages and assign to a new variable
    df["age_group"] = pd.cut(
        df["age"],
        bins=bin_ages,
        labels=label_ages,
        right=False,
    )

`Note:` This code snippet creates age bins and assigns a corresponding age group 
label to each age in the DataFrame. The ``pd.cut`` function from pandas is used to 
categorize the ages and assign them to a new column, ``age_group``. Adjust the bins 
and labels as needed for your specific data.



.. [1] Kohavi, Ron. (1996). Census Income. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S.