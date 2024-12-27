.. _data_management:   

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

Data Management Overview
===========================

In any data-driven project, effective management of data is crucial. This 
section provides essential techniques for handling and preparing data to ensure 
consistency, accuracy, and ease of analysis. From directory setup and data 
cleaning to advanced data processing, these methods form the backbone of reliable 
data management. Dive into the following topics to enhance your data handling 
capabilities and streamline your workflow.

Data Management Techniques
===============================

Path directories
----------------

**Ensure that the directory exists. If not, create it.**

.. function:: ensure_directory(path)

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


**Implementation Example**

In the example below, we demonstrate how to use the ``ensure_directory`` function 
to verify and create directories as needed. This example sets up paths for data and 
image directories, ensuring they exist before performing any operations that depend on them.

First, we define the base path as the parent directory of the current directory. 
The ``os.pardir`` constant, equivalent to ``"..""``, is used to navigate up one 
directory level. Then, we define paths for the data directory and data output 
directory, both located one level up from the current directory. 


Next, we set paths for the PNG and SVG image directories, located within an 
``images`` folder in the parent directory. Using the ``ensure_directory`` 
function, we then verify that these directories exist. If any of the specified 
directories do not exist, the function creates them.

.. code-block:: python

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


.. _Adding_Unique_Identifiers:

Adding Unique Identifiers
--------------------------

**Add a column of unique IDs with a specified number of digits to the DataFrame.**

.. function:: add_ids(df, id_colname="ID", num_digits=9, seed=None, set_as_index=False)

    :param df: The DataFrame to which IDs will be added.
    :type df: pd.DataFrame
    :param id_colname: The name of the new column for the unique IDs. Defaults to ``"ID"``.
    :type id_colname: str, optional
    :param num_digits: The number of digits for the unique IDs. Defaults to ``9``. The first digit will always be non-zero to ensure proper formatting.
    :type num_digits: int, optional
    :param seed: The seed for the random number generator to ensure reproducibility. Defaults to ``None``.
    :type seed: int, optional
    :param set_as_index: If ``True``, the generated ID column will replace the existing index of the DataFrame. Defaults to ``False``.
    :type set_as_index: bool, optional

    :returns: The updated DataFrame with a new column of unique IDs. If ``set_as_index`` is ``True``, the new ID column replaces the existing index.
    :rtype: pd.DataFrame

    :raises ValueError: If the number of rows in the DataFrame exceeds the pool of possible unique IDs for the specified ``num_digits``.

.. admonition:: Notes

    - The function ensures all IDs are unique by resolving potential collisions during generation, even for large datasets.
    - The total pool size of unique IDs is determined by :math:`9 \times 10^{(\text{num_digits} - 1)}`, since the first digit must be non-zero.
    - Warnings are printed if the number of rows in the DataFrame approaches the pool size of possible unique IDs, recommending increasing ``num_digits``.
    - If ``set_as_index`` is ``False``, the ID column will be added as the first column in the DataFrame.
    - Setting a random seed ensures reproducibility of the generated IDs.

The ``add_ids`` function is used to append a column of unique identifiers with a 
specified number of digits to a given dataframe. This is particularly useful for 
creating unique patient or record IDs in datasets. The function allows you to 
specify a custom column name for the IDs, the number of digits for each ID, and 
optionally set a seed for the random number generator to ensure reproducibility. 
Additionally, you can choose whether to set the new ID column as the index of the 
dataframe.

**Implementation Example**

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

.. code-block:: bash

    DataFrame index is unique.

.. raw:: html

    <style type="text/css">
    .tg-wrap {
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif;font-size:11px;
      overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:monospace, sans-serif;font-size:11px;
      font-weight:normal;overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-zv4m{border-color:#ffffff;text-align:left;vertical-align:top}
    .tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
    .tg .tg-aw21{border-color:#ffffff;font-weight:bold;text-align:center;vertical-align:top}
    </style>
    <div class="tg-wrap">
    <table class="tg">
      <thead>
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
        </tr>
      </thead>
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
          <td class="tg-zv4m">582248222</td>
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
          <td class="tg-zv4m">561810758</td>
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
          <td class="tg-zv4m">598098459</td>
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
          <td class="tg-zv4m">776705221</td>
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
          <td class="tg-zv4m">479262902</td>
          <td class="tg-8jgo">28</td>
          <td class="tg-8jgo">Private</td>
          <td class="tg-8jgo">338409</td>
          <td class="tg-8jgo">Bachelors</td>
          <td class="tg-8jgo">13</td>
          <td class="tg-8jgo">Married-civ-spouse</td>
          <td class="tg-8jgo">Prof-specialty</td>
          <td class="tg-8jgo">Wife</td>
        </tr>
      </tbody>
    </table>
    </div>


\


Trailing Period Removal
-----------------------

**Strip the trailing period from values in a specified column of a DataFrame, if present.**

.. function:: strip_trailing_period(df, column_name)

    :param df: The DataFrame containing the column to be processed.
    :type df: pd.DataFrame
    :param column_name: The name of the column containing values with potential trailing periods.
    :type column_name: str

    :returns: The updated DataFrame with trailing periods removed from the specified column.
    :rtype: pd.DataFrame

    :raises ValueError: If the specified ``column_name`` does not exist in the DataFrame, pandas will raise a ``ValueError``.

.. admonition:: Notes:

    - For string values, trailing periods are stripped directly.
    - For numeric values represented as strings (e.g., ``"1234."``), the trailing period is removed, and the value is converted back to a numeric type if possible.
    - ``NaN`` values are preserved and remain unprocessed.
    - Non-string and non-numeric types are returned as-is.


The ``strip_trailing_period`` function is designed to remove trailing periods 
from float values in a specified column of a DataFrame. This can be particularly 
useful when dealing with data that has been inconsistently formatted, ensuring 
that all float values are correctly represented.

**Implementation Example**

In the example below, we demonstrate how to use the ``strip_trailing_period`` function to clean a 
column in a DataFrame. We start by importing the necessary libraries and creating a sample DataFrame. 
We then use the ``strip_trailing_period`` function to remove any trailing periods from the specified column.

.. code-block:: python

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
            <td style="padding-right: 10px; font-family: Arial; font-size: 14px;">

                <strong>Before:</strong>

                <table border="1" style="width: 150px; text-align: center; font-family: Arial; font-size: 14px;">
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
            <td style="padding-left: 10px; font-family: Arial; font-size: 14px;">

                <strong>After:</strong>

                <table border="1" style="width: 150px; text-align: center; font-family: Arial; font-size: 14px;">
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

.. note:: 
    
    The last row shows 6 as an ``int`` with a trailing period with its conversion to ``float``.


\

Standardized Dates
-------------------

**Parse and standardize date strings based on the provided rule.**

.. function:: parse_date_with_rule(date_str)

    This function takes a date string and standardizes it to the ``ISO 8601`` format
    (``YYYY-MM-DD``). It assumes dates are provided in either **day/month/year** or
    **month/day/year** format. The function first checks if the first part of the
    date string (day or month) is greater than 12, which unambiguously indicates
    a **day/month/year** format. If the first part is 12 or less, the function
    attempts to parse the date as **month/day/year**, falling back to **day/month/year**
    if the former raises a ``ValueError`` due to an impossible date (e.g., month
    being greater than 12).

    :param date_str: A date string to be standardized.
    :type date_str: str

    :returns: A standardized date string in the format ``YYYY-MM-DD``.
    :rtype: str

    :raises ValueError: If ``date_str`` is in an unrecognized format or if the function
                        cannot parse the date.


**Implementation Example**

In the example below, we demonstrate how to use the ``parse_date_with_rule`` 
function to standardize date strings. We start by importing the necessary library 
and creating a sample list of date strings. We then use the ``parse_date_with_rule`` 
function to parse and standardize each date string to the ``ISO 8601`` format.

.. code-block:: python

    from eda_toolkit import parse_date_with_rule

    date_strings = ["15/04/2021", "04/15/2021", "01/12/2020", "12/01/2020"]

    standardized_dates = [parse_date_with_rule(date) for date in date_strings]

    print(standardized_dates)

**Output**

.. code-block:: python

    ['2021-04-15', '2021-04-15', '2020-12-01', '2020-01-12']



.. important:: 
    
    In the next example, we demonstrate how to apply the ``parse_date_with_rule`` 
    function to a DataFrame column containing date strings using the ``.apply()`` method. 
    This is particularly useful when you need to standardize date formats across an 
    entire column in a DataFrame.

.. code-block:: python

    # Creating the DataFrame
    data = {
        "date_column": [
            "31/12/2021",
            "01/01/2022",
            "12/31/2021",
            "13/02/2022",
            "07/04/2022",
        ],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "amount": [100.0, 150.5, 200.75, 250.25, 300.0],
    }

    df = pd.DataFrame(data)

    # Apply the function to the DataFrame column
    df["standardized_date"] = df["date_column"].apply(parse_date_with_rule)

    print(df)

**Output**

.. code-block:: text

       date_column     name  amount standardized_date
    0   31/12/2021    Alice  100.00        2021-12-31
    1   01/01/2022      Bob  150.50        2022-01-01
    2   12/31/2021  Charlie  200.75        2021-12-31
    3   13/02/2022    David  250.25        2022-02-13
    4   07/04/2022      Eve  300.00        2022-04-07


DataFrame Analysis
-------------------

**Analyze DataFrame columns, including data type, null values, and unique value counts.**

The ``dataframe_columns`` function provides a comprehensive summary of a DataFrame's columns, including information on data types, null values, unique values, and the most frequent values. It can output either a plain DataFrame for further processing or a styled DataFrame for visual presentation in Jupyter environments.

.. function:: dataframe_columns(df, background_color=None, return_df=False, sort_cols_alpha=False)

    :param df: The DataFrame to analyze.
    :type df: pandas.DataFrame
    :param background_color: Hex color code or color name for background styling in the output 
                             DataFrame. Applies to specific columns, such as unique value totals 
                             and percentages. Defaults to ``None``.
    :type background_color: str, optional
    :param return_df: If ``True``, returns the plain DataFrame with summary statistics. If ``False``, 
                      returns a styled DataFrame for visual presentation. Defaults to ``False``.
    :type return_df: bool, optional
    :param sort_cols_alpha: If ``True``, sorts the DataFrame columns alphabetically in the output. 
                            Defaults to ``False``.
    :type sort_cols_alpha: bool, optional

    :returns: 
        - If ``return_df`` is ``True`` or running in a terminal environment, returns the plain 
          DataFrame containing column summary statistics.
        - If ``return_df`` is ``False`` and running in a Jupyter Notebook, returns a styled 
          DataFrame with optional background styling.
    :rtype: pandas.DataFrame
    :raises ValueError: If the DataFrame is empty or contains no columns.

.. admonition:: Notes

    - Automatically detects whether the function is running in a Jupyter Notebook or terminal and adjusts the output accordingly.
    - In Jupyter environments, uses Pandas' Styler for visual presentation. If the installed Pandas version does not support ``hide``, falls back to ``hide_index``.
    - Utilizes a ``tqdm`` progress bar to show the status of column processing.
    - Preprocesses columns to handle NaN values and replaces empty strings with Pandas' ``pd.NA`` for consistency.



Census Income Example
""""""""""""""""""""""""""""""

In the example below, we demonstrate how to use the ``dataframe_columns`` 
function to analyze a DataFrame's columns. You may notice a new variable, 
``age_group``, is introduced. The logic for generating this variable is :ref:`provided here <Binning_Numerical_Columns>`.

.. code-block:: python

    from eda_toolkit import dataframe_columns

    dataframe_columns(df=df)


**Output**

`Result on Census Income Data (Adapted from Kohavi, 1996, UCI Machine Learning Repository)` [1]_

.. code-block:: python

    Shape:  (48842, 16) 

    Total seconds of processing time: 0.861555

.. raw:: html

    <style type="text/css">
    .tg-wrap {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    }
    .tg  {border:none;border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-style:solid;border-width:0px;font-family:Consolas, monospace;font-size:11px;overflow:hidden;padding:0px 6px;
    word-break:normal;}
    .tg th{border-style:solid;border-width:0px;font-family:Consolas, monospace;font-size:11px;font-weight:normal;
    overflow:hidden;padding:0px 6px;word-break:normal;}
    .tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
    .tg .tg-dvpl{border-color:inherit;text-align:right;vertical-align:top}
    .tg .tg-rvpl{border-color:inherit;text-align:right;vertical-align:top}
    </style>
    <div class="tg-wrap">
    <table class="tg">
        <thead>
        <tr>
            <th class="tg-rvpl"></th>
            <th class="tg-rvpl"><span style="font-weight:bold">column</span></th>
            <th class="tg-rvpl"><span style="font-weight:bold">dtype</span></th>
            <th class="tg-0pky"><span style="font-weight:bold">null_total</span></th>
            <th class="tg-0pky"><span style="font-weight:bold">null_pct</span></th>
            <th class="tg-0pky"><span style="font-weight:bold">unique_values_total</span></th>
            <th class="tg-0pky"><span style="font-weight:bold">max_unique_value</span></th>
            <th class="tg-0pky"><span style="font-weight:bold">max_unique_value_total</span></th>
            <th class="tg-0pky"><span style="font-weight:bold">max_unique_value_pct</span></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td class="tg-rvpl">0</td>
            <td class="tg-dvpl">age</td>
            <td class="tg-dvpl">int64</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">74</td>
            <td class="tg-dvpl">36</td>
            <td class="tg-dvpl">1348</td>
            <td class="tg-dvpl">2.76</td>
        </tr>
        <tr>
            <td class="tg-rvpl">1</td>
            <td class="tg-dvpl">workclass</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">963</td>
            <td class="tg-dvpl">1.97</td>
            <td class="tg-dvpl">9</td>
            <td class="tg-dvpl">Private</td>
            <td class="tg-dvpl">33906</td>
            <td class="tg-dvpl">69.42</td>
        </tr>
        <tr>
            <td class="tg-rvpl">2</td>
            <td class="tg-dvpl">fnlwgt</td>
            <td class="tg-dvpl">int64</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">28523</td>
            <td class="tg-dvpl">203488</td>
            <td class="tg-dvpl">21</td>
            <td class="tg-dvpl">0.04</td>
        </tr>
        <tr>
            <td class="tg-rvpl">3</td>
            <td class="tg-dvpl">education</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">16</td>
            <td class="tg-dvpl">HS-grad</td>
            <td class="tg-dvpl">15784</td>
            <td class="tg-dvpl">32.32</td>
        </tr>
        <tr>
            <td class="tg-rvpl">4</td>
            <td class="tg-dvpl">education-num</td>
            <td class="tg-dvpl">int64</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">16</td>
            <td class="tg-dvpl">9</td>
            <td class="tg-dvpl">15784</td>
            <td class="tg-dvpl">32.32</td>
        </tr>
        <tr>
            <td class="tg-rvpl">5</td>
            <td class="tg-dvpl">marital-status</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">7</td>
            <td class="tg-dvpl">Married-civ-spouse</td>
            <td class="tg-dvpl">22379</td>
            <td class="tg-dvpl">45.82</td>
        </tr>
        <tr>
            <td class="tg-rvpl">6</td>
            <td class="tg-dvpl">occupation</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">966</td>
            <td class="tg-dvpl">1.98</td>
            <td class="tg-dvpl">15</td>
            <td class="tg-dvpl">Prof-specialty</td>
            <td class="tg-dvpl">6172</td>
            <td class="tg-dvpl">12.64</td>
        </tr>
        <tr>
            <td class="tg-rvpl">7</td>
            <td class="tg-dvpl">relationship</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">6</td>
            <td class="tg-dvpl">Husband</td>
            <td class="tg-dvpl">19716</td>
            <td class="tg-dvpl">40.37</td>
        </tr>
        <tr>
            <td class="tg-rvpl">8</td>
            <td class="tg-dvpl">race</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">5</td>
            <td class="tg-dvpl">White</td>
            <td class="tg-dvpl">41762</td>
            <td class="tg-dvpl">85.5</td>
        </tr>
        <tr>
            <td class="tg-rvpl">9</td>
            <td class="tg-dvpl">sex</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">2</td>
            <td class="tg-dvpl">Male</td>
            <td class="tg-dvpl">32650</td>
            <td class="tg-dvpl">66.85</td>
        </tr>
        <tr>
            <td class="tg-rvpl">10</td>
            <td class="tg-dvpl">capital-gain</td>
            <td class="tg-dvpl">int64</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">123</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">44807</td>
            <td class="tg-dvpl">91.74</td>
        </tr>
        <tr>
            <td class="tg-rvpl">11</td>
            <td class="tg-dvpl">capital-loss</td>
            <td class="tg-dvpl">int64</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">99</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">46560</td>
            <td class="tg-dvpl">95.33</td>
        </tr>
        <tr>
            <td class="tg-rvpl">12</td>
            <td class="tg-dvpl">hours-per-week</td>
            <td class="tg-dvpl">int64</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">96</td>
            <td class="tg-dvpl">40</td>
            <td class="tg-dvpl">22803</td>
            <td class="tg-dvpl">46.69</td>
        </tr>
        <tr>
            <td class="tg-rvpl">13</td>
            <td class="tg-dvpl">native-country</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">274</td>
            <td class="tg-dvpl">0.56</td>
            <td class="tg-dvpl">42</td>
            <td class="tg-dvpl">United-States</td>
            <td class="tg-dvpl">43832</td>
            <td class="tg-dvpl">89.74</td>
        </tr>
        <tr>
            <td class="tg-rvpl">14</td>
            <td class="tg-dvpl">income</td>
            <td class="tg-dvpl">object</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">4</td>
            <td class="tg-dvpl">&lt;=50K</td>
            <td class="tg-dvpl">24720</td>
            <td class="tg-dvpl">50.61</td>
        </tr>
        <tr>
            <td class="tg-rvpl">15</td>
            <td class="tg-dvpl">age_group</td>
            <td class="tg-dvpl">category</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">0</td>
            <td class="tg-dvpl">9</td>
            <td class="tg-dvpl">18-29</td>
            <td class="tg-dvpl">13920</td>
            <td class="tg-dvpl">28.5</td>
        </tr>
        </tbody>
    </table>
    </div>



\

DataFrame Column Names
""""""""""""""""""""""""""""""

``unique_values_total``
    This column indicates the total number of unique values present in each column of the DataFrame. It measures the distinct values that a column holds. For example, in the ``age`` column, there are 74 unique values, meaning the ages vary across 74 distinct entries.

``max_unique_value``
    This column shows the most frequently occurring value in each column. For example, in the ``workclass`` column, the most common value is ``Private``, indicating that this employment type is the most represented in the dataset. For numeric columns like ``capital-gain`` and ``capital-loss``, the most common value is ``0``, which suggests that the majority of individuals have no capital gain or loss.

``max_unique_value_total``
    This represents the count of the most frequently occurring value in each column. For instance, in the ``native-country`` column, the value ``United-States`` appears ``43,832`` times, indicating that the majority of individuals in the dataset are from the United States.

``max_unique_value_pct``
    This column shows the percentage that the most frequent value constitutes of the total number of rows. For example, in the ``race`` column, the value ``White`` makes up ``85.5%`` of the data, suggesting a significant majority of the dataset belongs to this racial group.

Calculation Details
""""""""""""""""""""""""""""""
- ``unique_values_total`` is calculated using the ``nunique()`` function, which counts the number of unique values in a column.
- ``max_unique_value`` is determined by finding the value with the highest frequency using ``value_counts()``. For string columns, any missing values (if present) are replaced with the string ``"null"`` before computing the frequency.
- ``max_unique_value_total`` is the frequency count of the ``max_unique_value``.
- ``max_unique_value_pct`` is the percentage of ``max_unique_value_total`` divided by the total number of rows in the DataFrame, providing an idea of how dominant the most frequent value is.

This analysis helps in identifying columns with a high proportion of dominant values, like ``<=50K`` in the ``income`` column, which appears ``24,720`` times, making up ``50.61%`` of the entries. This insight can be useful for understanding data distributions, identifying potential data imbalances, or even spotting opportunities for feature engineering in further data processing steps.

Generating Summary Tables for Variable Combinations
-----------------------------------------------------

**This function generates summary tables for all possible combinations of specified variables 
in a DataFrame and save them to an Excel file.**


.. function:: summarize_all_combinations(df, variables, data_path, data_name, min_length=2)

    :param df: The pandas DataFrame containing the data.
    :type df: pandas.DataFrame
    :param variables: List of column names from the DataFrame to generate combinations.
    :type variables: list of str
    :param data_path: Path where the output Excel file will be saved.
    :type data_path: str
    :param data_name: Name of the output Excel file.
    :type data_name: str
    :param min_length: Minimum size of the combinations to generate. Defaults to ``2``.
    :type min_length: int, optional

    :returns: A tuple containing a dictionary of summary tables and a list of all generated combinations.
    :rtype: tuple(dict, list)

.. note::

    - The function will create an Excel file with a sheet for each combination of the specified variables, as well as a "Table of Contents" sheet with hyperlinks to each summary table.
    - The sheet names are limited to 31 characters due to Excel's constraints.

The function returns two outputs:

    1. ``summary_tables``: A dictionary where each key is a tuple representing a combination 
    of variables, and each value is a DataFrame containing the summary table for that combination. 
    Each summary table includes the count and proportion of occurrences for each unique combination of values.

    2. ``all_combinations``: A list of all generated combinations of the specified variables. 
    This is useful for understanding which combinations were analyzed and included in the summary tables.

**Implementation Example**

Below, we use the ``summarize_all_combinations`` function to generate summary tables for the specified 
variables from a DataFrame containing the census data [1]_.

.. note:: 

    Before proceeding with any further examples in this documentation, ensure that the ``age`` variable is binned into a new variable, ``age_group``.  
    Detailed instructions for this process can be found under :ref:`Binning Numerical Columns <Binning_Numerical_Columns>`.


.. code-block:: python

    from eda_toolkit import summarize_all_combinations

    # Define unique variables for the analysis
    unique_vars = [
        "age_group",
        "workclass",
        "education",
        "occupation",
        "race",
        "sex",
        "income",
    ]

    # Generate summary tables for all combinations of the specified variables
    summary_tables, all_combinations = summarize_all_combinations(
        df=df,
        data_path=data_output,
        variables=unique_vars,
        data_name="census_summary_tables.xlsx",
    )

    # Print all combinations of variables
    print(all_combinations)

**Output**

.. code-blocK:: text 

    [('age_group', 'workclass'),
    ('age_group', 'education'),
    ('age_group', 'occupation'),
    ('age_group', 'race'),
    ('age_group', 'sex'),
    ('age_group', 'income'),
    ('workclass', 'education'),
    ('workclass', 'occupation'),
    ('workclass', 'race'),
    ('workclass', 'sex'),
    ('workclass', 'income'),
    ('education', 'occupation'),
    ('education', 'race'),
    ('education', 'sex'),
    ('education', 'income'),
    ('occupation', 'race'),
    ('occupation', 'sex'),
    ('occupation', 'income'),
    ('race', 'sex'),
    ('race', 'income'),
    ('sex', 'income'),
    ('age_group', 'workclass', 'education'),
    ('age_group', 'workclass', 'occupation'),
    ('age_group', 'workclass', 'race'),
    ('age_group', 'workclass', 'sex'),
    ('age_group', 'workclass', 'income'),
    ('age_group', 'education', 'occupation'),
    ('age_group', 'education', 'race'),
    ('age_group', 'education', 'sex'),
    ('age_group', 'education', 'income'),
    ('age_group', 'occupation', 'race'),
    ('age_group', 'occupation', 'sex'),
    ('age_group', 'occupation', 'income'),
    ('age_group', 'race', 'sex'),
    ('age_group', 'race', 'income'),
    ('age_group', 'sex', 'income'),
    ('workclass', 'education', 'occupation'),
    ('workclass', 'education', 'race'),
    ('workclass', 'education', 'sex'),
    ('workclass', 'education', 'income'),
    ('workclass', 'occupation', 'race'),
    ('workclass', 'occupation', 'sex'),
    ('workclass', 'occupation', 'income'),
    ('workclass', 'race', 'sex'),
    ('workclass', 'race', 'income'),
    ('workclass', 'sex', 'income'),
    ('education', 'occupation', 'race'),
    ('education', 'occupation', 'sex'),
    ('education', 'occupation', 'income'),
    ('education', 'race', 'sex'),
    ('education', 'race', 'income'),
    ('education', 'sex', 'income'),
    ('occupation', 'race', 'sex'),
    ('occupation', 'race', 'income'),
    ('occupation', 'sex', 'income'),
    ('race', 'sex', 'income'),
    ('age_group', 'workclass', 'education', 'occupation'),
    ('age_group', 'workclass', 'education', 'race'),
    ('age_group', 'workclass', 'education', 'sex'),
    ('age_group', 'workclass', 'education', 'income'),
    ('age_group', 'workclass', 'occupation', 'race'),
    ('age_group', 'workclass', 'occupation', 'sex'),
    ('age_group', 'workclass', 'occupation', 'income'),
    ('age_group', 'workclass', 'race', 'sex'),
    ('age_group', 'workclass', 'race', 'income'),
    ('age_group', 'workclass', 'sex', 'income'),
    ('age_group', 'education', 'occupation', 'race'),
    ('age_group', 'education', 'occupation', 'sex'),
    ('age_group', 'education', 'occupation', 'income'),
    ('age_group', 'education', 'race', 'sex'),
    ('age_group', 'education', 'race', 'income'),
    ('age_group', 'education', 'sex', 'income'),
    ('age_group', 'occupation', 'race', 'sex'),
    ('age_group', 'occupation', 'race', 'income'),
    ('age_group', 'occupation', 'sex', 'income'),
    ('age_group', 'race', 'sex', 'income'),
    ('workclass', 'education', 'occupation', 'race'),
    ('workclass', 'education', 'occupation', 'sex'),
    ('workclass', 'education', 'occupation', 'income'),
    ('workclass', 'education', 'race', 'sex'),
    ('workclass', 'education', 'race', 'income'),
    ('workclass', 'education', 'sex', 'income'),
    ('workclass', 'occupation', 'race', 'sex'),
    ('workclass', 'occupation', 'race', 'income'),
    ('workclass', 'occupation', 'sex', 'income'),
    ('workclass', 'race', 'sex', 'income'),
    ('education', 'occupation', 'race', 'sex'),
    ('education', 'occupation', 'race', 'income'),
    ('education', 'occupation', 'sex', 'income'),
    ('education', 'race', 'sex', 'income'),
    ('occupation', 'race', 'sex', 'income'),
    ('age_group', 'workclass', 'education', 'occupation', 'race'),
    ('age_group', 'workclass', 'education', 'occupation', 'sex'),
    ('age_group', 'workclass', 'education', 'occupation', 'income'),
    ('age_group', 'workclass', 'education', 'race', 'sex'),
    ('age_group', 'workclass', 'education', 'race', 'income'),
    ('age_group', 'workclass', 'education', 'sex', 'income'),
    ('age_group', 'workclass', 'occupation', 'race', 'sex'),
    ('age_group', 'workclass', 'occupation', 'race', 'income'),
    ('age_group', 'workclass', 'occupation', 'sex', 'income'),
    ('age_group', 'workclass', 'race', 'sex', 'income'),
    ('age_group', 'education', 'occupation', 'race', 'sex'),
    ('age_group', 'education', 'occupation', 'race', 'income'),
    ('age_group', 'education', 'occupation', 'sex', 'income'),
    ('age_group', 'education', 'race', 'sex', 'income'),
    ('age_group', 'occupation', 'race', 'sex', 'income'),
    ('workclass', 'education', 'occupation', 'race', 'sex'),
    ('workclass', 'education', 'occupation', 'race', 'income'),
    ('workclass', 'education', 'occupation', 'sex', 'income'),
    ('workclass', 'education', 'race', 'sex', 'income'),
    ('workclass', 'occupation', 'race', 'sex', 'income'),
    ('education', 'occupation', 'race', 'sex', 'income'),
    ('age_group', 'workclass', 'education', 'occupation', 'race', 'sex'),
    ('age_group', 'workclass', 'education', 'occupation', 'race', 'income'),
    ('age_group', 'workclass', 'education', 'occupation', 'sex', 'income'),
    ('age_group', 'workclass', 'education', 'race', 'sex', 'income'),
    ('age_group', 'workclass', 'occupation', 'race', 'sex', 'income'),
    ('age_group', 'education', 'occupation', 'race', 'sex', 'income'),
    ('workclass', 'education', 'occupation', 'race', 'sex', 'income'),
    ('age_group',
    'workclass',
    'education',
    'occupation',
    'race',
    'sex',
    'income')]


When applied to the US Census data, the output Excel file will contain summary tables for all possible combinations of the specified variables. 
The first sheet will be a Table of Contents with hyperlinks to each summary table.

.. raw:: html

   <div class="no-click">

.. image:: ../assets/summarize_combos.gif
   :alt: EDA Toolkit Logo
   :align: left
   :width: 900px

.. raw:: html

   </div>

.. raw:: html
   
   <div style="height: 106px;"></div>



Saving DataFrames to Excel with Customized Formatting
-----------------------------------------------------
**Save multiple DataFrames to separate sheets in an Excel file with progress tracking and formatting.**

This section explains how to save multiple DataFrames to separate sheets in an 
Excel file with customized formatting using the ``save_dataframes_to_excel`` function.

.. function:: save_dataframes_to_excel(file_path, df_dict, decimal_places=0)

    Save DataFrames to an Excel file, applying formatting and autofitting columns.

    :param file_path: Full path to the output Excel file.
    :type file_path: str

    :param df_dict: Dictionary where keys are sheet names and values are DataFrames to save.
    :type df_dict: dict

    :param decimal_places: Number of decimal places to round numeric columns. Default is ``0``. When set to ``0``, numeric columns are saved as integers.
    :type decimal_places: int

.. admonition:: Notes

    - **Progress Tracking**: The function uses ``tqdm`` to display a progress bar while saving DataFrames to Excel sheets.
    - **Column Formatting**:
        - Columns are autofitted to content length and left-aligned by default.
        - Numeric columns are rounded to the specified decimal places and formatted accordingly.
        - Non-numeric columns are left-aligned.
    - **Header Styling**:
        - Headers are bold, left-aligned, and borderless.
    - **Dependencies**: This function requires the ``xlsxwriter`` library.

**The function performs the following tasks**:

- Writes each DataFrame to its respective sheet in the Excel file.
- Rounds numeric columns to the specified number of decimal places.
- Applies customized formatting to headers and cells.
- Autofits columns based on content length for improved readability.
- Tracks the saving process with a progress bar, making it user-friendly for large datasets.


**Implementation Example**

Below, we use the ``save_dataframes_to_excel`` function to save two DataFrames: 
the original DataFrame and a filtered DataFrame with ages between `18` and `40`.

.. code-block:: python

    from eda_toolkit import save_dataframes_to_excel

    file_name = "df_census.xlsx"  # Name of the output Excel file
    file_path = os.path.join(data_path, file_name) 

    # filter DataFrame to Ages 18-40
    filtered_df = df[(df["age"] > 18) & (df["age"] < 40)]

    df_dict = {
        "original_df": df,
        "ages_18_to_40": filtered_df,
    }

    save_dataframes_to_excel(
        file_path=file_path,
        df_dict=df_dict,
        decimal_places=0,
    )


**Output**

The output Excel file will contain the original DataFrame and a filtered DataFrame as a separate tab with ages 
between `18` and `40`, each on separate sheets with customized formatting.


Creating Contingency Tables
----------------------------

**Create a contingency table from one or more columns in a DataFrame, with sorting options.**

This section explains how to create contingency tables from one or more columns in a DataFrame, with options to sort the results using the ``contingency_table`` function.

.. function:: contingency_table(df, cols=None, sort_by=0)

    :param df: The DataFrame to analyze.
    :type df: pandas.DataFrame
    :param cols: Name of the column (as a string) for a single column or list of column names for multiple columns. Must provide at least one column.
    :type cols: str or list of str, optional
    :param sort_by: Enter ``0`` to sort results by column groups; enter ``1`` to sort results by totals in descending order. Defaults to ``0``.
    :type sort_by: int, optional
    :raises ValueError: If no columns are specified or if ``sort_by`` is not ``0`` or ``1``.
    :returns: A DataFrame containing the contingency table with the specified columns, a ``'Total'`` column representing the count of occurrences, and a ``'Percentage'`` column representing the percentage of the total count.
    :rtype: pandas.DataFrame

**Implementation Example**

Below, we use the ``contingency_table`` function to create a contingency table 
from the specified columns in a DataFrame containing census data [1]_

.. code-block:: python

    from eda_toolkit import contingency_table

    contingency_table(
        df=df,
        cols=[
            "age_group",
            "workclass",
            "race",
            "sex",
        ],
        sort_by=1,
    )

.. note:: 

    In the example below, we demonstrate how to use the ``dataframe_columns`` 
    function to analyze a DataFrame's columns. You may notice a new variable, 
    ``age_group``, is introduced. The logic for generating this variable is :ref:`provided here <Binning_Numerical_Columns>`.


**Output**

The output will be a contingency table with the specified columns, showing the 
total counts and percentages of occurrences for each combination of values. The 
table will be sorted by the ``'Total'`` column in descending order because ``sort_by`` 
is set to ``1``.


.. code-block:: text

    
        age_group     workclass                race     sex  Total  Percentage
    0       30-39       Private               White    Male   5856       11.99
    1       18-29       Private               White    Male   5623       11.51
    2       40-49       Private               White    Male   4267        8.74
    3       18-29       Private               White  Female   3680        7.53
    4       50-59       Private               White    Male   2565        5.25
    ..        ...           ...                 ...     ...    ...         ...
    467     50-59   Federal-gov               Other    Male      1        0.00
    468     50-59     Local-gov  Asian-Pac-Islander  Female      1        0.00
    469     70-79  Self-emp-inc               Black    Male      1        0.00
    470     80-89     Local-gov  Asian-Pac-Islander    Male      1        0.00
    471                                                      48842      100.00

    [472 rows x 6 columns]


\

Highlighting Specific Columns in a DataFrame
---------------------------------------------

This section explains how to highlight specific columns in a DataFrame using the ``highlight_columns`` function.

**Highlight specific columns in a DataFrame with a specified background color.**

.. function:: highlight_columns(df, columns, color="yellow")

    :param df: The DataFrame to be styled.
    :type df: pandas.DataFrame
    :param columns: List of column names to be highlighted.
    :type columns: list of str
    :param color: The background color to be applied for highlighting (default is ``"yellow"``).
    :type color: str, optional

    :returns: A Styler object with the specified columns highlighted.
    :rtype: pandas.io.formats.style.Styler

**Implementation Example**

Below, we use the ``highlight_columns`` function to highlight the ``age`` and ``education`` 
columns in the first 5 rows of the census [1]_ DataFrame with a pink background color.

.. code-block:: python

    from eda_toolkit import highlight_columns

    highlighted_df = highlight_columns(
        df=df.head(),
        columns=["age", "education"],
        color="#F8C5C8",
    )

    highlighted_df

**Output**

The output will be a DataFrame with the specified columns highlighted in the given background color. 
The ``age`` and ``education`` columns will be highlighted in pink.

The resulting styled DataFrame can be displayed in a Jupyter Notebook or saved to an 
HTML file using the ``.render()`` method of the Styler object.


.. raw:: html

    <style type="text/css">
    .tg  {border:none;border-collapse:collapse;border-spacing:0;margin:0px auto;}
    .tg td{border-style:solid;border-width:0px;font-family:monospace, sans-serif;font-size:11px;overflow:hidden;padding:0px 5px;
    word-break:normal;}
    .tg th{border-style:solid;border-width:0px;font-family:monospace, sans-serif;font-size:11px;font-weight:normal;
    overflow:hidden;padding:0px 5px;word-break:normal;}
    .tg .tg-zv36{background-color:#ffffff;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
    .tg .tg-c6of{background-color:#ffffff;border-color:inherit;text-align:left;vertical-align:top}
    .tg .tg-7g6k{background-color:#ffffff;border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
    .tg .tg-3xi5{background-color:#ffffff;border-color:inherit;text-align:center;vertical-align:top}
    .tg .tg-6qlg{background-color:#FFCCCC;border-color:inherit;text-align:center;vertical-align:top}
    @media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;margin: auto 0px;}}</style>
    <div class="tg-wrap"><table class="tg"><thead>
    <tr>
        <th class="tg-c6of"></th>
        <th class="tg-7g6k">age</th>
        <th class="tg-7g6k">workclass</th>
        <th class="tg-7g6k">fnlwgt</th>
        <th class="tg-7g6k">education</th>
        <th class="tg-7g6k">education-num</th>
        <th class="tg-7g6k">marital-status</th>
        <th class="tg-7g6k">occupation</th>
        <th class="tg-7g6k">relationship</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-zv36">census_id</td>
        <td class="tg-3xi5"></td>
        <td class="tg-3xi5"></td>
        <td class="tg-3xi5"></td>
        <td class="tg-3xi5"></td>
        <td class="tg-3xi5"></td>
        <td class="tg-3xi5"></td>
        <td class="tg-3xi5"></td>
        <td class="tg-3xi5"></td>
    </tr>
    <tr>
        <td class="tg-c6of">582248222</td>
        <td class="tg-6qlg">39</td>
        <td class="tg-3xi5">State-gov</td>
        <td class="tg-3xi5">77516</td>
        <td class="tg-6qlg">Bachelors</td>
        <td class="tg-3xi5">13</td>
        <td class="tg-3xi5">Never-married</td>
        <td class="tg-3xi5">Adm-clerical</td>
        <td class="tg-3xi5">Not-in-family</td>
    </tr>
    <tr>
        <td class="tg-c6of">561810758</td>
        <td class="tg-6qlg">50</td>
        <td class="tg-3xi5">Self-emp-not-inc</td>
        <td class="tg-3xi5">83311</td>
        <td class="tg-6qlg">Bachelors</td>
        <td class="tg-3xi5">13</td>
        <td class="tg-3xi5">Married-civ-spouse</td>
        <td class="tg-3xi5">Exec-managerial</td>
        <td class="tg-3xi5">Husband</td>
    </tr>
    <tr>
        <td class="tg-c6of">598098459</td>
        <td class="tg-6qlg">38</td>
        <td class="tg-3xi5">Private</td>
        <td class="tg-3xi5">215646</td>
        <td class="tg-6qlg">HS-grad</td>
        <td class="tg-3xi5">9</td>
        <td class="tg-3xi5">Divorced</td>
        <td class="tg-3xi5">Handlers-cleaners</td>
        <td class="tg-3xi5">Not-in-family</td>
    </tr>
    <tr>
        <td class="tg-c6of">776705221</td>
        <td class="tg-6qlg">53</td>
        <td class="tg-3xi5">Private</td>
        <td class="tg-3xi5">234721</td>
        <td class="tg-6qlg">11th</td>
        <td class="tg-3xi5">7</td>
        <td class="tg-3xi5">Married-civ-spouse</td>
        <td class="tg-3xi5">Handlers-cleaners</td>
        <td class="tg-3xi5">Husband</td>
    </tr>
    <tr>
        <td class="tg-c6of">479262902</td>
        <td class="tg-6qlg">28</td>
        <td class="tg-3xi5">Private</td>
        <td class="tg-3xi5">338409</td>
        <td class="tg-6qlg">Bachelors</td>
        <td class="tg-3xi5">13</td>
        <td class="tg-3xi5">Married-civ-spouse</td>
        <td class="tg-3xi5">Prof-specialty</td>
        <td class="tg-3xi5">Wife</td>
    </tr>
    </tbody></table></div>

\

.. _Binning_Numerical_Columns:

Binning Numerical Columns
---------------------------

Binning numerical columns is a technique used to convert continuous numerical 
data into discrete categories or "bins." This is especially useful for simplifying 
analysis, creating categorical features from numerical data, or visualizing the 
distribution of data within specific ranges. The process of binning involves 
dividing a continuous range of values into a series of intervals, or "bins," and 
then assigning each value to one of these intervals.

.. note::

    The code snippets below create age bins and assign a corresponding age group 
    label to each age in the DataFrame. The ``pd.cut`` function from pandas is used to 
    categorize the ages and assign them to a new column, ``age_group``. Adjust the bins 
    and labels as needed for your specific data.


Below, we use the ``age`` column of the census data [1]_ from the UCI Machine Learning Repository as an example:

1. **Bins Definition**:
   The bins are defined by specifying the boundaries of each interval. For example, 
   in the code snippet below, the ``bin_ages`` list specifies the boundaries for age groups:

   .. code-block:: python

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


   Each pair of consecutive elements in ``bin_ages`` defines a bin. For example:
   
   - The first bin is ``[0, 18)``,
   - The second bin is ``[18, 30)``,
   - and so on.  

\

2. **Labels for Bins**:
   The `label_ages` list provides labels corresponding to each bin:

   .. code-block:: python

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

   These labels are used to categorize the numerical values into meaningful groups.

3. **Applying the Binning**:
   The `pd.cut <https://pandas.pydata.org/docs/reference/api/pandas.cut.html>`_ function 
   from Pandas is used to apply the binning process. For each value in the ``age`` 
   column of the DataFrame, it assigns a corresponding label based on which bin the 
   value falls into. Here, ``right=False`` indicates that each bin includes the 
   left endpoint but excludes the right endpoint. For example, if ``bin_ages = 
   [0, 10, 20, 30]``, then a value of ``10`` will fall into the bin ``[10, 20)`` and 
   be labeled accordingly.

   .. code-block:: python

       df["age_group"] = pd.cut(
           df["age"],
           bins=bin_ages,
           labels=label_ages,
           right=False,
       )

   **Mathematically**, for a given value `x` in the ``age`` column:

   .. math::

       \text{age_group} = 
       \begin{cases} 
        < 18 & \text{if } 0 \leq x < 18 \\
        18-29 & \text{if } 18 \leq x < 30 \\
        \vdots \\
        100 + & \text{if } x \geq 100 
       \end{cases}

   The parameter ``right=False`` in ``pd.cut`` means that the bins are left-inclusive 
   and right-exclusive, except for the last bin, which is always right-inclusive 
   when the upper bound is infinity (``float("inf")``).


.. [1] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.

