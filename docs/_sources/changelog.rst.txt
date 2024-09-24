.. _changelog:   

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

Changelog
=========

`Version 0.0.7`_
-------------------------

.. _Version 0.0.7: https://lshpaner.github.io/eda_toolkit/v0.0.7/

**Added Function for Customizable Correlation Matrix Visualization**

This release introduces a new function, ``flex_corr_matrix``, which allows users to 
generate both full and upper triangular correlation heatmaps with a high degree 
of customization. The function includes options to annotate the heatmap, save the 
plots, and pass additional parameters to ``seaborn.heatmap()``.

**Summary of Changes**

- **New Function**: ``flex_corr_matrix``.

  - **Functionality**:
    - Generates a correlation heatmap for a given DataFrame.
    - Supports both full and upper triangular correlation matrices based on the ``triangular`` parameter.
    - Allows users to customize various aspects of the plot, including colormap, figure size, axis label rotation, and more.
    - Accepts additional keyword arguments via ``**kwargs`` to pass directly to ``seaborn.heatmap()``.
    - Includes validation to ensure the ``triangular``, ``annot``, and ``save_plots`` parameters are boolean values.
    - Raises an exception if ``save_plots=True`` but neither ``image_path_png`` nor ``image_path_svg`` is specified.

**Usage**

.. code-block:: python

   # Full correlation matrix example
   flex_corr_matrix(df=my_dataframe, triangular=False, cmap="coolwarm", annot=True)

   # Upper triangular correlation matrix example
   flex_corr_matrix(df=my_dataframe, triangular=True, cmap="coolwarm", annot=True)


**Contingency table df to object type**

Convert all columns in the DataFrame to object type to prevent issues with numerical columns.

.. code-block:: python

   df = df.astype(str).fillna("")


`Version 0.0.6`_
-------------------------

.. _Version 0.0.6: https://lshpaner.github.io/eda_toolkit/v0.0.6/

**Added validation for Plot Type Parameter in KDE Distributions Function**

This release adds a validation step for the ``plot_type`` parameter in the ``kde_distributions`` function. The allowed values for ``plot_type`` are ``"hist"``, ``"kde"``, and ``"both"``. If an invalid value is provided, the function will now raise a ``ValueError`` with a clear message indicating the accepted values. This change improves the robustness of the function and helps prevent potential errors due to incorrect parameter values.

.. code-block:: python 
   
    # Validate plot_type parameter
    valid_plot_types = ["hist", "kde", "both"]
    if plot_type.lower() not in valid_plot_types:
        raise ValueError(
            f"Invalid plot_type value. Expected one of {valid_plot_types}, "
            f"got '{plot_type}' instead."
        )

`Version 0.0.5`_
-------------------------

.. _Version 0.0.5: https://lshpaner.github.io/eda_toolkit/v0.0.5/

**Ensure Consistent Font Size and Text Wrapping Across Plot Elements**

This PR addresses inconsistencies in font sizes and text wrapping across various plot elements in the ``stacked_crosstab_plot`` function. The following updates have been implemented to ensure uniformity and improve the readability of plots:

1. **Title Font Size and Text Wrapping:**
   - Added a ``text_wrap`` parameter to control the wrapping of plot titles.
   - Ensured that title font sizes are consistent with axis label font sizes by explicitly setting the font size using ``ax.set_title()`` after plot generation.

2. **Legend Font Size Consistency:**
   - Incorporated ``label_fontsize`` into the legend font size by directly setting the font size of the legend text using ``plt.setp(legend.get_texts(), fontsize=label_fontsize)``.
   - This ensures that the legend labels are consistent with the title and axis labels.

**Testing**

- Verified that titles now wrap correctly and match the specified ``label_fontsize``.
- Confirmed that legend text scales according to ``label_fontsize``, ensuring consistent font sizes across all plot elements.


Version 0.0.4 
---------------------------

- **Stable release**

  - No new updates to the codebase.
  
  - Updated the project ``description`` variable in ``setup.py`` to re-emphasize key elements of the library.
  
  - Minor README cleanup:
  
    - Added icons for sections that did not have them.


Version 0.0.3 
---------------------------

- **Stable release**

  - Updated logo size, fixed citation title, and made minor README cleanup:

    - Added an additional section for documentation, cleaned up verbiage, moved acknowledgments section before licensing and support.

Version 0.0.2 
---------------------------

- **First stable release**
   - No new updates to the codebase; minimal documentation updates to README and ``setup.py`` files.
   - Added logo, badges, and Zenodo-certified citation to README.

Version 0.0.1rc0 
-------------------------------

- No new updates to the codebase; minimal documentation updates to README and ``setup.py`` files.

Version 0.0.1b0 
-----------------------------

**New Scatter Fit Plot and Additional Updates**

- Added new ``scatter_fit_plot()``, removed unused ``data_types()``, and added comment section headers.

**Added xlim and ylim Inputs to KDE Distribution**

- ``kde_distribution()``:

    - Added ``xlim`` and ``ylim`` inputs to allow users to customize axes limits in ``kde_distribution()``.

**Added xlim and ylim Params to Stacked Crosstab Plot**

- ``stacked_crosstab_plot()``:

    - Added ``xlim`` and ``ylim`` input parameters to ``stacked_crosstab_plot()`` to give users more flexibility in controlling axes limits.

**Added x and y Limits to Box and Violin Plots**

- ``box_violin_plot()``: 

    - Changed function name from ``metrics_box_violin()`` to ``box_violin_plot()``.
    - Added ``xlim`` and ``ylim`` inputs to control x and y-axis limits of ``box_violin_plot()`` (formerly ``metrics_box_violin``).

**Added Ability to Remove Stacks from Plots, Plot All or One at a Time**

**Key Changes**

1. **Plot Type Parameter**
   - ``plot_type``: This parameter allows the user to choose between ``"regular"``, ``"normalized"``, or ``"both"`` plot types.

2. **Remove Stacks Parameter**
   - ``remove_stacks``: This parameter, when set to ``True``, generates a regular bar plot using only the ``col`` parameter instead of a stacked bar plot. It only works when ``plot_type`` is set to "regular". If ``remove_stacks`` is set to ``True`` while ``plot_type`` is anything other than "regular", the function will raise an exception.

**Explanation of Changes**

- **Plot Type Parameter**

  - Provides flexibility to the user, allowing specification of the type of plot to generate:

    - ``"regular"``: Standard bar plot.

    - ``"normalized"``: Normalized bar plot.

    - ``"both"``: Both regular and normalized bar plots.

- **Remove Stacks Parameter**
  - ``remove_stacks``: Generates a regular bar plot using only the ``col`` parameter, removing the stacking of the bars. Applicable only when ``plot_type`` is set to "regular". An exception is raised if used with any other ``plot_type``.

These changes enhance the flexibility and functionality of the ``stacked_crosstab_plot`` function, allowing for more customizable and specific plot generation based on user requirements.

Version 0.0.1b0 
-----------------------------

**Refined KDE Distributions**

**Key Changes**

1. **Alpha Transparency for Histogram Fill**
   - Added a ``fill_alpha`` parameter to control the transparency of the histogram bars' fill color.
   - Default value is ``0.6``. An exception is raised if ``fill=False`` and ``fill_alpha`` is specified.

2. **Custom Font Sizes**
   - Introduced ``label_fontsize`` and ``tick_fontsize`` parameters to control font size of axis labels and tick marks independently.

3. **Scientific Notation Toggle**
   - Added a ``disable_sci_notation`` parameter to enable or disable scientific notation on axes.

4. **Improved Error Handling**
   - Added validation for the ``stat`` parameter to ensure valid options are accepted.
   - Added checks for proper usage of ``fill_alpha`` and ``hist_edgecolor`` when ``fill`` is set to ``False``.

5. **General Enhancements**
   - Updated the function's docstring to reflect new parameters and provide comprehensive guidance on usage.

Version 0.0.1b0 
-----------------------------

**Enhanced KDE Distributions Function**

**Added Parameters**

1. **Grid Figsize and Single Figsize**
   - Control the size of the overall grid figure and individual figures separately.

2. **Hist Color and KDE Color`**
   - Allow customization of histogram and KDE plot colors.

3. **Edge Color**
   - Allows customization of histogram bar edges.

4. **Hue**
   - Allows grouping data by a column.

5. **Fill**
   - Controls whether to fill histogram bars with color.

6. **Y-axis Label`**
   - Customizable y-axis label.

7. **Log-Scaling**
   - Specifies which variables to apply log scale.

8. **Bins and Bin Width**
   - Control the number and width of bins.

9. **``stat``:**
   - Allows different statistics for the histogram (``count``, ``density``, ``frequency``, ``probability``, ``proportion``, ``percent``).

**Improvements**

1. **Validation and Error Handling**
   - Checks for invalid ``log_scale_vars`` and throws a ``ValueError`` if any are found.
   - Throws a ``ValueError`` if ``edgecolor`` is changed while ``fill`` is set to ``False``.
   - Issues a ``PerformanceWarning`` if both ``bins`` and ``binwidth`` are specified, warning of potential performance impacts.

2. **Customizable Y-Axis Label**
   - Allows users to specify custom y-axis labels.

3. **Warning for KDE with Count**
   - Issues a warning if KDE is used with ``stat='count'``, as it may produce misleading plots.

**Updated Function to Ensure Unique IDs and Index Check**

- Ensured that each generated ID in ``add_ids`` starts with a non-zero digit.
- Added a check to verify that the DataFrame index is unique.
- Printed a warning message if duplicate index entries are found.

These changes improve the robustness of the function, ensuring that the IDs generated are always unique and valid, and provide necessary feedback when the DataFrame index is not unique.

**Check for Unique Indices**
- Before generating IDs, the function now checks if the DataFrame index is unique.
- If duplicates are found, a warning is printed along with the list of duplicate index entries.

**Generate Non-Zero Starting IDs**

- The ID generation process is updated to ensure that the first digit of each ID is always non-zero.

**Ensure Unique IDs**

- A set is used to store the generated IDs, ensuring all IDs are unique before adding them to the DataFrame.

**Fix Int Conversion for Numeric Columns, Reset Decimal Places**

- Fixed integer conversion issue for numeric columns when ``decimal_places=0`` in the ``save_dataframes_to_excel`` function.
- Reset ``decimal_places`` default value to ``0``.

These changes ensure correct formatting and avoid errors during conversion.

**Contingency Table Updates**

1. **Error Handling for Columns**
   - Added a check to ensure at least one column is specified.
   - Updated the function to accept a single column as a string or multiple columns as a list.
   - Raised a ``ValueError`` if no columns are provided or if ``cols`` is not correctly specified.

2. **Function Parameters**
   - Changed parameters from ``col1`` and ``col2`` to a single parameter ``cols`` which can be either a string or a list.

3. **Error Handling**
   - Renamed ``SortBy`` to ``sort_by`` to standardize nomenclature.
   - Added a check to ensure ``sort_by`` is either 0 or 1.
   - Raised a ``ValueError`` if ``sort_by`` is not 0 or 1.

5. **Sorting Logic**
   - Updated the sorting logic to handle the new ``cols`` parameter structure.

6. **Handling Categorical Data**
   - Modified code to convert categorical columns to strings to avoid issues with ``fillna("")``.

7. **Handling Missing Values**
   - Added ``df = df.fillna('')`` to fill NA values within the function to account for missing data.

8. **Improved Function Documentation**
   - Updated function documentation to reflect new parameters and error handling.

Version 0.0.1b0 
-----------------------------

**Contingency Table Updates**

- ``fillna('')`` added to output so that null values come through, removed ``'All'`` column name from output, sort options ``0`` and ``1``, updated docstring documentation. Tested successfully on ``Python 3.7.3``.

**Compatibility Enhancement**

1. Added a version check for ``Python 3.7`` and above.

   - Conditional import of ``datetime`` to handle different Python versions.

.. code-block:: python

    if sys.version_info >= (3, 7):
        from datetime import datetime
    else:
        import datetime
