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

Version 0.0.8
--------------------

:class:`stacked_crosstab_plot` 

- **Flexible `save_formats` Input**:
  - `save_formats` now accepts a string, tuple, or list for specifying formats (e.g., `"png"`, `("png", "svg")`, or `["png", "svg"]`).
  - Single strings or tuples are automatically converted to lists for consistent processing.

- **Dynamic Error Handling**:
  - Added checks to ensure a valid path is provided for each format in `save_formats`.
  - Raises a `ValueError` if a format is specified without a corresponding path, with a clear, dynamic error message.

- **Improved Plot Saving Logic**:
  - Updated logic allows saving plots in one format (e.g., only `"png"` or `"svg"`) without requiring the other.
  - Simplified and more intuitive path handling for saving plots.


:class:`plot_3d_pdp`
 
This update introduces several key changes to the `plot_3d_pdp` function, simplifying the function's interface and improving usability, while maintaining the flexibility needed for diverse visualization needs.

**1. Parameter Changes**


- **Removed Parameters:**
  
  - The parameters ``x_label_plotly``, ``y_label_plotly``, and ``z_label_plotly`` have been removed. These parameters previously allowed custom axis labels specifically for the Plotly plot, defaulting to the general ``x_label``, ``y_label``, and ``z_label``. Removing these parameters simplifies the function signature while maintaining flexibility.

- **Default Values for Labels:**

  - The parameters ``x_label``, ``y_label``, and ``z_label`` are now optional, with ``None`` as the default. If not provided, these labels will automatically default to the names of the features in the ``feature_names_list``. This change makes the function more user-friendly, particularly for cases where default labels are sufficient.

- **Changes in Default Values for View Angles:**

  - The default values for camera positioning parameters have been updated: ``horizontal`` is now ``-1.25``, ``depth`` is now ``1.25``, and ``vertical`` is now ``1.25``. These adjustments refine the default 3D view perspective for the Plotly plot, providing a more intuitive starting view.

**2. Plot Generation Logic**

- **Conditionally Checking Labels:**

  - The function now checks whether ``x_label``, ``y_label``, and ``z_label`` are provided. If these are ``None``, the function will automatically assign default labels based on the ``feature_names_list``. This enhancement reduces the need for users to manually specify labels, making the function more adaptive.

- **Camera Position Adjustments:**

  - The camera positions for the Plotly plot are now adjusted by multiplying ``horizontal``, ``depth``, and ``vertical`` by ``zoom_out_factor``. This change allows for more granular control over the 3D view, enhancing the interactivity and flexibility of the Plotly visualizations.

- **Surface Plot Coordinates Adjustments:**

  - The order of the coordinates for the Plotly plot’s surface has been changed from ``ZZ, XX, YY[::-1]`` to ``ZZ, XX, YY``. This adjustment ensures the proper alignment of axes and grids, resulting in more accurate visual representations.

**3. Code Simplifications**

- **Removed Complexity:**

  - By removing the ``x_label_plotly``, ``y_label_plotly``, and ``z_label_plotly`` parameters, the code is now simpler and easier to maintain. This change reduces potential confusion and streamlines the function for users who do not need distinct labels for Matplotlib and Plotly plots.

- **Fallback Mechanism for Grid Values:**

  - The function continues to implement a fallback mechanism when extracting grid values, ensuring compatibility with various versions of scikit-learn. This makes the function robust across different environments.

**4. Style Adjustments**

- **Label Formatting:**

  - The new version consistently uses ``y_label``, ``x_label``, and ``z_label`` for axis labels in the Matplotlib plot, aligning the formatting across different plot types.

- **Color Bar Adjustments:**

  - The color bar configuration in the Matplotlib plot has been slightly adjusted with a shrink value of ``0.6`` and a pad value of ``0.02``. These adjustments result in a more refined visual appearance, particularly in cases where space is limited.

**5. Potential Use Case Differences**

- **Simplified Interface:**

  - The updated function is more streamlined for users who prefer a simplified interface without the need for separate label customizations for Plotly and Matplotlib plots. This makes it easier to use in common scenarios.

- **Less Granular Control:**

  - Users who need more granular control, particularly for presentations or specific formatting, may find the older version more suitable. The removal of the ``*_plotly`` label parameters means that all plots now use the same labels across Matplotlib and Plotly.

**6. Matplotlib Plot Adjustments**

- **Wireframe and Surface Plot Enhancements:**

  - The logic for plotting wireframes and surface plots in Matplotlib remains consistent with previous versions, with subtle enhancements to color and layout management to improve overall aesthetics.

**Summary**

- Version ``0.0.8d`` of the `plot_3d_pdp` function introduces simplifications that reduce the number of parameters and streamline the plotting process. While some customizability has been removed, the function remains flexible enough for most use cases and is easier to use.
- Key updates include adjusted default camera views for 3D plots, removal of Plotly-specific label parameters, and improved automatic labeling and plotting logic.

**Decision Point**

- This update may be especially useful for users who prefer a cleaner and more straightforward interface. However, those requiring detailed customizations may want to continue using the older version, depending on their specific needs.


Version 0.0.8c
------------------------

Version 0.0.8c is a follow-up release to version 0.0.8b. This update includes minor enhancements and refinements based on feedback and additional testing. It serves as an incremental step towards improving the stability and functionality of the toolkit.

**Key Updates in 0.0.8c:**

- **Bug Fixes:** Addressed minor issues identified in version ``0.0.8b`` to ensure smoother performance and better user experience.
- **Additional Testing:** Incorporated further tests to validate the changes introduced in previous versions and to prepare for future stable releases.
- **Refinements:** Made small enhancements to existing features based on user feedback and internal testing results.

**Summary of Changes**

1. New Features & Enhancements

- ``plot_3d_pdp`` Function:
  
  - Added ``show_modebar`` Parameter: Introduced a new boolean parameter, ``show_modebar``, to allow users to toggle the visibility of the mode bar in Plotly interactive plots.
  
  - Custom Margins and Layout Adjustments:
    
    - Added parameters for ``left_margin``, ``right_margin``, and ``top_margin`` to provide users with more control over the plot layout in Plotly.
    
    - Adjusted default values and added options for better customization of the Plotly color bar (``cbar_x``, ``cbar_thickness``) and title positioning (``title_x``, ``title_y``).
  
  - Plotly Configuration:
    
    - Enhanced the configuration options to allow users to enable or disable zoom functionality (``enable_zoom``) in the interactive Plotly plots.
    
    - Updated the code to reflect these new parameters, allowing for greater flexibility in the appearance and interaction with the Plotly plots.
  
  - Error Handling:
    
    - Added input validation for ``html_file_path`` and ``html_file_name`` to ensure these are provided when necessary based on the selected ``plot_type``.

- ``plot_2d_pdp`` Function:
  
  - Introduced ``file_prefix`` Parameter:
    
    - Added a new ``file_prefix`` parameter to allow users to specify a prefix for filenames when saving grid plots. This change streamlines the naming process for saved plots and improves file organization.
  
  - Enhanced Plot Type Flexibility:
    
    - The ``plot_type`` parameter now includes an option to generate both grid and individual plots (``both``). This feature allows users to create a combination of both layout styles in one function call.
    
    - Updated input validation and logic to handle this new option effectively.
  
  - Added ``save_plots`` Parameter:
    
    - Introduced a new parameter, ``save_plots``, to control the saving of plots. Users can specify whether to save all plots, only individual plots, only grid plots, or none.
  
  - Custom Margins and Layout Adjustments:
    
    - Included the ``save_plots`` parameter in the validation process to ensure paths are provided when needed for saving the plots.

2. Documentation Updates

- Docstrings:
  
  - Updated docstrings for both functions to reflect the new parameters and enhancements, providing clearer and more comprehensive guidance for users.
  
  - Detailed the use of new parameters such as ``show_modebar``, ``file_prefix``, ``save_plots``, and others, ensuring that the function documentation is up-to-date with the latest changes.

3. Refactoring & Code Cleanup

- Code Structure:
  
  - Improved the code structure to maintain clarity and readability, particularly around the new functionality.
  
  - Consolidated the layout configuration settings for the Plotly plots into a more flexible and user-friendly format, making it easier for users to customize their plots.


Version 0.0.8b
--------------------------------

Version 0.0.8b is an exact replica of version ``0.0.8a``. The purpose of this 
beta release was to test whether releasing it as the latest version would update 
its status on PyPI to reflect it as the latest release. However, it continues to 
be identified as a pre-release on PyPI.


Version 0.0.8a
--------------------------------

Version 0.0.8a introduces significant enhancements and new features to improve 
the usability and functionality of the EDA Toolkit.

**New Features:**

1. Optional ``file_prefix`` in ``stacked_crosstab_plot`` Function
   
   - The ``stacked_crosstab_plot`` function has been updated to make the ``file_prefix`` argument optional. If the user does not provide a ``file_prefix``, the function will now automatically generate a default prefix based on the ``col`` and ``func_col`` parameters. This change streamlines the process of generating plots by reducing the number of required arguments.
   
   - **Key Improvement:**
     
     - Users can now omit the ``file_prefix`` argument, and the function will still produce appropriately named plot files, enhancing ease of use.
     
     - Backward compatibility is maintained, allowing users who prefer to specify a custom ``file_prefix`` to continue doing so without any issues.

2. **Introduction of 3D and 2D Partial Dependence Plot Functions**
   
   - Two new functions, ``plot_3d_pdp`` and ``plot_2d_pdp``, have been added to the toolkit, expanding the visualization capabilities for machine learning models.
     
     - ``plot_3d_pdp``: Generates 3D partial dependence plots for two features, supporting both static visualizations (using Matplotlib) and interactive plots (using Plotly). The function offers extensive customization options, including labels, color maps, and saving formats.
     
     - ``plot_2d_pdp``: Creates 2D partial dependence plots for specified features with flexible layout options (grid or individual plots) and customization of figure size, font size, and saving formats.
   
   - **Key Features:**
     
     - **Compatibility:** Both functions are compatible with various versions of scikit-learn, ensuring broad usability.
     
     - **Customization:** Extensive options for customizing visual elements, including figure size, font size, and color maps.
     
     - **Interactive 3D Plots:** The ``plot_3d_pdp`` function supports interactive visualizations, providing an enhanced user experience for exploring model predictions in 3D space.

**Impact:**

- These updates improve the user experience by reducing the complexity of function calls and introducing powerful new tools for model interpretation.
- The optional ``file_prefix`` enhancement simplifies plot generation while maintaining the flexibility to define custom filenames.
- The new partial dependence plot functions offer robust visualization options, making it easier to analyze and interpret the influence of specific features in machine learning models.



`Version 0.0.7`_
----------------------

.. _Version 0.0.7: file:///C:/Users/lshpaner/Documents/Python_Projects/eda_toolkit/docs/v0.0.7/index.html

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
----------------------

.. _Version 0.0.6: file:///C:/Users/lshpaner/Documents/Python_Projects/eda_toolkit/docs/v0.0.6/index.html

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
----------------------

.. _Version 0.0.5: file:///C:/Users/lshpaner/Documents/Python_Projects/eda_toolkit/docs/v0.0.5/index.html


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
