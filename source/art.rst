.. _art:   

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


==========
ASCII Art 
==========

Overview
========
The ``art.py`` module provides a collection of ASCII art illustrations and utilities 
for displaying or saving them. This documentation outlines the available ASCII art, 
the dictionary structure, and how to use the ``print_art`` function to interact with the artwork.

Features
========
- A dictionary of predefined ASCII art.
- Flexible options to display one, multiple, or all available ASCII art illustrations.
- Capability to save the output to a specified file and directory.

ASCII Art Collection
=====================
The ``ascii_art`` dictionary contains the following predefined artworks:

- **eda_toolkit_logo**: ASCII representation of the EDA Toolkit logo.
- **leon_shpaner_bb**: Black-and-blue ASCII art of "Leon Shpaner".
- **leon_shpaner_wb**: White-and-black ASCII art of "Leon Shpaner".
- **oscar_gil_bb**: Black-and-blue ASCII art of "Oscar Gil".
- **oscar_gil_wb**: White-and-black ASCII art of "Oscar Gil".
- **royce_hall_bb**: Black-and-blue ASCII art of UCLA's Royce Hall.
- **royce_hall_wb**: White-and-black ASCII art of UCLA's Royce Hall.
- **ca_state_bb**: Black-and-blue ASCII art of the State of California.
- **ca_state_wb**: White-and-black ASCII art of the State of California.

.. function:: print_art(*art_names, all=False, output_file=None, output_path=None)

    Print ASCII art based on user input and optionally save output to a specified path.

    :param art_names: Names of the ASCII art to print. Each name should match a key in the ``ascii_art`` dictionary.
    :type art_names: str
    :param all: If ``True``, all available ASCII art will be printed. Defaults to ``False``.
    :type all: bool, optional
    :param output_file: Name of the file to save the output. Defaults to ``.txt`` if no extension is provided.
    :type output_file: str, optional
    :param output_path: Directory where the output file should be saved. If not specified, the current working directory will be used. Non-existent directories will be created automatically.
    :type output_path: str, optional

    :raises ValueError: If both ``art_names`` and ``all=True`` are specified simultaneously.

Examples
========
**Display a Single Artwork**::

    from art import print_art

    print_art("eda_toolkit_logo")

**Display All Artworks**::

    from art import print_art

    print_art(all=True)

**Save Artwork to a File**::

    from art import print_art

    print_art("royce_hall_bb", output_file="royce_hall.txt", output_path="./artworks")

**Handle Missing Artwork**::

    from art import print_art

    print_art("unknown_art")
    # Output: 'unknown_art' not found. Available options are: eda_toolkit_logo, leon_shpaner_bb, ...

Key Details
===========
- **Default Behavior**: If no artwork name is provided and `all` is `False`, a list of available options is displayed.
- **Output Handling**: If `output_file` and `output_path` are specified, the function creates the directory (if needed) and saves the ASCII art to the file.

Notes
=====
Ensure that the `ascii_art` dictionary contains the desired artwork and keys are correctly referenced when using the `print_art` function.
