# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EDA Toolkit"
copyright = "2024, Leonid Shpaner, Oscar Gil"
author = "Leonid Shpaner, Oscar Gil"
release = "0.0.1b0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # "sphinxcontrib.bibtex",
]

# Add this line to specify the bibliography file
# bibtex_bibfiles = ["references.bib"]

extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

copybutton_prompt_text = r">|\$ "
copybutton_prompt_is_regexp = True


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# If your documentation is served from a subdirectory, set this to the subdirectory path
html_show_sourcelink = False


# def setup(app):
#     app.add_css_file("custom.css")
#     app.add_js_file("custom.js")
