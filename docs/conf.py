#
# sphinx-next documentation build configuration file.
# Minimal sphinx configuration supporting export of API-docs to JSPN.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(os.path.abspath('..'))

print("python exec:", sys.executable)
print("sys.path:", sys.path)

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
]

autosummary_generate = True
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

autodoc_mock_imports = ["xesmf", "tensorflow"]


# Sphinx project configuration
source_suffix = ".rst"
needs_sphinx = "1.8"
# The encoding of source files.
source_encoding = "utf-8-sig"
root_doc = "index"

# General information about the project.
project = "cmip6_downscaling"
copyright = "2022, CarbonPlan"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
# version = cmip6_downscaling.__version__
# # The full version, including alpha/beta/rc tags.
# release = cmip6_downscaling.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = "Python"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- Options for HTML output ----------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_title = "CMIP6-Downscaling"

html_theme = "sphinx_book_theme"
html_title = ""
repository = "carbonplan/cmip6-downscaling"
repository_url = "https://github.com/carbonplan/cmip6-downscaling"
