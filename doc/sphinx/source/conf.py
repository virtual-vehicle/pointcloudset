# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import codecs


sys.path.insert(0, os.path.abspath("../../../pointcloudset"))


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# -- Project information -----------------------------------------------------

project = "pointcloudset"
copyright = "VIRTUAL VEHICLE Research GmbH"
author = (
    "Thomas Goelles, Birgit Schlager, Stefan Muckenhuber, Sarah Haas, Tobias Hammer"
)


version = get_version("../../../pointcloudset/__init__.py")
release = version

# --nbshinx settings ---------------------------------------------------
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
nbsphinx_kernel_name = "base"

nbsphinx_execute = "never"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinxcontrib.napoleon",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinx_click",
]

intersphinx_mapping = {
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "pyntcloud": ("https://pyntcloud.readthedocs.io/en/latest", None),
    "open3d": ("http://www.open3d.org/docs/latest", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
}
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "titles_only": True,
    "prev_next_buttons_location": "both",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images"]
