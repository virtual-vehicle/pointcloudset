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
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

sys.path.insert(0, os.path.abspath("../../../src/pointcloudset"))


def get_version(_rel_path):
    try:
        return pkg_version("pointcloudset")
    except PackageNotFoundError:
        pass

    pyproject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../pyproject.toml"))
    with open(pyproject_path, "rb") as fp:
        pyproject = tomllib.load(fp)

    return pyproject["project"]["version"]


# -- Project information -----------------------------------------------------

project = "pointcloudset"
copyright = "VIRTUAL VEHICLE Research GmbH"
author = "Thomas Goelles, Birgit Schlager, Stefan Muckenhuber, Sarah Haas, Tobias Hammer"


version = get_version("../../../src/pointcloudset/__init__.py")
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
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinx_click",
]

intersphinx_mapping = {
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "pyntcloud": ("https://pyntcloud.readthedocs.io/en/latest", None),
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
