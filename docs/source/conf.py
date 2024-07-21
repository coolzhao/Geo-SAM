# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Geo-SAM"
copyright = "2023-2024, Joey, Fancy"
author = "Joey, Fancy"
release = "v1.3-rc"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    # "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = ["colon_fence"]
myst_url_schemes = ["http", "https", "mailto"]
suppress_warnings = ["mystnb.unknown_mime_type"]
nb_execution_mode = "off"
autodoc_inherit_docstrings = True
# templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise site
    "github_user": "coolzhao",
    "github_repo": "Geo-SAM",
    "github_version": "main",
    "doc_path": "docs/source",
}


html_theme_options = {
    "show_toc_level": 2,
    "show_nav_level": 2,
    "use_edit_page_button": True,
    "header_links_before_dropdown": 10,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/coolzhao/Geo-SAM",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/GeoSAM-Image-Encoder",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "announcement": "<a href='https://geo-sam.readthedocs.io/en/latest/blog/2024/05-02_crash_on_QGIS.html'>GeoSAM crash on QGIS 3.34/3.36!</a>",
    # "announcement": "<a href='blog/2024/05-02_crash_on_QGIS.html'>GeoSAM crash on QGIS 3.34/3.36!</a>",
}

video_enforce_extra_source = True
