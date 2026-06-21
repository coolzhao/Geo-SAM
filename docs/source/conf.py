# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Geo-SAM"
copyright = "2023-2026, Joey, Fancy"
author = "Joey, Fancy"
release = "v2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinxcontrib.video",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = []

# -- MyST extensions ---------------------------------------------------------
# Enable Markdown features used by sphinx-design and rich docs.

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

video_enforce_extra_source = True

# -- Theme options -----------------------------------------------------------
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/coolzhao/Geo-SAM",
            "icon": "fab fa-github",
        },
        {
            "name": "Documentation",
            "url": "https://geo-sam.readthedocs.io/en/latest/",
            "icon": "fas fa-book",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

html_context = {
    "github_user": "coolzhao",
    "github_repo": "Geo-SAM",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Copybutton options ------------------------------------------------------
# Strip the prompt characters when copying code blocks.

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Intersphinx options -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
