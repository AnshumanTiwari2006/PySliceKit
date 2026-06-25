import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'PySliceKit'
copyright = '2026, Anshuman Tiwari'
author = 'Anshuman Tiwari'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = [
    'dark_theme.css',
]

pygments_style = 'monokai'
