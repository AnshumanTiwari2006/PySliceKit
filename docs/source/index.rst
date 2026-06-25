Welcome to PySliceKit's documentation!
======================================

.. image:: https://badge.fury.io/py/pyslicekit.svg
    :target: https://pypi.org/project/pyslicekit/
.. image:: https://img.shields.io/badge/Tests-passing-brightgreen.svg
    :target: https://github.com/AnshumanTiwari2006/PySliceKit/actions
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/github/stars/AnshumanTiwari2006/PySliceKit.svg?style=social&label=Star
    :target: https://github.com/AnshumanTiwari2006/PySliceKit

**GitHub Repository:** `github.com/AnshumanTiwari2006/PySliceKit <https://github.com/AnshumanTiwari2006/PySliceKit>`_

**The Problem:** Relying on global metrics like "95% accuracy" masks critical algorithmic bias, data drift, and localized underfitting where your model is secretly failing.

**PySliceKit** is an automated detective for Machine Learning models that solves this by doing five things automatically:

1. **Bins** continuous columns into quartiles.
2. **Cross-Products** features to find intersectional failures.
3. **Applies Statistical Rigor** (Z-Tests, Fisher's Exact, Bootstrapping) to ensure failures are real.
4. **Flags** low-sample segments.
5. **Enforces a Visual Contract** where "Red always means bad", regardless of metric direction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   api
   faq
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
