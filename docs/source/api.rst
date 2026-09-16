API Reference
=============

Core Functions
--------------

.. autofunction:: pyslicekit.api.evaluate

Data Types
----------

.. autoclass:: pyslicekit.types.SliceResult
   :members:
   :undoc-members:

Exporters
---------

.. autofunction:: pyslicekit.exporter.to_csv
.. autofunction:: pyslicekit.exporter.to_json

Exceptions
----------

.. autoexception:: pyslicekit.exceptions.PySliceKitError
   :members:
.. autoexception:: pyslicekit.exceptions.PySliceKitValidationError
   :members:
.. autoexception:: pyslicekit.exceptions.PySliceKitNoSegmentsError
   :members:


Correction
----------

.. autofunction:: pyslicekit.correction.apply_correction

.. autofunction:: pyslicekit.correction_render.render_correction_comparison

.. autofunction:: pyslicekit.correction_report.export_correction_report
