"""Built-in pipeline modules.

Importing this package registers every built-in module with
:data:`blenny.MODULES` as a side effect, so `Pipeline.from_config`
configs can refer to them by their short registry names
(``"load_image"``, ``"detect_plate"``, ...).
"""

from blenny.modules.correct_illumination import IlluminationCorrection
from blenny.modules.detect_plate import PlateDetector
from blenny.modules.export_annotated import AnnotatedImageExporter
from blenny.modules.export_csv import CSVExporter
from blenny.modules.identity import IdentityPreprocessor
from blenny.modules.load_image import ImageFileLoader
from blenny.modules.measure_colonies import ColonyMeasurer
from blenny.modules.threshold_segment import ThresholdSegmenter

__all__ = [
    "AnnotatedImageExporter",
    "CSVExporter",
    "ColonyMeasurer",
    "IdentityPreprocessor",
    "IlluminationCorrection",
    "ImageFileLoader",
    "PlateDetector",
    "ThresholdSegmenter",
]
