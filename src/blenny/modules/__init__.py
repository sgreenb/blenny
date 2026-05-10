"""Built-in pipeline modules.

Importing this package registers every built-in module with
:data:`blenny.MODULES` as a side effect, so `Pipeline.from_config`
configs can refer to them by their short registry names
(``"load_image"``, ``"detect_plate"``, ...).
"""

from blenny.modules.add_manual_colonies import ManualColonyAdder
from blenny.modules.classify_interior import InteriorColonyClassifier
from blenny.modules.classify_threshold import ThresholdClassifier
from blenny.modules.correct_illumination import IlluminationCorrection
from blenny.modules.detect_multi_plate import MultiPlateDetector
from blenny.modules.detect_plate import PlateDetector
from blenny.modules.estimate_multiplicity import MultiplicityEstimator
from blenny.modules.export_annotated import AnnotatedImageExporter
from blenny.modules.export_csv import CSVExporter
from blenny.modules.export_summary import SummaryExporter
from blenny.modules.filter_by_id import IDFilter
from blenny.modules.identity import IdentityPreprocessor
from blenny.modules.load_image import ImageFileLoader
from blenny.modules.mask_exclusion import ExclusionMasker
from blenny.modules.measure_colonies import ColonyMeasurer
from blenny.modules.sub_pipeline import SubPipeline
from blenny.modules.threshold_segment import ThresholdSegmenter
from blenny.modules.yolo_detector import YoloDetector

__all__ = [
    "AnnotatedImageExporter",
    "CSVExporter",
    "ColonyMeasurer",
    "ExclusionMasker",
    "IDFilter",
    "IdentityPreprocessor",
    "IlluminationCorrection",
    "ImageFileLoader",
    "InteriorColonyClassifier",
    "ManualColonyAdder",
    "MultiPlateDetector",
    "MultiplicityEstimator",
    "PlateDetector",
    "SubPipeline",
    "SummaryExporter",
    "ThresholdClassifier",
    "ThresholdSegmenter",
    "YoloDetector",
]
