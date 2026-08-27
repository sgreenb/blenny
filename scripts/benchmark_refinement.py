import time

import numpy as np
from PIL import Image
from skimage import color

from blenny.modules.detect_multi_plate import MultiPlateDetector
from blenny.pipeline.context import ImageData


def profile_refinement():
    img_path = "sandbox/diverse_tests/case01_standard_2x3.jpg"
    img = np.asarray(Image.open(img_path))
    data = ImageData(source=img_path, image=img)

    # 1. With Refinement (Current State)
    detector = MultiPlateDetector(grid=[2, 3], min_confidence_score=0.05)
    t0 = time.perf_counter()
    detector.run(data)
    t_with = time.perf_counter() - t0
    print(f"Multi-plate detection WITH refinement: {t_with:.3f}s")

    # 2. Mocking removal of refinement by monkey-patching or just observing code
    # Actually, I'll just look at the provenance if I run it through the pipeline
    # but the module doesn't split it out.
    # Let's just do a quick manual check of how long Canny takes on a crop.
    gray = color.rgb2gray(img)
    h, w = gray.shape
    cell_h, cell_w = h // 2, w // 3
    crop = gray[0:cell_h, 0:cell_w]

    from skimage import feature

    t0 = time.perf_counter()
    for _ in range(6):
        feature.canny(crop, sigma=1.5)
    t_canny = time.perf_counter() - t0
    print(f"Time spent in 6x High-Res Canny: {t_canny:.3f}s")


if __name__ == "__main__":
    profile_refinement()
