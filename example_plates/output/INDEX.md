# Plate evaluation — default pipeline, no per-image tuning

For each image, look at the annotated output and the diagnostic
intermediates and call out what's wrong. We tune *after* this, not before.

Total images: **10**

| Image | Input shape | Detected | # flags | Time |
| --- | --- | --- | --- | --- |
| `CFU.jpg` | `(367, 371, 3)` | **54** | 0 | 1840 ms |
| `Example of a spread plate.jpg` | `(378, 385, 3)` | **93** | 0 | 1901 ms |
| `PXL_20251002_031302811.jpg` | `(1506, 2000, 3)` | **170** | 1 | 20710 ms |
| `PXL_20251002_031421585.jpg` | `(1506, 2000, 3)` | **147** | 1 | 19251 ms |
| `PXL_20251002_031445353.jpg` | `(1506, 2000, 3)` | **215** | 1 | 19322 ms |
| `PXL_20251002_031539648.jpg` | `(1506, 2000, 3)` | **213** | 1 | 18299 ms |
| `PXL_20251002_031609943.jpg` | `(1506, 2000, 3)` | **141** | 1 | 19281 ms |
| `PXL_20251002_031623931.jpg` | `(1506, 2000, 3)` | **81** | 1 | 17877 ms |
| `Petri_Easy.jpg` | `(601, 609, 3)` | **217** | 0 | 4542 ms |
| `gm091.jpg` | `(700, 700, 3)` | **95** | 0 | 3601 ms |

## `CFU.jpg`

- Detected colonies: **54**
- Input shape: `(367, 371, 3)`
- Total time: 1840 ms

**Final annotated:**

![annotated](CFU/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](CFU/01_input.jpg)
![plate](CFU/02_plate_overlay.jpg)
![cropped](CFU/03_cropped.jpg)
![illum](CFU/04_illumination_corrected.jpg)
![seg](CFU/05_segmentation.jpg)

---

## `Example of a spread plate.jpg`

- Detected colonies: **93**
- Input shape: `(378, 385, 3)`
- Total time: 1901 ms

**Final annotated:**

![annotated](Example_of_a_spread_plate/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](Example_of_a_spread_plate/01_input.jpg)
![plate](Example_of_a_spread_plate/02_plate_overlay.jpg)
![cropped](Example_of_a_spread_plate/03_cropped.jpg)
![illum](Example_of_a_spread_plate/04_illumination_corrected.jpg)
![seg](Example_of_a_spread_plate/05_segmentation.jpg)

---

## `PXL_20251002_031302811.jpg`

- Detected colonies: **170**
- Input shape: `(1506, 2000, 3)`
- Total time: 20710 ms
- Quality flags:
  - [info] image_resized (in ): Image downscaled from 4080x3072 to 2000x1506 (scale=0.49). Pass max_dimension=None to load at native resolution.

**Final annotated:**

![annotated](PXL_20251002_031302811/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](PXL_20251002_031302811/01_input.jpg)
![plate](PXL_20251002_031302811/02_plate_overlay.jpg)
![cropped](PXL_20251002_031302811/03_cropped.jpg)
![illum](PXL_20251002_031302811/04_illumination_corrected.jpg)
![seg](PXL_20251002_031302811/05_segmentation.jpg)

---

## `PXL_20251002_031421585.jpg`

- Detected colonies: **147**
- Input shape: `(1506, 2000, 3)`
- Total time: 19251 ms
- Quality flags:
  - [info] image_resized (in ): Image downscaled from 4080x3072 to 2000x1506 (scale=0.49). Pass max_dimension=None to load at native resolution.

**Final annotated:**

![annotated](PXL_20251002_031421585/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](PXL_20251002_031421585/01_input.jpg)
![plate](PXL_20251002_031421585/02_plate_overlay.jpg)
![cropped](PXL_20251002_031421585/03_cropped.jpg)
![illum](PXL_20251002_031421585/04_illumination_corrected.jpg)
![seg](PXL_20251002_031421585/05_segmentation.jpg)

---

## `PXL_20251002_031445353.jpg`

- Detected colonies: **215**
- Input shape: `(1506, 2000, 3)`
- Total time: 19322 ms
- Quality flags:
  - [info] image_resized (in ): Image downscaled from 4080x3072 to 2000x1506 (scale=0.49). Pass max_dimension=None to load at native resolution.

**Final annotated:**

![annotated](PXL_20251002_031445353/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](PXL_20251002_031445353/01_input.jpg)
![plate](PXL_20251002_031445353/02_plate_overlay.jpg)
![cropped](PXL_20251002_031445353/03_cropped.jpg)
![illum](PXL_20251002_031445353/04_illumination_corrected.jpg)
![seg](PXL_20251002_031445353/05_segmentation.jpg)

---

## `PXL_20251002_031539648.jpg`

- Detected colonies: **213**
- Input shape: `(1506, 2000, 3)`
- Total time: 18299 ms
- Quality flags:
  - [info] image_resized (in ): Image downscaled from 4080x3072 to 2000x1506 (scale=0.49). Pass max_dimension=None to load at native resolution.

**Final annotated:**

![annotated](PXL_20251002_031539648/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](PXL_20251002_031539648/01_input.jpg)
![plate](PXL_20251002_031539648/02_plate_overlay.jpg)
![cropped](PXL_20251002_031539648/03_cropped.jpg)
![illum](PXL_20251002_031539648/04_illumination_corrected.jpg)
![seg](PXL_20251002_031539648/05_segmentation.jpg)

---

## `PXL_20251002_031609943.jpg`

- Detected colonies: **141**
- Input shape: `(1506, 2000, 3)`
- Total time: 19281 ms
- Quality flags:
  - [info] image_resized (in ): Image downscaled from 4080x3072 to 2000x1506 (scale=0.49). Pass max_dimension=None to load at native resolution.

**Final annotated:**

![annotated](PXL_20251002_031609943/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](PXL_20251002_031609943/01_input.jpg)
![plate](PXL_20251002_031609943/02_plate_overlay.jpg)
![cropped](PXL_20251002_031609943/03_cropped.jpg)
![illum](PXL_20251002_031609943/04_illumination_corrected.jpg)
![seg](PXL_20251002_031609943/05_segmentation.jpg)

---

## `PXL_20251002_031623931.jpg`

- Detected colonies: **81**
- Input shape: `(1506, 2000, 3)`
- Total time: 17877 ms
- Quality flags:
  - [info] image_resized (in ): Image downscaled from 4080x3072 to 2000x1506 (scale=0.49). Pass max_dimension=None to load at native resolution.

**Final annotated:**

![annotated](PXL_20251002_031623931/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](PXL_20251002_031623931/01_input.jpg)
![plate](PXL_20251002_031623931/02_plate_overlay.jpg)
![cropped](PXL_20251002_031623931/03_cropped.jpg)
![illum](PXL_20251002_031623931/04_illumination_corrected.jpg)
![seg](PXL_20251002_031623931/05_segmentation.jpg)

---

## `Petri_Easy.jpg`

- Detected colonies: **217**
- Input shape: `(601, 609, 3)`
- Total time: 4542 ms

**Final annotated:**

![annotated](Petri_Easy/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](Petri_Easy/01_input.jpg)
![plate](Petri_Easy/02_plate_overlay.jpg)
![cropped](Petri_Easy/03_cropped.jpg)
![illum](Petri_Easy/04_illumination_corrected.jpg)
![seg](Petri_Easy/05_segmentation.jpg)

---

## `gm091.jpg`

- Detected colonies: **95**
- Input shape: `(700, 700, 3)`
- Total time: 3601 ms

**Final annotated:**

![annotated](gm091/06_annotated.jpg)

**Stages (input → plate → cropped → illumination → segmentation):**

![input](gm091/01_input.jpg)
![plate](gm091/02_plate_overlay.jpg)
![cropped](gm091/03_cropped.jpg)
![illum](gm091/04_illumination_corrected.jpg)
![seg](gm091/05_segmentation.jpg)

---
