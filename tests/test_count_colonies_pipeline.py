"""End-to-end integration test: assemble the full colony-counting pipeline
from registry names and verify it counts correctly on a synthetic plate."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from blenny import Pipeline
from blenny.testing import make_synthetic_plate


def test_count_colonies_pipeline_via_from_config(tmp_path: Path) -> None:
    plate = make_synthetic_plate(n_colonies=25, image_size=(512, 512), seed=0)
    image_path = tmp_path / "plate.png"
    Image.fromarray(plate.image).save(image_path)

    csv_path = tmp_path / "results.csv"
    annotated_path = tmp_path / "annotated.png"

    pipe = Pipeline.from_config(
        [
            {"name": "load_image"},
            {"name": "detect_plate", "params": {"crop": True}},
            {"name": "correct_illumination", "params": {"radius": 20}},
            {"name": "threshold_segment", "params": {"roi_mask_key": "plate"}},
            {"name": "measure_colonies"},
            {
                "name": "export_csv",
                "params": {
                    "output_path": str(csv_path),
                    "include_provenance": True,
                },
            },
            {
                "name": "export_annotated",
                "params": {"output_path": str(annotated_path)},
            },
        ]
    )

    out = pipe.run(image_path)

    # Count is in the right ballpark (allow some merge/miss tolerance).
    n_found = out.metadata["colony_count"]
    assert abs(n_found - plate.n_colonies) <= 4, (
        f"Expected ~{plate.n_colonies} colonies, found {n_found}"
    )

    # Per-step provenance was recorded for every module.
    assert [p.step for p in out.provenance] == [
        "load_image",
        "detect_plate",
        "correct_illumination",
        "threshold_segment",
        "measure_colonies",
        "export_csv",
        "export_annotated",
    ]

    # CSV has a header + one row per measurement + the provenance comment.
    csv_text = csv_path.read_text()
    csv_lines = csv_text.strip().splitlines()
    assert csv_lines[0].startswith("# provenance:")
    assert csv_lines[1].startswith("label,")
    assert len(csv_lines) - 2 == n_found  # data rows

    # Annotated image was written and is readable.
    assert annotated_path.exists()
    with Image.open(annotated_path) as im:
        assert im.size[0] > 0 and im.size[1] > 0

    # Plate detection added geometry to metadata.
    assert "plate_center" in out.metadata
    assert "plate_radius" in out.metadata
