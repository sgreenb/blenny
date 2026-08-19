"""Tests for the synthetic plate generator."""

from __future__ import annotations

import numpy as np

from blenny.testing import make_synthetic_plate


def test_generator_is_deterministic_with_seed() -> None:
    a = make_synthetic_plate(n_colonies=20, seed=42)
    b = make_synthetic_plate(n_colonies=20, seed=42)
    assert np.array_equal(a.image, b.image)
    assert a.colony_centers == b.colony_centers


def test_generator_places_requested_colonies_when_room_allows() -> None:
    plate = make_synthetic_plate(n_colonies=15, image_size=(512, 512), seed=0)
    assert plate.n_colonies == 15
    assert len(plate.colony_radii) == 15
    # Output format sanity: uint8 RGB image of the requested size.
    assert plate.image.dtype == np.uint8
    assert plate.image.shape == (512, 512, 3)


def test_generator_reports_actual_count_when_crowding_limits_placement() -> None:
    # Asking for far too many colonies in a small plate; rejection sampling
    # will give up before reaching n_colonies.
    plate = make_synthetic_plate(
        n_colonies=10_000, image_size=(128, 128), colony_radius_range=(6, 8), seed=0
    )
    assert plate.n_colonies < 10_000
    assert plate.n_colonies > 0
