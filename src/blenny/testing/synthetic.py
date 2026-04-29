"""Generate synthetic plate images for tests and demos.

The generated image looks like a phone photo of a Petri dish:
  - a roughly circular plate region against a darker background,
  - a configurable number of small bright "colonies" placed inside
    the plate without overlapping each other,
  - a multiplicative illumination gradient (so naive global thresholds
    fail and the IlluminationCorrection step has something to do),
  - a sprinkle of Gaussian noise.

The generator is deterministic given a seed, which lets integration
tests assert exact colony counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticPlate:
    """Bundle of the generated image and ground-truth annotations."""

    image: np.ndarray
    """RGB uint8 image of shape (H, W, 3)."""

    plate_center: tuple[int, int]
    """(row, col) center of the plate."""

    plate_radius: int
    """Radius of the plate in pixels."""

    colony_centers: list[tuple[int, int]]
    """(row, col) centers of every colony actually placed."""

    colony_radii: list[int]
    """Radius of each colony in pixels, parallel to ``colony_centers``."""

    @property
    def n_colonies(self) -> int:
        return len(self.colony_centers)


def make_synthetic_plate(
    n_colonies: int = 30,
    image_size: tuple[int, int] = (512, 512),
    colony_radius_range: tuple[int, int] = (5, 9),
    seed: int | None = 0,
    illumination_strength: float = 0.35,
    noise_sigma: float = 4.0,
) -> SyntheticPlate:
    """Render a synthetic Petri-dish photo.

    Colonies are placed via rejection sampling so they don't overlap
    each other or the plate edge. If ``n_colonies`` cannot be placed
    after a generous number of attempts, the function returns however
    many it managed — the actual count is reported in the returned
    :class:`SyntheticPlate`.
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    plate_center = (h // 2, w // 2)
    plate_radius = int(0.45 * min(h, w))

    yy, xx = np.mgrid[0:h, 0:w]
    dist_from_center = np.sqrt((yy - plate_center[0]) ** 2 + (xx - plate_center[1]) ** 2)
    plate_mask = dist_from_center <= plate_radius

    # Base image: dark background, medium-bright plate.
    img = np.full((h, w), 30.0)  # background
    img[plate_mask] = 140.0  # plate interior

    # Place non-overlapping colonies.
    colony_centers: list[tuple[int, int]] = []
    colony_radii: list[int] = []
    max_attempts = n_colonies * 200
    rmin, rmax = colony_radius_range
    safe_radius = plate_radius - rmax - 4
    for _ in range(max_attempts):
        if len(colony_centers) >= n_colonies:
            break
        # Sample uniformly inside the plate (polar to keep it simple).
        rho = float(rng.uniform(0, safe_radius))
        theta = float(rng.uniform(0, 2 * np.pi))
        cy = int(plate_center[0] + rho * np.sin(theta))
        cx = int(plate_center[1] + rho * np.cos(theta))
        cr = int(rng.integers(rmin, rmax + 1))
        # Reject if too close to an existing colony.
        ok = True
        for (oy, ox), orad in zip(colony_centers, colony_radii, strict=True):
            min_sep = cr + orad + 2
            if (cy - oy) ** 2 + (cx - ox) ** 2 < min_sep**2:
                ok = False
                break
        if not ok:
            continue
        colony_centers.append((cy, cx))
        colony_radii.append(cr)

    # Stamp colonies as bright disks with a soft edge.
    for (cy, cx), cr in zip(colony_centers, colony_radii, strict=True):
        local_dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        soft = np.clip(1.0 - (local_dist - cr + 1.5) / 1.5, 0.0, 1.0)
        img += 90.0 * soft

    # Multiplicative illumination gradient (corner-to-corner).
    grad_y = np.linspace(1.0 - illumination_strength, 1.0 + illumination_strength, h)[:, None]
    grad_x = np.linspace(1.0 - illumination_strength, 1.0 + illumination_strength, w)[None, :]
    img *= grad_y * grad_x

    # Gaussian noise.
    img += rng.normal(0.0, noise_sigma, size=img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)

    return SyntheticPlate(
        image=rgb,
        plate_center=plate_center,
        plate_radius=plate_radius,
        colony_centers=colony_centers,
        colony_radii=colony_radii,
    )
