import math
from typing import Iterable, List, Tuple

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt, binary_dilation

try:
    # Optional dependency within this package
    from rps.utilities.obstacles import CircleObstacle, RectangleObstacle
except Exception:
    # Fallback types to avoid hard dependency during import order
    CircleObstacle = Tuple  # type: ignore
    RectangleObstacle = Tuple  # type: ignore


# -----------------------------
# Grid geometry and transforms
# -----------------------------

def compute_grid_size(boundaries: List[float], resolution: float) -> Tuple[int, int]:
    x0, y0, width, height = boundaries
    nx = int(math.ceil(width / resolution))
    ny = int(math.ceil(height / resolution))
    return ny, nx


def world_to_grid(x: float, y: float, boundaries: List[float], resolution: float) -> Tuple[int, int]:
    x0, y0, width, height = boundaries
    ix = int(math.floor((x - x0) / resolution))
    iy = int(math.floor((y - y0) / resolution))
    ny, nx = compute_grid_size(boundaries, resolution)
    ix = max(0, min(nx - 1, ix))
    iy = max(0, min(ny - 1, iy))
    return ix, iy


def grid_to_world(ix: int, iy: int, boundaries: List[float], resolution: float) -> Tuple[float, float]:
    x0, y0, width, height = boundaries
    x = x0 + (ix + 0.5) * resolution
    y = y0 + (iy + 0.5) * resolution
    return x, y


def create_empty_grid(boundaries: List[float], resolution: float) -> np.ndarray:
    ny, nx = compute_grid_size(boundaries, resolution)
    return np.zeros((ny, nx), dtype=np.uint8)


# -----------------------------
# Rasterization helpers
# -----------------------------

def rasterize_boundary(grid: np.ndarray) -> None:
    # One-cell border as occupied (1)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1


def _rasterize_circle(grid: np.ndarray, center: Tuple[float, float], radius: float,
                      boundaries: List[float], resolution: float) -> None:
    ny, nx = grid.shape

    # Bounding box in grid indices
    cx, cy = center
    r = radius
    x_min = cx - r
    x_max = cx + r
    y_min = cy - r
    y_max = cy + r

    ix_min, iy_min = world_to_grid(x_min, y_min, boundaries, resolution)
    ix_max, iy_max = world_to_grid(x_max, y_max, boundaries, resolution)

    ix0 = max(0, min(ix_min, ix_max))
    ix1 = min(nx - 1, max(ix_min, ix_max))
    iy0 = max(0, min(iy_min, iy_max))
    iy1 = min(ny - 1, max(iy_min, iy_max))

    if ix1 < ix0 or iy1 < iy0:
        return

    # Cell centers
    x_coords = np.linspace(0, ix1 - ix0, ix1 - ix0 + 1)
    y_coords = np.linspace(0, iy1 - iy0, iy1 - iy0 + 1)
    X_idx, Y_idx = np.meshgrid(x_coords, y_coords)
    X_world = boundaries[0] + (ix0 + 0.5 + X_idx) * resolution
    Y_world = boundaries[1] + (iy0 + 0.5 + Y_idx) * resolution

    mask = (X_world - cx) ** 2 + (Y_world - cy) ** 2 <= r ** 2
    grid[iy0:iy1 + 1, ix0:ix1 + 1][mask] = 1


def _rectangle_corners(center: Tuple[float, float], width: float, height: float, angle_deg: float) -> np.ndarray:
    cx, cy = center
    w2 = width / 2.0
    h2 = height / 2.0
    # local rectangle corners
    corners = np.array([
        [-w2, -h2],
        [w2, -h2],
        [w2, h2],
        [-w2, h2],
    ])
    if angle_deg != 0.0:
        theta = math.radians(angle_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T
    corners[:, 0] += cx
    corners[:, 1] += cy
    return corners


def _rasterize_rectangle(grid: np.ndarray, center: Tuple[float, float], width: float, height: float, angle_deg: float,
                         boundaries: List[float], resolution: float) -> None:
    ny, nx = grid.shape
    verts = _rectangle_corners(center, width, height, angle_deg)
    poly = Path(verts)

    # Bounding box in world
    x_min = verts[:, 0].min()
    x_max = verts[:, 0].max()
    y_min = verts[:, 1].min()
    y_max = verts[:, 1].max()

    ix_min, iy_min = world_to_grid(x_min, y_min, boundaries, resolution)
    ix_max, iy_max = world_to_grid(x_max, y_max, boundaries, resolution)

    ix0 = max(0, min(ix_min, ix_max))
    ix1 = min(nx - 1, max(ix_min, ix_max))
    iy0 = max(0, min(iy_min, iy_max))
    iy1 = min(ny - 1, max(iy_min, iy_max))

    if ix1 < ix0 or iy1 < iy0:
        return

    # Cell centers in this bbox
    x_coords = boundaries[0] + (np.arange(ix0, ix1 + 1) + 0.5) * resolution
    y_coords = boundaries[1] + (np.arange(iy0, iy1 + 1) + 0.5) * resolution
    Xw, Yw = np.meshgrid(x_coords, y_coords)
    pts = np.column_stack([Xw.ravel(), Yw.ravel()])
    inside = poly.contains_points(pts)
    inside = inside.reshape((iy1 - iy0 + 1, ix1 - ix0 + 1))
    grid[iy0:iy1 + 1, ix0:ix1 + 1][inside] = 1


# -----------------------------
# Public API
# -----------------------------

def build_occupancy_grid(boundaries: List[float], obstacles: Iterable, resolution: float,
                         include_boundary: bool = True) -> np.ndarray:
    """Build a binary occupancy grid from geometric obstacles and boundaries.

    0 = free, 1 = occupied
    """
    grid = create_empty_grid(boundaries, resolution)

    if include_boundary:
        rasterize_boundary(grid)

    for obs in obstacles or []:
        # Circle
        if hasattr(obs, "radius") and hasattr(obs, "center"):
            _rasterize_circle(grid, tuple(obs.center), float(obs.radius), boundaries, resolution)
            continue
        # Rectangle
        if hasattr(obs, "width") and hasattr(obs, "height") and hasattr(obs, "center"):
            angle = float(getattr(obs, "angle_deg", 0.0))
            _rasterize_rectangle(grid, tuple(obs.center), float(obs.width), float(obs.height), angle, boundaries, resolution)
            continue

    return grid


def inflate_occupancy_grid(grid: np.ndarray, inflation_radius_m: float, resolution: float) -> np.ndarray:
    """Inflate (C-space) obstacles by a metric radius using a distance transform.

    Returns a new binary grid (uint8) with inflated obstacles.
    """
    occupied = grid.astype(bool)
    free = ~occupied
    # Distance in cells from free cells to nearest occupied cell
    dist_cells = distance_transform_edt(free)
    thresh = float(inflation_radius_m) / float(resolution)
    inflated = occupied.copy()
    # Any free cell within threshold becomes occupied
    inflated[(free) & (dist_cells <= thresh)] = True
    return inflated.astype(np.uint8)


def overlay_grid_on_axes(ax, grid: np.ndarray, boundaries: List[float], cmap: str = "Reds", alpha: float = 0.25):
    """Optional helper to visualize occupancy grid on a Matplotlib Axes.
    """
    x0, y0, width, height = boundaries
    extent = [x0, x0 + width, y0, y0 + height]
    ax.imshow(np.flipud(grid), extent=extent, origin="lower", cmap=cmap, alpha=alpha, interpolation="none")


