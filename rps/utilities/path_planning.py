import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rps.utilities.occupancy_grid import (
    world_to_grid,
    grid_to_world,
)


GridIndex = Tuple[int, int]  # (ix, iy)


def _neighbors_8(ix: int, iy: int, nx: int, ny: int) -> Sequence[Tuple[int, int, float]]:
    for dx, dy in [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]:
        jx = ix + dx
        jy = iy + dy
        if 0 <= jx < nx and 0 <= jy < ny:
            step_cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
            yield (jx, jy, step_cost)


def _astar_grid(grid: np.ndarray, start: GridIndex, goal: GridIndex) -> Optional[List[GridIndex]]:
    ny, nx = grid.shape

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < nx and 0 <= sy < ny):
        return None
    if not (0 <= gx < nx and 0 <= gy < ny):
        return None
    if grid[sy, sx] != 0 or grid[gy, gx] != 0:
        return None

    def h(ix: int, iy: int) -> float:
        return math.hypot(ix - gx, iy - gy)

    g_cost = np.full((ny, nx), np.inf, dtype=float)
    came_from_x = np.full((ny, nx), -1, dtype=np.int32)
    came_from_y = np.full((ny, nx), -1, dtype=np.int32)

    open_heap: List[Tuple[float, int, int]] = []
    g_cost[sy, sx] = 0.0
    heapq.heappush(open_heap, (h(sx, sy), sx, sy))

    in_open = np.zeros((ny, nx), dtype=bool)
    in_open[sy, sx] = True

    while open_heap:
        f, ix, iy = heapq.heappop(open_heap)
        in_open[iy, ix] = False
        if (ix, iy) == (gx, gy):
            # reconstruct
            path: List[GridIndex] = []
            cx, cy = ix, iy
            while cx != -1 and cy != -1:
                path.append((cx, cy))
                px, py = came_from_x[cy, cx], came_from_y[cy, cx]
                if px == -1:
                    break
                cx, cy = px, py
            path.reverse()
            return path

        for jx, jy, step in _neighbors_8(ix, iy, nx, ny):
            if grid[jy, jx] != 0:
                continue
            tentative_g = g_cost[iy, ix] + step
            if tentative_g < g_cost[jy, jx]:
                g_cost[jy, jx] = tentative_g
                came_from_x[jy, jx] = ix
                came_from_y[jy, jx] = iy
                fscore = tentative_g + h(jx, jy)
                if not in_open[jy, jx]:
                    heapq.heappush(open_heap, (fscore, jx, jy))
                    in_open[jy, jx] = True

    return None


def _bresenham(ix0: int, iy0: int, ix1: int, iy1: int):
    dx = abs(ix1 - ix0)
    dy = -abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx + dy
    x, y = ix0, iy0
    while True:
        yield x, y
        if x == ix1 and y == iy1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _has_line_of_sight(grid: np.ndarray, a: GridIndex, b: GridIndex) -> bool:
    # Sample along the segment with sub-cell resolution to avoid diagonal tunneling near corners
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    steps = int(max(abs(dx), abs(dy)) * 3) + 1  # 3x supersampling
    for s in range(steps + 1):
        t = s / max(1, steps)
        x = ax + t * dx
        y = ay + t * dy
        ix = int(round(x))
        iy = int(round(y))
        if grid[iy, ix] != 0:
            return False
    return True


def _smooth_path(grid: np.ndarray, path: List[GridIndex]) -> List[GridIndex]:
    if len(path) <= 2:
        return path
    smoothed: List[GridIndex] = []
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        # find furthest j with line of sight from i
        while j > i + 1 and not _has_line_of_sight(grid, path[i], path[j]):
            j -= 1
        smoothed.append(path[i])
        i = j
    smoothed.append(path[-1])
    return smoothed


def plan_path_astar(
    inflated_grid: np.ndarray,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    boundaries: List[float],
    resolution: float,
    smooth: bool = True,
) -> Optional[np.ndarray]:
    """Plan a path with A* on an inflated occupancy grid.

    Returns: 2xM numpy array of waypoints in world coordinates, or None if no path.
    """
    sx, sy = start_xy
    gx, gy = goal_xy
    isx, isy = world_to_grid(sx, sy, boundaries, resolution)
    igx, igy = world_to_grid(gx, gy, boundaries, resolution)

    path_idx = _astar_grid(inflated_grid, (isx, isy), (igx, igy))
    if path_idx is None:
        return None
    if smooth:
        path_idx = _smooth_path(inflated_grid, path_idx)

    # Convert to world
    waypoints = np.zeros((2, len(path_idx)), dtype=float)
    for k, (ix, iy) in enumerate(path_idx):
        xw, yw = grid_to_world(ix, iy, boundaries, resolution)
        waypoints[0, k] = xw
        waypoints[1, k] = yw
    return waypoints


