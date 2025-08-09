import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import matplotlib.patches as patches


@dataclass
class CircleObstacle:
    center: Tuple[float, float]
    radius: float
    facecolor: str = "#B0B0B0"  # light gray
    edgecolor: str = "#808080"  # gray
    alpha: float = 0.6
    zorder: int = 1

    def create_patch(self) -> patches.Circle:
        return patches.Circle(
            xy=self.center,
            radius=self.radius,
            facecolor=self.facecolor,
            edgecolor=self.edgecolor,
            alpha=self.alpha,
            linewidth=1.0,
            zorder=self.zorder,
        )


@dataclass
class RectangleObstacle:
    center: Tuple[float, float]
    width: float
    height: float
    angle_deg: float = 0.0
    facecolor: str = "#B0B0B0"
    edgecolor: str = "#808080"
    alpha: float = 0.6
    zorder: int = 1

    def create_patch(self) -> patches.Polygon:
        # Build corners by rotating a centered rectangle around its center
        cx, cy = self.center
        w2 = self.width / 2.0
        h2 = self.height / 2.0
        corners = np.array([
            [-w2, -h2],
            [ w2, -h2],
            [ w2,  h2],
            [-w2,  h2],
        ])
        if self.angle_deg != 0.0:
            theta = math.radians(self.angle_deg)
            c = math.cos(theta)
            s = math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            corners = corners @ R.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        return patches.Polygon(
            corners,
            closed=True,
            facecolor=self.facecolor,
            edgecolor=self.edgecolor,
            alpha=self.alpha,
            linewidth=1.0,
            zorder=self.zorder,
        )


def is_obstacle(obj) -> bool:
    return isinstance(obj, (CircleObstacle, RectangleObstacle))