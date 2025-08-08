import math
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

    def create_patch(self) -> patches.Rectangle:
        # Matplotlib Rectangle expects bottom-left corner; convert from center
        x_center, y_center = self.center
        bottom_left = (x_center - self.width / 2.0, y_center - self.height / 2.0)
        return patches.Rectangle(
            xy=bottom_left,
            width=self.width,
            height=self.height,
            angle=self.angle_deg,
            facecolor=self.facecolor,
            edgecolor=self.edgecolor,
            alpha=self.alpha,
            linewidth=1.0,
            zorder=self.zorder,
        )


def is_obstacle(obj) -> bool:
    return isinstance(obj, (CircleObstacle, RectangleObstacle))