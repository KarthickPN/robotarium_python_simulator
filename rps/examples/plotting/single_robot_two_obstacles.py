import numpy as np

import rps.robotarium as robotarium
from rps.utilities.obstacles import CircleObstacle, RectangleObstacle
from rps.utilities.occupancy_grid import (
    build_occupancy_grid,
    inflate_occupancy_grid,
    overlay_grid_on_axes,
)
from rps.utilities.path_planning import plan_path_astar
from rps.utilities.controllers import create_waypoint_follower
from rps.utilities.transformations import create_si_to_uni_dynamics_with_backwards_motion, create_si_to_uni_mapping


def main():
    # One robot with a fixed initial pose
    N = 1
    initial_conditions = np.array([[ -1.2 ],  # x
                                   [ -0.6 ],  # y
                                   [  0.0 ]]) # theta

    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=True,
        initial_conditions=initial_conditions,
        sim_in_real_time=False,
    )

    # Define two simple obstacles (world coordinates, meters)
    obstacles = [
        CircleObstacle(center=(0.45, 0.0), radius=0.15),
        RectangleObstacle(center=(-0.6, 0.2), width=0.35, height=0.22, angle_deg=10),
    ]

    # Draw obstacles on the simulator axes
    for obs in obstacles:
        r.axes.add_patch(obs.create_patch())

    # Build and visualize occupancy grid (including external boundary)
    resolution = 0.02  # meters per cell (finer to avoid discretization artifacts)
    grid = build_occupancy_grid(r.boundaries, obstacles, resolution, include_boundary=True)
    inflation_radius = r.robot_radius + 0.03
    inflated = inflate_occupancy_grid(grid, inflation_radius, resolution)
    overlay_grid_on_axes(r.axes, inflated, r.boundaries, cmap="Reds", alpha=0.25)

    # Plan from current robot SI position to a fixed goal, then follow waypoints
    # Convert current unicycle pose to single-integrator state
    _, uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

    x = r.get_poses()
    x_si = uni_to_si_states(x)
    start_xy = (float(x_si[0, 0]), float(x_si[1, 0]))

    goal_xy = (1.0, 0.6)  # simple fixed goal inside the arena
    waypoints = plan_path_astar(inflated, start_xy, goal_xy, r.boundaries, resolution, smooth=False)

    if waypoints is not None and waypoints.shape[1] >= 2:
        r.axes.plot(waypoints[0, :], waypoints[1, :], 'b--', linewidth=2, zorder=-1)
    else:
        print("No path found; robot will not move.")

    # Increase follower speed up to platform's max linear velocity (~0.2 m/s)
    follower = create_waypoint_follower(max_speed=0.40, waypoint_tolerance=0.04)
    wp_idx = 0

    # Satisfy Robotarium API: after a get_poses(), call step() before the next get_poses()
    r.step()

    # Iterate and follow the path if available
    iterations = 600
    for _ in range(iterations):
        x = r.get_poses()
        x_si = uni_to_si_states(x)

        if waypoints is not None and wp_idx < waypoints.shape[1]:
            dxi, wp_idx = follower(x_si, waypoints, robot_index=0, current_wp_idx=wp_idx)
        else:
            dxi = np.zeros((2, 1))

        dxu = si_to_uni_dyn(dxi, x)
        r.set_velocities(np.arange(N), dxu)
        r.step()

    r.call_at_scripts_end()


if __name__ == "__main__":
    main()


