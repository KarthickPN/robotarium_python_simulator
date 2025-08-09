import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from rps.utilities.occupancy_grid import build_occupancy_grid, inflate_occupancy_grid, overlay_grid_on_axes
from rps.utilities.path_planning import plan_path_astar
from rps.utilities.obstacles import CircleObstacle, RectangleObstacle
from rps.utilities.obstacles import *

import numpy as np
import time

# Instantiate Robotarium object
N = 5
initial_conditions = np.array(np.asmatrix('1 0.5 -0.5 0 0.28; 0.8 -0.3 -0.75 0.1 0.34; 0 0 0 0 0'))

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)

# Minimal obstacles for planning/visualization
obstacles = [
    CircleObstacle(center=(0.45, 0.0), radius=0.15),
    RectangleObstacle(center=(-0.6, -0.2), width=0.35, height=0.22, angle_deg=10),
]
for obs in obstacles:
    r.axes.add_patch(obs.create_patch())

try:
    r.add_obstacle(CircleObstacle(center=[0.45, 0.0], radius=0.15))
    r.add_obstacle(RectangleObstacle(center=[-0.6, -0.2], width=0.35, height=0.22, angle_deg=10))
except Exception:
    pass
    

# Define goal points by removing orientation from poses
goal_points = generate_initial_conditions(N, width=r.boundaries[2]-2*r.robot_diameter, height = r.boundaries[3]-2*r.robot_diameter, spacing=0.5)

# Build and inflate occupancy grid
resolution = 0.03
raw_grid = build_occupancy_grid(r.boundaries, obstacles, resolution, include_boundary=True)
inflation_radius = r.robot_radius + 0.03
inflated_grid = inflate_occupancy_grid(raw_grid, inflation_radius, resolution)
overlay_grid_on_axes(r.axes, inflated_grid, r.boundaries, cmap='Reds', alpha=0.22)

# Create single integrator position controller and waypoint follower
single_integrator_position_controller = create_si_position_controller()
waypoint_follower = create_waypoint_follower(max_speed=0.15, waypoint_tolerance=0.04)

# Create barrier certificates to avoid collision
#si_barrier_cert = create_single_integrator_barrier_certificate()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

_, uni_to_si_states = create_si_to_uni_mapping()

# Create mapping from single integrator velocity commands to unicycle velocity commands
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

# define x initially
x = r.get_poses()
x_si = uni_to_si_states(x)

# Plan a path from robot 0 to its goal
start_xy = x_si[:, 0]
goal_xy = goal_points[:2, 0]
waypoints = plan_path_astar(inflated_grid, (start_xy[0], start_xy[1]), (goal_xy[0], goal_xy[1]), r.boundaries, resolution, smooth=True)

# Draw planned path if available
if waypoints is not None and waypoints.shape[1] >= 2:
    r.axes.plot(waypoints[0, :], waypoints[1, :], 'b--', linewidth=2, zorder=-1)

# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
goal_marker_size_m = 0.2
robot_marker_size_m = 0.15
marker_size_goal = determine_marker_size(r,goal_marker_size_m)
marker_size_robot = determine_marker_size(r, robot_marker_size_m)
font_size = determine_font_size(r,0.1)
line_width = 5

# Create Goal Point Markers
#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-2)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-2)
for ii in range(goal_points.shape[1])]
robot_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width) 
for ii in range(goal_points.shape[1])]



r.step()

# While the number of robots at the required poses is less
# than N...
while (np.size(at_pose(np.vstack((x_si,x[2,:])), goal_points, rotation_error=100)) != N):

    # Get poses of agents
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    #Update Plot
    # Update Robot Marker Plotted Visualization
    for i in range(x.shape[1]):
        robot_markers[i].set_offsets(x[:2,i].T)
        # This updates the marker sizes if the figure window size is changed. 
        # This should be removed when submitting to the Robotarium.
        robot_markers[i].set_sizes([determine_marker_size(r, robot_marker_size_m)])

    for j in range(goal_points.shape[1]):
        goal_markers[j].set_sizes([determine_marker_size(r, goal_marker_size_m)])

    # If a path exists for robot 0, follow it; otherwise fall back to direct point control
    if waypoints is not None:
        dxi_wp, wp_idx = waypoint_follower(x_si, waypoints, robot_index=0, current_wp_idx=0)
        # If finished waypoints, stop using waypoint follower
        if wp_idx >= waypoints.shape[1]:
            dxi = single_integrator_position_controller(x_si, goal_points[:2][:])
        else:
            dxi = dxi_wp
    else:
        dxi = single_integrator_position_controller(x_si, goal_points[:2][:])

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
