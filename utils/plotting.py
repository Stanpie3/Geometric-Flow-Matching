import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display

def plot_flow_on_sphere(results_list, 
                        samples_list=None, 
                        gt_samples=None,
                        label=None, 
                        plot_samples=True,
                        elev=-90,
                        azim=0,
                        dynamic=False,
                        title="Policy",
                        one_canvas=False,
                        line_width_traj=1,
                        line_width_gt=1,
                        line_type='.--',
                        axis_off=False):
    """
    Plots multiple flows from sampled points to results on the 2-sphere.

    Args:
        results_list (list of torch.Tensor or numpy.ndarray): List of tensors/arrays, each of shape (N, 3), representing final points.
        samples_list (list of torch.Tensor or numpy.ndarray): List of tensors/arrays, each of shape (N, 3), representing initial sampled points.
        gt_samples (torch.Tensor or numpy.ndarray): (M, 3) ground truth points on S².
        plot_samples (bool): Whether to plot the initial sample points and connections.
        elev (float): Elevation angle for the 3D view.
        azim (float): Azimuthal angle for the 3D view.
        dynamic (bool): If True, adds interactive sliders for elev and azim.
        one_canvas (bool): If True, plots all trajectories on a single canvas.
    """
    if one_canvas:
        assert not plot_samples, "Cannot plot samples when one_canvas is True."

    def plot_sphere(elev, azim):
        if one_canvas:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            axes = [ax]
        else:
            num_plots = len(results_list)
            fig, axes = plt.subplots(1, num_plots, subplot_kw={'projection': '3d'}, figsize=(4 * num_plots, 4))
            if num_plots == 1:
                axes = [axes]

        gt_samples_np = gt_samples.cpu().numpy() if isinstance(gt_samples, torch.Tensor) else gt_samples

        for idx, (results, samples) in enumerate(zip(results_list, samples_list)):
            results_np = results.cpu().numpy() if isinstance(results, torch.Tensor) else results
            samples_np = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
            ax = axes[0] if one_canvas else axes[idx]

            if not one_canvas and plot_samples:
                for i in range(samples_np.shape[0]):
                    ax.plot([samples_np[i, 0], results_np[i, 0]],
                            [samples_np[i, 1], results_np[i, 1]],
                            [samples_np[i, 2], results_np[i, 2]], 
                            color="gold", alpha=0.3)

                ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], color="green", s=20, label="Base Distribution")

            if one_canvas:
                if idx == 0:
                    ax.plot(results_np[0:, 0], results_np[0:, 1], results_np[0:, 2], line_type, color="blue", 
                        linewidth=line_width_traj, label="Learned Path")
                else:
                    ax.plot(results_np[0:, 0], results_np[0:, 1], results_np[0:, 2], line_type, color="blue", 
                    linewidth=line_width_traj)
            else:
                ax.plot(results_np[:, 0], results_np[:, 1], results_np[:, 2], line_type, color="blue", 
                        linewidth=line_width_traj, label="Learned Path")

            if not one_canvas:
                ax.plot(gt_samples_np[:, 0], gt_samples_np[:, 1], gt_samples_np[:, 2], color="red", linewidth=line_width_gt, label="Ground Truth")

                # Sphere wireframe
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 50)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)

                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(title)
                ax.legend()

        if one_canvas:
            ax = axes[0]
            ax.plot(gt_samples_np[:, 0], gt_samples_np[:, 1], gt_samples_np[:, 2], color="red", linewidth=line_width_gt, label="Ground Truth")

            # Sphere wireframe
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(title)
            ax.legend()

        plt.tight_layout()
        if axis_off:
            ax.set_axis_off()
        plt.show()

    if dynamic:
        elev_slider = widgets.SelectionSlider(
            options=[-90, -45, 0, 45], 
            value=elev, 
            description="Elev:",
            continuous_update=False
        )
        azim_slider = widgets.SelectionSlider(
            options=[0, 90, 180, 270], 
            value=azim, 
            description="Azim:",
            continuous_update=False
        )
        ui = widgets.VBox([elev_slider, azim_slider])
        out = widgets.interactive_output(plot_sphere, {'elev': elev_slider, 'azim': azim_slider})
        display(ui, out)
    else:
        plot_sphere(elev, azim)

def plot_3d_points(points, title="3D Scatter Plot", color="blue", s=20, show_grid=True):
    """
    Plots a set of 3D points interactively.

    Args:
        points (numpy.ndarray or torch.Tensor): Shape (N, 3), representing 3D points.
        title (str): Title of the plot.
        color (str): Color of the points.
        s (int): Size of the scatter points.
        show_grid (bool): Whether to show grid lines.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=s)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    ax.grid(show_grid)

    max_range = (points.max() - points.min()) / 2
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()

def plot_3d_points_interactive(points_list, title="3D Scatter Plot", colors=None, s=20, show_grid=True, same_canvas=True):
    """
    Plots one or more sets of 3D points in a Jupyter Notebook with interactive view angle sliders.

    Args:
        points_list (list of torch.Tensor or np.ndarray): Each tensor/array should be of shape (N, 3).
        title (str): Title of the plot (for same_canvas=True) or plot prefix (for same_canvas=False).
        colors (list of str or str): List of colors or a single color for all sets.
        s (int): Size of scatter points.
        show_grid (bool): Whether to show grid lines.
        same_canvas (bool): If True, plots all point sets on the same canvas. Else, creates separate plots.
    """
    if not isinstance(points_list, list):
        points_list = [points_list]

    # Normalize and convert all to numpy
    points_list = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in points_list]

    # Default color setup
    if colors is None:
        cmap = plt.cm.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(points_list))]
    elif isinstance(colors, str):
        colors = [colors] * len(points_list)

    # Sliders for elevation and azimuth
    elev_slider = widgets.IntSlider(value=30, min=-180, max=180, step=45, description='Elevation:')
    azim_slider = widgets.IntSlider(value=45, min=-180, max=180, step=45, description='Azimuth:')

    def plot(elev, azim):
        if same_canvas:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")

            # Combine all points to determine global limits
            all_points = np.vstack(points_list)
            max_range = (all_points.max() - all_points.min()) / 2
            mid = all_points.mean(axis=0)

            for pts, col in zip(points_list, colors):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=col, s=s)

            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.grid(show_grid)
            ax.view_init(elev=elev, azim=azim)

            plt.show()

        else:
            for i, (pts, col) in enumerate(zip(points_list, colors)):
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=col, s=s)

                max_range = (pts.max() - pts.min()) / 2
                mid = pts.mean(axis=0)
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

                ax.set_title(f"{title} #{i+1}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.grid(show_grid)
                ax.view_init(elev=elev, azim=azim)

                plt.show()

    out = widgets.interactive_output(plot, {'elev': elev_slider, 'azim': azim_slider})
    display(widgets.VBox([elev_slider, azim_slider]), out)


def plot_flow(results_list, 
              samples_list, 
              gt_samples, 
              plot_samples=True):
    """
    Plots multiple flows from sampled points to results in 2D space, side by side.

    Args:
        results_list (list of torch.Tensor or numpy.ndarray): List of tensors/arrays, each of shape (N, 2), representing final points.
        samples_list (list of torch.Tensor or numpy.ndarray): List of tensors/arrays, each of shape (N, 2), representing initial sampled points.
        plot_samples (bool): Whether to plot the initial sample points and connections.
    """
    num_plots = len(results_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    gt_samples = gt_samples.cpu().numpy() if isinstance(gt_samples, torch.Tensor) else gt_samples

    lim_min, lim_max = gt_samples.min(axis=0), gt_samples.max(axis=0)
    if num_plots == 1:
        axes = [axes]
    
    for ax, results, samples in zip(axes, results_list, samples_list):
        results = results.cpu().numpy() if isinstance(results, torch.Tensor) else results
        samples = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples

        if plot_samples:
            for i in range(samples.shape[0]):
                ax.plot([samples[i, 0], results[i, 0]], [samples[i, 1], results[i, 1]], color='gold', alpha=0.1)

            ax.scatter(samples[:, 0], samples[:, 1], color='green', s=20, label='Base Distribution')
            ax.scatter(results[:, 0], results[:, 1], color='blue', s=20, label='Target Points')

        ax.plot(results[:, 0], results[:, 1], color='blue', linewidth=2, label='Learned Path')
        ax.plot(gt_samples[:, 0], gt_samples[:, 1], color='red', linewidth=1, label='Ground truth')

        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(lim_min[0]-1, lim_max[0]+1)
        ax.set_ylim(lim_min[1]-1, lim_max[1]+1)
        ax.set_title("Flow from Base to Target")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_error_for_each_point(obs_coordinate, error):
    """
    Visualizes a trajectory where each point is colored based on its error value.

    Args:
        obs_coordinate (torch.Tensor or np.ndarray): Array of shape (N, 2) containing (x, y) coordinates.
        error (torch.Tensor or np.ndarray): Array of shape (N,) containing error values for each point.
    """

    obs_coordinate = obs_coordinate.cpu().numpy() if isinstance(obs_coordinate, torch.Tensor) else obs_coordinate
    error = error.cpu().numpy() if isinstance(error, torch.Tensor) else error

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(obs_coordinate[:, 0], obs_coordinate[:, 1], 
                          c=error, cmap="plasma", s=20, edgecolor="k", alpha=0.75)

    cbar = plt.colorbar(scatter)
    cbar.set_label("MSE between target horizon and approximated", fontsize=12)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Trajectory Error")
    plt.grid(True)
    
    plt.show()