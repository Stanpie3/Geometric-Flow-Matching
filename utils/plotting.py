import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_flow(results_list, samples_list, gt_samples, plot_samples=True):
    """
    Plots multiple flows from sampled points to results in 2D space, side by side.

    Args:
        results_list (list of torch.Tensor or numpy.ndarray): List of tensors/arrays, each of shape (N, 2), representing final points.
        samples_list (list of torch.Tensor or numpy.ndarray): List of tensors/arrays, each of shape (N, 2), representing initial sampled points.
        plot_samples (bool): Whether to plot the initial sample points and connections.
    """
    num_plots = len(results_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable
    
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

    # Convert to numpy if tensors are given
    obs_coordinate = obs_coordinate.cpu().numpy() if isinstance(obs_coordinate, torch.Tensor) else obs_coordinate
    error = error.cpu().numpy() if isinstance(error, torch.Tensor) else error

    # Create scatter plot with magma colormap
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(obs_coordinate[:, 0], obs_coordinate[:, 1], 
                          c=error, cmap="plasma", s=20, edgecolor="k", alpha=0.75)

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("MSE between target horizon and approximated", fontsize=12)

    # Labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Trajectory Error")
    plt.grid(True)
    
    plt.show()