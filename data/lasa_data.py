import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pyLasaDataset as lasa
from flow_matching.utils.manifolds import Manifold

def wrap(manifold, samples, dim_from, dim_to):
    """
    Projects points from R^(dim_from) to an dim_to-dimensional manifold.

    Args:
        manifold: The manifold object with an `expmap` function.
        samples: Tensor of shape (..., dim_from), representing points in R^(dim_from).
        dim_from: The original dimension of the input space (n-k).
        dim_to: The target dimension of the output space (n).

    Returns:
        Tensor of shape (..., dim_to), mapped onto the manifold.
    """
    k = dim_to - dim_from

    if k <= 0:
        raise ValueError("dim_to must be greater than dim_from")

    center = torch.cat([torch.zeros_like(samples), torch.ones_like(samples[..., :k])], dim=-1)

    samples = torch.cat([samples, torch.zeros_like(samples[..., :k])], dim=-1) / 2

    return manifold.expmap(center, samples)

class StatePyLASADataset(Dataset):
    def __init__(self, dataset_names: list,
                 train: list,
                 horizon_size: int = 5,
                 normalize: bool = True,
                 scaling_factor: float = 1.0,
                 downsample: int = 1,
                 manifold: Manifold = None,
                 dim_from: int = 2,
                 dim_to: int = 3,
                 start_points: dict = None):
        """
        PyTorch Dataset wrapper for multiple LASA datasets with normalization and structured observations.

        Args:
            dataset_names (list): List of dataset names to load.
            horizon_size (int): Number of future steps for action horizon.
            normalize (bool): Whether to normalize data.
            scaling_factor (float): Scaling factor for data.
            downsample (int): Downsampling factor.
            manifold (Manifold, optional): If provided, maps data onto the given manifold.
            dim_from (int): Dimension of the input space.
            dim_to (int): Dimension of the output space.
        """
        self.horizon_size = horizon_size
        self.downsample = downsample
        self.manifold = manifold
        self.dim_from = dim_from
        self.dim_to = dim_to
        self.train = train
        self.start_points = start_points if start_points else {}

        self.demos = []
        self.horizons = []
        self.labels = []
        self.class_mapping = {name: i for i, name in enumerate(dataset_names)}

        for dataset_name in dataset_names:

            label = self.class_mapping[dataset_name]
            dataset = getattr(lasa.DataSet, dataset_name)

            for demo in dataset.demos:
                demo_data = demo.pos.T
                if self.downsample > 1:
                    demo_data = demo_data[::self.downsample]
                if normalize:
                    demo_data = self._normalize(demo_data, dataset_name)
                demo_data = demo_data * scaling_factor
                demo_data = torch.tensor(demo_data, dtype=torch.float32)
                if self.manifold:
                    demo_data = wrap(manifold=self.manifold, 
                                    samples=demo_data,
                                    dim_from=self.dim_from, 
                                    dim_to=self.dim_to)
                
                self.demos.append(demo_data)
                self.horizons.append(self._get_horizons(demo_data))
                self.labels.append(torch.tensor(label, dtype=torch.long).repeat(demo_data.shape[:-1]))

    def _get_horizons(self, demo):
        N, dim = demo.shape
        padded_demo = torch.nn.functional.pad(demo, (0, 0, 0, self.horizon_size), value=0)
        strides = (padded_demo.stride(0), padded_demo.stride(0), padded_demo.stride(1))
        return torch.as_strided(padded_demo, size=(N, self.horizon_size, dim), stride=strides)

    def _normalize(self, data, dataset_name):
        mean_vals = data.mean(axis=0)
        centered_data = data - mean_vals
        min_vals, max_vals = centered_data.min(axis=0), centered_data.max(axis=0)
        eps = 1e-8  

        normalized_data = 2 * (centered_data - min_vals) / (max_vals - min_vals + eps) - 1
        if dataset_name in self.start_points:
            offset = self.start_points[dataset_name] - normalized_data[0]
            normalized_data += offset

        return normalized_data
    
    def _sample_context(self, demo):
        N = demo.shape[0]

        indices = np.arange(N)
        sampled_indices = np.array([np.random.randint(0, k+1) for k in indices])  # i ≤ k

        differences = (indices - sampled_indices) / (indices + 1)

        gt_obs_k = demo[indices]
        gt_obs_i = demo[sampled_indices]

        result = torch.cat((gt_obs_k, gt_obs_i, torch.tensor(differences, dtype=torch.float32).unsqueeze(1)), dim=1)

        return result
    
    def get_label_maping(self):
        return self.class_mapping

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        """
        Returns:
            obs: Concatenated observation vector.
            a: Future action horizon (zero-padded if needed).
            t: Normalized time index.
            label: Class label of the sample.
        """    
        demo = self.demos[idx]
        label = self.labels[idx]
        if idx in self.train:
            index_permutation = torch.randperm(demo.shape[0])
        else:
            index_permutation = torch.arange(demo.shape[0])
        a = self.horizons[idx][index_permutation]
        obs = self._sample_context(demo)[index_permutation]
        return obs, a, label

if __name__ == '__main__':
    LASA_datasets = ["Sine", "Angle"]
    sine_data = StatePyLASADataset(LASA_datasets,
                               train=list(range(6)),
                               horizon_size=8,
                               scaling_factor=2.0,
                               downsample = 5,
                               manifold=None,
                               dim_to=3,
                               normalize=True)
    obs, a, label = next(iter(sine_data))
    print(obs.shape, a.shape, label)
    print(a[-10:])