import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pyLasaDataset as lasa
from flow_matching.utils.manifolds import Manifold

def wrap(manifold, samples):
    center = torch.cat([torch.zeros_like(samples), torch.ones_like(samples[..., 0:1])], dim=-1)
    samples = torch.cat([samples, torch.zeros_like(samples[..., 0:1])], dim=-1) / 2

    return manifold.expmap(center, samples)

class StatePyLASADataset(Dataset):
    def __init__(self, dataset_name: str,
                 horizon_size: int = 5,
                 normalize: bool = True,
                 scaling_factor: float = 1.0,
                 downsample: int = 1,
                 manifold : Manifold = None):
        """
        PyTorch Dataset wrapper for LASA with normalization and structured observations.

        Args:
            dataset_name (str): Name of the dataset to load (e.g., "Angle", "Sine").
            horizon_size (int): Number of future steps for action horizon.
            normalize (bool): Whether to normalize x and y values.
        """
        self.dataset = getattr(lasa.DataSet, dataset_name)
        self.horizon_size = horizon_size
        self.downsample = downsample
        self.manifold=manifold

        self.sample_size = self.dataset.demos[0].pos.T.shape[0] // downsample
        self.data = self._concatenate_demos()
        if normalize:
            self._normalize()

        self._scale(scaling_factor)

        self.data = torch.tensor(self.data, dtype=torch.float32)

        if self.manifold:
            self.data = wrap(manifold=manifold, samples=self.data)

    def _concatenate_demos(self):
        """Concatenates all demonstrations into a single sequence, downsampling each demo if needed."""
        data_list = []
        for demo in self.dataset.demos:
            # demo.pos.T should be shape (N, 2)
            demo_data = demo.pos.T
            if self.downsample > 1:
                demo_data = demo_data[::self.downsample]
            data_list.append(demo_data)
        return np.concatenate(data_list, axis=0)

    def _normalize(self):
      """Centers data at zero and normalizes x and y columns to range [-1, 1]."""
      mean_vals = self.data.mean(axis=0)
      centered_data = self.data - mean_vals

      min_vals = centered_data.min(axis=0)
      max_vals = centered_data.max(axis=0)

      self.data = 2 * (centered_data - min_vals) / (max_vals - min_vals) - 1


    def _scale(self, factor):
        """Scales data by factor."""
        self.data = self.data * factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            obs: [o_{τ-1}, o_c, τ - c]  (Concatenated observation vector)
            a: Future action horizon (with zero-padding if necessary)
            t: Normalized time index
        """
        # Current position (x, y)
        o_tau_1 = self.data[idx]

        # Determine which demo the index belongs to
        global_pos = idx // self.sample_size

        # Context observation (sample from the same demo)
        demo_start = global_pos * self.sample_size

        if idx > demo_start:
            c = np.random.randint(demo_start, idx)
        else:
            c = demo_start
        o_c = self.data[c]

        # Distance τ - c
        tau_minus_c = torch.tensor((idx - c) / self.sample_size,
                                   dtype=torch.float32).unsqueeze(0)

        # Observation vector
        obs = torch.cat([o_tau_1, o_c, tau_minus_c])

        # Horizon within the same demo
        demo_end = (global_pos + 1) * self.sample_size
        available_steps = demo_end - (idx + 1)
        steps_to_use = min(self.horizon_size, available_steps)
        a = self.data[idx + 1 : idx + 1 + steps_to_use]
        if steps_to_use < self.horizon_size:
            pad_amount = self.horizon_size - steps_to_use
            a = torch.nn.functional.pad(a, (0, 0, 0, pad_amount), mode="constant")

        return (obs, a)
