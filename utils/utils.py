import torch
import numpy as np
from tqdm import trange

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from utils.schemes import Euler, RK4
from models.state_mlp import WrappedVF





def sample_normal_source_euc(batch_size:int=1,
                            dim:int=2,
                            horizon:int=1,
                            mean:float=0.0, 
                            std:float=1.0):
    samples = torch.randn((batch_size, 1, dim)) * std + mean
    samples = samples.repeat(1, horizon, 1)
    return samples

def step(vf, batch, 
         run_parameters, 
         manifold,
         path,
         device='cpu'):
    obs, a1, label = batch
    obs, a1, label = obs.to(device), a1.to(device), label.to(device)

    label = label.view(-1)
    obs = obs.view(-1, obs.shape[-1])
    a1 = a1.view(-1, a1.shape[-2], a1.shape[-1])

    batch_size=a1.shape[0]

    a0 = sample_normal_source_euc(batch_size=batch_size,
                                dim=2,
                                horizon=run_parameters['data']['horizon_size'], 
                                mean=run_parameters['data']['mean'],
                                std=run_parameters['data']['std'])
    
    t = torch.rand(a0.shape[0]).to(device)
    t_flat = t.unsqueeze(1).repeat(1, a0.shape[1]).view(a0.shape[0] * a0.shape[1])

    path_sample = path.sample(t=t_flat, 
                            x_0=a0.view(a0.shape[0]*a0.shape[1], a0.shape[2]), 
                            x_1=a1.view(a0.shape[0]*a0.shape[1], a0.shape[2]))

    result_vf = vf(obs=obs, label=label, x=path_sample.x_t.view(a0.shape), t=t)
    result_vf = manifold.proju(path_sample.x_t.view(a0.shape), result_vf)

    target_vf = path_sample.dx_t.view(a0.shape)

    loss = torch.pow(result_vf - target_vf, 2).mean()
    return loss

def infer_model(model, 
                start, 
                scheme=Euler, 
                num_steps=100, 
                sample_points=1000, 
                inference_horizon=4,
                model_horizon=8,
                action_dim=2):
  start = start.squeeze()
  results = torch.zeros((sample_points + 1,) + start.shape, 
                        dtype=start.dtype, 
                        device=start.device)
  samples = torch.zeros_like(results)

  results[0] = start.clone()
  samples[0] = start.clone()

  step_idx = 0

  for i in trange(sample_points//inference_horizon):
    idx = step_idx
    o_tau_1 = results[idx]
    if idx > 0:
        c = np.random.randint(0, idx)
    else:
        c = 0
    o_c = results[c]
    tau_minus_c = torch.tensor((idx - c) / (sample_points+1),
                                dtype=torch.float32).unsqueeze(0)
    context = torch.cat([o_tau_1, o_c, tau_minus_c]).unsqueeze(0)

    wrapped_vf = WrappedVF(model, context)
    wrapped_vf.eval()

    a0 = torch.randn(1, 1, action_dim)
    a0 = a0.repeat(1, model_horizon, 1)

    at = scheme(wrapped_vf, a0, num_steps=num_steps)
    new_idx = step_idx + inference_horizon
    if new_idx < results.shape[0]:
        results[step_idx + 1 : new_idx + 1] = at.squeeze()[:inference_horizon].clone()
        samples[step_idx + 1 : new_idx + 1] = a0.squeeze()[:inference_horizon].clone()

    step_idx = new_idx 
  return results, samples

def sample_from_gt_obs(obs):
    """
    Samples values from gt_obs such that for the k-th sample, the sampled index i is not larger than k.

    Args:
        gt_obs (torch.Tensor): Tensor of shape (N, D) containing ground truth observations.

    Returns:
        torch.Tensor: A tensor of shape (N, D + D + 1), containing:
                      - gt_obs[k] (selected based on k)
                      - gt_obs[i] (randomly sampled i ≤ k)
                      - (k - i) / k (normalized difference)
    """
    batch_size = obs.shape[0]

    indices = np.arange(batch_size)
    sampled_indices = np.array([np.random.randint(0, k+1) for k in indices])  # i ≤ k

    differences = (indices - sampled_indices) / (indices + 1)

    gt_obs_k = obs[indices]
    gt_obs_i = obs[sampled_indices]

    result = torch.cat((gt_obs_k, gt_obs_i, torch.tensor(differences, dtype=torch.float32).unsqueeze(1)), dim=1)

    return result

def evaluate_model(model, 
                   gt_obs,
                   horizon_obs,
                   scheme=Euler, 
                   num_steps=100,
                   model_horizon=8,
                   action_dim=2):
  
  error = torch.zeros(gt_obs.shape[0])

  context = sample_from_gt_obs(gt_obs)

  wrapped_vf = WrappedVF(model, context)
  wrapped_vf.eval()

  a0 = torch.randn(gt_obs.shape[0], 1, action_dim)
  a0 = a0.repeat(1, model_horizon, 1)

  at = scheme(wrapped_vf, a0, num_steps=num_steps)

  error = torch.sqrt(((at - horizon_obs).clone()**2).sum(axis=(1,2)))
  return error
   
if __name__ == '__main__':
   print("Run as script")