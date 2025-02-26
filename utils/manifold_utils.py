import torch
import numpy as np
from tqdm import trange

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from models.state_mlp import WrappedVF, ProjectToTangent
from flow_matching.solver.solver import Solver
from flow_matching.utils.manifolds import Manifold, Sphere
from flow_matching.solver import RiemannianODESolver
from data.lasa_data import wrap

def sample_normal_source(batch_size:int=1,
                        dim:int=2,
                        horizon:int=1,
                        mean:float=0.0, 
                        std:float=1.0, 
                        manifold:Manifold=None):
    samples = torch.randn((batch_size, 1, dim)) * std + mean ##dangerous. Logic: sample on tangent + project
    samples = samples.repeat(1, horizon, 1)

    
    if manifold is not None:
        samples = wrap(manifold, samples)

    return samples

def sample_context(idx:int,               
                   sample_points:int):
    if idx > 0:
        c = np.random.randint(0, idx)
    else:
        c = 0
    tau_minus_c = torch.tensor((idx - c) / (sample_points+1),
                                dtype=torch.float32).unsqueeze(0)
    return c, tau_minus_c

def infer_model(model, 
                start,
                method:str="midpoint",
                manifold:Manifold=None,
                step_size:float=0.01, 
                sample_points:int=1000, 
                inference_horizon:int=4,
                model_horizon:int=8,
                mean:float=0.0,
                std:float=1.0,
                return_intermediates=False,
                verbose=False):
  start = start.squeeze()
  results = torch.zeros((sample_points + 1,) + start.shape, 
                        dtype=start.dtype, 
                        device=start.device)
  samples = torch.zeros_like(results)

  results[0] = start.clone()
  samples[0] = start.clone()

  step_idx = 0

  for i in trange(sample_points//inference_horizon):
    c, tau_minus_c = sample_context(idx=step_idx, sample_points=sample_points)
    context = torch.cat([results[step_idx], results[c], tau_minus_c]).unsqueeze(0)

    wrapped_vf = WrappedVF(model=model,
                           obs=context)
    wrapped_vf.eval()

    a0 = sample_normal_source(dim=start.shape[-1]-1, 
                              horizon=model_horizon, 
                              manifold=manifold, 
                              mean=mean,
                              std=std)
    
    
    solver = RiemannianODESolver(velocity_model=wrapped_vf, 
                                 manifold=manifold)

    #T = torch.linspace(0, 1, 3)  # sample times

    a_infer = solver.sample(
                    x_init=a0,
                    step_size=step_size,
                    method=method,
                    return_intermediates=return_intermediates,
                    verbose=verbose,
                )
    
    new_idx = step_idx + inference_horizon
    if new_idx < results.shape[0]:
        results[step_idx + 1 : new_idx + 1] = a_infer.squeeze()[:inference_horizon].clone()
        samples[step_idx + 1 : new_idx + 1] = a0.squeeze()[:inference_horizon].clone()

    step_idx = new_idx 
  return results, samples

def step(vf, batch, 
         run_parameters, 
         manifold,
         path,
         device='cpu'):
    
    obs, a1 = batch
    obs, a1 = obs.to(device), a1.to(device)

    batch_size=a1.shape[0]

    a0 = sample_normal_source(batch_size=batch_size,
                                dim=run_parameters['dim']-1, 
                                horizon=run_parameters['horizon_size'], 
                                manifold=manifold, 
                                mean=run_parameters['mean'],
                                std=run_parameters['std'])
    
    t = torch.rand(a0.shape[0]).to(device)
    t_flat = t.unsqueeze(1).repeat(1, a0.shape[1]).view(a0.shape[0] * a0.shape[1])

    path_sample = path.sample(t=t_flat, 
                            x_0=a0.view(a0.shape[0]*a0.shape[1], a0.shape[2]), 
                            x_1=a1.view(a0.shape[0]*a0.shape[1], a0.shape[2]))

    result_vf = vf(obs=obs, x=path_sample.x_t.view(a0.shape), t=t)
    result_vf = manifold.proju(path_sample.x_t.view(a0.shape), result_vf)

    target_vf = path_sample.dx_t.view(a0.shape)

    loss = torch.pow(result_vf - target_vf, 2).mean()
    return loss

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

def evaluate_model_manifold(model,
                            gt_obs:torch.Tensor,
                            method:str="midpoint",
                            manifold:Manifold=None,
                            step_size:float=0.01, 
                            inference_horizon:int=4,
                            model_horizon:int=8,
                            mean:float=0.0,
                            std:float=1.0,
                            return_intermediates=False,
                            verbose=False):

    context = sample_from_gt_obs(gt_obs)

    wrapped_vf = WrappedVF(model=model,
                            obs=context)
    wrapped_vf.eval()

    a0 = sample_normal_source(dim=gt_obs.shape[-1]-1, 
                                horizon=model_horizon, 
                                manifold=manifold, 
                                mean=mean,
                                std=std)

    solver = RiemannianODESolver(velocity_model=wrapped_vf, 
                                 manifold=manifold)

    #T = torch.linspace(0, 1, 3)  # sample times

    a_infer = solver.sample(
                    x_init=a0,
                    step_size=step_size,
                    method=method,
                    return_intermediates=return_intermediates,
                    verbose=verbose,
                )

    return error
   
if __name__ == '__main__':
   start = torch.randn(1,2)
   print(start)
   start = wrap(Sphere(), start)
   print(start)