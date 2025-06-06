import torch
import numpy as np
from tqdm import trange, tqdm

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from models.state_mlp import WrappedVF, ProjectToTangent
from flow_matching.solver.solver import Solver
from flow_matching.utils.manifolds import Manifold, Sphere, Euclidean
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from data.lasa_data import wrap
from .so3 import *
from .pytorch3d import quaternion_to_matrix, matrix_to_quaternion

def quat_obs_to_mat_obs(obs, base='R9'):
    if base == 'R9':
        b, _ = obs.shape
        obs1_mat = BM_2_R9(quaternion_to_matrix(obs[:,:4]))
        obsk_mat = BM_2_R9(quaternion_to_matrix(obs[:,4:8]))
        dist = obs[:,8]
        obs_mat = torch.cat((obs1_mat, obsk_mat, dist.unsqueeze(1)), dim=1)
    elif base == 'R6':
        b, _ = obs.shape
        obs1_mat = BM_2_R6(quaternion_to_matrix(obs[:,:4]))
        obsk_mat = BM_2_R6(quaternion_to_matrix(obs[:,4:8]))
        dist = obs[:,8]
        obs_mat = torch.cat((obs1_mat, obsk_mat, dist.unsqueeze(1)), dim=1)
    else:
        raise ValueError("Unknown domain \'" + base + "\'")

    return obs_mat

def sample_normal_source(batch_size:int=1,
                        dim:int=2,
                        horizon:int=1,
                        mean:float=0.0, 
                        std:float=1.0, 
                        manifold:Manifold=None,
                        dim_to = 3):
    samples = torch.randn((batch_size, 1, dim)) * std + mean
    samples = samples.repeat(1, horizon, 1)

    
    if manifold is not None:
        samples = wrap(manifold, samples, dim, dim_to)

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
                dim_manifold,
                label,
                method:str="midpoint",
                manifold:Manifold=None,
                step_size:float=0.01, 
                sample_points:int=1000, 
                inference_horizon:int=4,
                model_horizon:int=8,
                mean:float=0.0,
                std:float=1.0,
                return_intermediates=False,
                project_infered=False,
                verbose=False,
                base='S'):
  start = start.squeeze()
  results = torch.zeros((sample_points + 1,) + start.shape, 
                        dtype=start.dtype, 
                        device=start.device)
  samples = torch.zeros_like(results)

  results[0] = start.clone()
  samples[0] = start.clone()
#   print(results.shape)

  step_idx = 0
  paths = []
  if return_intermediates:
      T = torch.linspace(0, 1, int(1/step_size))
  else:
      T = torch.tensor([0.0,1.0])
  for i in tqdm(range(sample_points//inference_horizon), desc="Sampling trajectory", leave=False):
    c, tau_minus_c = sample_context(idx=step_idx, sample_points=sample_points)
    context = torch.cat([results[step_idx], results[c], tau_minus_c]).unsqueeze(0)
    # print(context.shape)
    wrapped_vf = WrappedVF(model=ProjectToTangent(vecfield=model, 
                                                  manifold=manifold),
                           obs=context,
                           label=label)
    wrapped_vf.eval()

    if base == 'S':
        a0 = sample_normal_source(dim=dim_manifold-1,# start.shape[-1]-1, ##################################### 2
                                horizon=model_horizon, 
                                manifold=manifold, 
                                mean=mean,
                                std=std,
                                dim_to=dim_manifold)
    elif base == 'R9':
        a0 = sample_normal_SO3(batch_size=1,
                               horizon=model_horizon, 
                               mean=None,
                               std=torch.tensor(std))
        a0 = BM_2_R9(a0)
    elif base == 'R6':
        a0 = sample_normal_SO3(batch_size=1,
                               horizon=model_horizon, 
                               mean=None,
                               std=torch.tensor(std))
        a0 = BM_2_R6(a0)
    
    # print(a0.shape)
    
    solver = RiemannianODESolver(velocity_model=wrapped_vf, 
                                 manifold=manifold)
    path_sample = solver.sample(
                    x_init=a0,
                    step_size=step_size,
                    method=method,
                    return_intermediates=return_intermediates,
                    verbose=verbose,
                    time_grid = T 
                )
    if return_intermediates:
        a_infer = path_sample[-1]
        paths.append(path_sample)
    else:
        a_infer = path_sample
    new_idx = step_idx + inference_horizon
    if new_idx < results.shape[0]:
        infered_data = a_infer.squeeze()[:inference_horizon].clone()
        sampled_data = a0.squeeze()[:inference_horizon].clone()
        if base == 'R6':
            infered_data = zhou_6d_to_so3(infered_data)
            sampled_data = zhou_6d_to_so3(sampled_data)
        elif base == 'R9':
            infered_data = procrustes_to_so3(infered_data)
            sampled_data = procrustes_to_so3(sampled_data)
        if project_infered:
            infered_data = Sphere().projx(infered_data)
        results[step_idx + 1 : new_idx + 1] = infered_data
        samples[step_idx + 1 : new_idx + 1] = sampled_data

    step_idx = new_idx 
  return results, samples, paths

def infer_model_tangent(model, 
                        start,
                        dim_manifold,
                        label,
                        method:str="midpoint",
                        manifold:Manifold=Sphere(),
                        tangent_manifold:Manifold=Euclidean(),
                        step_size:float=0.01, 
                        sample_points:int=1000, 
                        inference_horizon:int=4,
                        model_horizon:int=8,
                        mean:float=0.0,
                        std:float=1.0,
                        verbose=False):
  start_point_sphere = start.squeeze()
  start = manifold.logmap(start_point_sphere, start_point_sphere)
  results = torch.zeros((sample_points + 1,) + start.shape, 
                        dtype=start.dtype, 
                        device=start.device)
  samples = torch.zeros_like(results)

  results[0] = start.clone()
  samples[0] = start.clone()

  step_idx = 0
  T = torch.tensor([0.0,1.0])
  for i in tqdm(range(sample_points//inference_horizon), desc="Sampling trajectory", leave=False):
    c, tau_minus_c = sample_context(idx=step_idx, sample_points=sample_points)
    context = torch.cat([results[step_idx], results[c], tau_minus_c]).unsqueeze(0)
    
    wrapped_vf = WrappedVF(model=ProjectToTangent(vecfield=model, 
                                                  manifold=tangent_manifold),
                                                #   tangent_point=start_point_sphere),
                           obs=context,
                           label=label)
    wrapped_vf.eval()

    a0 = sample_normal_source(dim=dim_manifold,# start.shape[-1]-1, ####################### 2
                              horizon=model_horizon, 
                              manifold=None,  ############################# manifold
                              mean=mean,
                              std=std,
                              dim_to=dim_manifold)
    
    a0_tang = manifold.proju(start_point_sphere, a0) #a0#manifold.logmap(start_point_sphere, a0) ######################################3333
    
    solver = RiemannianODESolver(velocity_model=wrapped_vf, 
                                 manifold=tangent_manifold)
    a_infer = solver.sample(
                    x_init=a0_tang,
                    step_size=step_size,
                    method=method,
                    return_intermediates=False,
                    verbose=verbose,
                    time_grid = T 
                )
    new_idx = step_idx + inference_horizon
    if new_idx < results.shape[0]:
        results[step_idx + 1 : new_idx + 1] = a_infer.squeeze()[:inference_horizon].clone()  # manifold.proju(start_point_sphere, a_infer.squeeze()[:inference_horizon].clone()) 
        samples[step_idx + 1 : new_idx + 1] = a0.squeeze()[:inference_horizon].clone()

    step_idx = new_idx 
  return results, samples

def step(vf, batch, 
         run_parameters, 
         manifold,
         path,
         device='cpu',
         base="S"):
    
    obs, a1, label = batch
    obs, a1, label = obs.to(device), a1.to(device), label.to(device)

    label = label.view(-1)
    obs = obs.view(-1, obs.shape[-1])
    a1 = a1.view(-1, a1.shape[-2], a1.shape[-1])

    batch_size=a1.shape[0]

    if base == 'S':
        a0 = sample_normal_source(batch_size=batch_size,
                                    dim=run_parameters['data']['dim']-1, ######################################## 2
                                    horizon=run_parameters['data']['horizon_size'], 
                                    manifold=manifold, 
                                    mean=run_parameters['data']['mean'],
                                    std=run_parameters['data']['std'],
                                    dim_to=run_parameters['data']['dim'])
    elif base == 'R9':
        assert(obs.shape[-1] == 9), "Input does not represent quaternion"
        a0 = sample_normal_SO3(batch_size=batch_size, 
                               horizon=run_parameters['data']['horizon_size'], 
                               mean=None,
                               std=torch.tensor(run_parameters['data']['std']))
        a0 = BM_2_R9(a0)
        a1 = BM_2_R9(quaternion_to_matrix(a1))
        obs = quat_obs_to_mat_obs(obs, base=base)
    elif base == 'R6':
        assert(obs.shape[-1] == 9), "Input does not represent quaternion"
        a0 = sample_normal_SO3(batch_size=batch_size, 
                               horizon=run_parameters['data']['horizon_size'], 
                               mean=None,
                               std=torch.tensor(run_parameters['data']['std']))
        a0 = BM_2_R6(a0)
        a1 = BM_2_R6(quaternion_to_matrix(a1))
        obs = quat_obs_to_mat_obs(obs, base=base)
    else:
        raise ValueError("Unknown domain \'" + base + "\'")
    
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

def step_euc_sphere(vf, 
                    batch, 
                    run_parameters, 
                    path,
                    device='cpu'):
    
    obs, a1, label = batch
    obs, a1, label = obs.to(device), a1.to(device), label.to(device)

    label = label.view(-1)
    obs = obs.view(-1, obs.shape[-1])
    a1 = a1.view(-1, a1.shape[-2], a1.shape[-1])

    batch_size=a1.shape[0]

    a0 = sample_normal_source(batch_size=batch_size,
                                dim=run_parameters['data']['dim'], 
                                horizon=run_parameters['data']['horizon_size'], 
                                manifold=None, 
                                mean=run_parameters['data']['mean'],
                                std=run_parameters['data']['std'],
                                dim_to=run_parameters['data']['dim'])
    
    t = torch.rand(a0.shape[0]).to(device)
    t_flat = t.unsqueeze(1).repeat(1, a0.shape[1]).view(a0.shape[0] * a0.shape[1])

    path_sample = path.sample(t=t_flat, 
                            x_0=a0.view(a0.shape[0]*a0.shape[1], a0.shape[2]), 
                            x_1=a1.view(a0.shape[0]*a0.shape[1], a0.shape[2]))

    result_vf = vf(obs=obs, label=label, x=path_sample.x_t.view(a0.shape), t=t)
    # result_vf = manifold.proju(path_sample.x_t.view(a0.shape), result_vf)

    target_vf = path_sample.dx_t.view(a0.shape)

    loss = torch.pow(result_vf - target_vf, 2).mean()
    return loss

def step_tangent(vf, 
                 batch,
                 start_point,
                 run_parameters, 
                 manifold,
                 result_vf_manifold,
                 path,
                 device='cpu'):
    
    obs, a1, label = batch
    obs, a1, label = obs.to(device), a1.to(device), label.to(device)

    obs_tang = torch.zeros_like(obs)
    obs_tang[:, :, :3] = manifold.logmap(start_point, obs[:, :, :3])
    obs_tang[:, :, 3:6] = manifold.logmap(start_point, obs[:, :, 3:6])
    obs_tang[:, :, 6:] = obs[:,:,6:]
    a1_tang = manifold.logmap(start_point, a1)

    # print(obs.shape, a1.shape)
    # print(obs_tang.shape, a1_tang.shape)

    label = label.view(-1)
    obs_tang = obs_tang.view(-1, obs.shape[-1])
    a1_tang = a1_tang.view(-1, a1.shape[-2], a1.shape[-1])

    batch_size=a1_tang.shape[0]

    a0 = sample_normal_source(batch_size=batch_size,
                                dim=run_parameters['data']['dim'], ############################### 2
                                horizon=run_parameters['data']['horizon_size'], 
                                manifold=None, ############## manifold
                                mean=run_parameters['data']['mean'],
                                std=run_parameters['data']['std'],
                                dim_to=run_parameters['data']['dim'])
    
    a0_tang = manifold.proju(start_point, a0) #manifold.logmap(start_point, a0) ####################################################################3
    # print(a0.shape, a0_tang.shape)
    
    t = torch.rand(a0.shape[0]).to(device)
    t_flat = t.unsqueeze(1).repeat(1, a0.shape[1]).view(a0.shape[0] * a0.shape[1])

    path_sample = path.sample(t=t_flat, 
                            x_0=a0_tang.view(a0.shape[0]*a0.shape[1], a0.shape[2]), 
                            x_1=a1_tang.view(a0.shape[0]*a0.shape[1], a0.shape[2]))

    result_vf = vf(obs=obs_tang, label=label, x=path_sample.x_t.view(a0.shape), t=t)
    # result_vf = manifold.proju(start_point, result_vf)

    target_vf = path_sample.dx_t.view(a0.shape)

    loss = torch.pow(result_vf - target_vf, 2).mean()
    return loss

def geodesic_mse(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute geodesic MSE between two [n, dim] tensors on the sphere S^{dim-1}.
    
    Args:
        X, Y: torch.Tensor of shape [n, dim], each row on the unit sphere
        eps: small constant to prevent numerical issues with arccos

    Returns:
        torch scalar: mean squared geodesic distance (in radians^2)
    """
    assert X.shape == Y.shape, "Input tensors must have the same shape"

    dot_products = torch.sum(X * Y, dim=1)
    dot_products = torch.clamp(dot_products, -1.0 + eps, 1.0 - eps)  # for stability
    angles = torch.acos(dot_products)
    mse = torch.mean(angles ** 2)
    return mse



def run_inference(manifold, model, run_parameters, class_labels, gt_obs, step_size=None, inference_type="Native"):
    output = dict()
    if step_size is None:
        step_size = run_parameters['train']['inf_run_step']
    for label_name in list(class_labels.keys()):
        tmp = dict()
        tmp['results'] = []
        tmp['samples'] = []
        tmp['paths'] = []
        for _ in range(run_parameters['train']['inf_runs_num']):
            if inference_type=="Tangent":
                label = torch.tensor(class_labels[label_name],dtype=torch.long).unsqueeze(0)
                res, samp = infer_model_tangent(
                                        model=model, 
                                        start=gt_obs[class_labels[label_name],0,:run_parameters['data']['dim']], 
                                        label=label,
                                        dim_manifold=run_parameters['data']['dim'],
                                        model_horizon=run_parameters['data']['horizon_size'],
                                        inference_horizon=run_parameters['data']['inference_horizon'],
                                        sample_points=run_parameters['data']['sample_points'],
                                        mean=run_parameters['data']['mean'],
                                        std=run_parameters['data']['std'],
                                        step_size=step_size
                                    )
                tmp['results'].append(res)
                tmp['samples'].append(samp)
            elif inference_type=="Native":
                label = torch.tensor(class_labels[label_name],dtype=torch.long).unsqueeze(0)
                res, samp, paths = infer_model(
                                        model=model, 
                                        start=gt_obs[class_labels[label_name],0,:run_parameters['data']['dim']], 
                                        manifold=manifold,
                                        label=label,
                                        dim_manifold=run_parameters['data']['dim'],
                                        model_horizon=run_parameters['data']['horizon_size'],
                                        inference_horizon=run_parameters['data']['inference_horizon'],
                                        sample_points=run_parameters['data']['sample_points'],
                                        mean=run_parameters['data']['mean'],
                                        std=run_parameters['data']['std'],
                                        step_size=step_size
                                    )
                tmp['results'].append(res)
                tmp['samples'].append(samp)
                tmp['paths'].append(paths)
            elif inference_type=="Euclidean sphere":
                label = torch.tensor(class_labels[label_name],dtype=torch.long).unsqueeze(0)
                res, samp, paths = infer_model(
                                        model=model, 
                                        start=gt_obs[class_labels[label_name],0,:run_parameters['data']['dim']], 
                                        manifold=manifold,
                                        label=label,
                                        dim_manifold=run_parameters['data']['dim'],
                                        model_horizon=run_parameters['data']['horizon_size'],
                                        inference_horizon=run_parameters['data']['inference_horizon'],
                                        sample_points=run_parameters['data']['sample_points'],
                                        mean=run_parameters['data']['mean'],
                                        std=run_parameters['data']['std'],
                                        project_infered=True,
                                        step_size=step_size
                                    )
                tmp['results'].append(res)
                tmp['samples'].append(samp)
                tmp['paths'].append(paths)
            else:
                raise ValueError("Not implemented inference type \'" + inference_type + "\'")
        output[label_name] = tmp
    return output

def curve_geodesic_MSE(manifold, x_curve, y_curve):
    assert(x_curve.shape == y_curve.shape)
    dist = np.zeros(x_curve.shape[0])
    for i in range(x_curve.shape[0]):
        dist[i]=manifold.dist(x_curve[i], y_curve[i])
    return (dist[~np.isnan(dist)]**2).mean()

def sample_uniform_geodesic_path(manifold, start, finish, num_points):
    assert(start.shape == finish.shape)
    assert(num_points >= 2)
    path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)
    t = torch.linspace(0.0, 1.0, num_points)
    start = start.unsqueeze(0)
    start = start.repeat(num_points, 1)
    finish = finish.unsqueeze(0).repeat(num_points, 1)
    path_sample = path.sample(t=t, x_0=start, x_1=finish)
    return path_sample.x_t.view(num_points, -1)

   
if __name__ == '__main__':
    start = torch.randn(1,2)
    finish = torch.rand(1,2)
    print(start, finish)
    start = wrap(Sphere(), start, dim_from=2, dim_to=3)
    finish = wrap(Sphere(), finish, dim_from=2, dim_to=3)
    print(start, finish)
    path1 = sample_uniform_geodesic_path(Sphere(), start[0], finish[0], 100)
    path2 = path1
    print(curve_geodesic_MSE(manifold=Sphere(), x_curve=path1, y_curve=path2))