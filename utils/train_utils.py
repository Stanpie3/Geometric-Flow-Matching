import torch

def CondOT_flow(x0, x1, t):
  xt = (1-t[:,None,None])*x0 + t[:,None,None]*x1
  return xt

def CondOT_ut(x0, x1, t):
  cond_ut = x1-x0
  return cond_ut

def validate(vf_model, 
             validation_data, 
             device='cpu'):
  for batch_eval in validation_data:
    obs, a1 = batch_eval
    obs, a1 = obs.to(device), a1.to(device)

    a0 = torch.randn(a1.shape[0], 1, a1.shape[2])
    a0 = a0.repeat(1, a1.shape[1], 1).to(device)
    t = torch.rand(a1.shape[0]).to(device)

    a_t = CondOT_flow(a0, a1, t)
    da_t = CondOT_ut(a0, a1, t)

    loss = torch.pow(vf_model(obs=obs, x=a_t, t=t) - da_t, 2).mean()

    return loss.item()